"""Scrape AO3 fandom tags into a CSV with one row per posted chapter.

Be a polite client: the defaults wait 4-7 s between requests, stop immediately when AO3 or
Cloudflare signals automated traffic (HTTP 403/503/522/525), and checkpoint after every listing
page so a stopped run resumes where it left off when re-run with the same arguments. Check
AO3's Terms of Service before scraping and do not redistribute the collected texts.

The chapter year used by the analysis comes from each chapter's publication date in the work's
chapter index (``/works/<id>/navigate``) -- the date on a listing is only the work's most recent
update. Dates are author-set, not verified composition timestamps.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from ..logging_utils import get_logger

logger = get_logger(__name__)

BASE = "https://archiveofourown.org"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
# Lets anonymous browsing include Explicit/Mature works in listings (what any visitor sees after
# clicking through AO3's content notice); without it rating distributions would be biased.
COOKIES = {"view_adult": "true"}
BLOCKED_STATUSES = (403, 503, 522, 525)

METADATA_FIELDNAMES = [
    "work_id", "chapter_id", "chapter_number", "chapter_title", "chapter_date",
    "chapter_word_count", "title", "authors", "fandoms", "rating", "warnings",
    "category", "status", "listing_date", "relationships", "characters", "additional_tags",
    "freeform_tags", "language", "word_count", "chapters_posted",
    "chapters_expected", "comments", "kudos", "bookmarks", "hits",
]
FIELDNAMES = METADATA_FIELDNAMES + ["story_text"]

# AO3 "Additional Tags" are freeform author labels. These patterns catch common labels for
# pornographic or explicit sexual material; the negation check keeps tags such as "No Smut".
SEXUAL_CONTENT_TAG_PATTERNS = (
    re.compile(r"\b(?:porn|pornography|pornographic)\b", re.IGNORECASE),
    re.compile(r"\bsmut\b", re.IGNORECASE),
    re.compile(r"\b(?:pwp|plot\s+what\s+plot)\b", re.IGNORECASE),
    re.compile(r"\berotica?\b", re.IGNORECASE),
    re.compile(r"\b(?:sex|sexual)(?:\s+(?:content|acts?|scenes?|themes?))?\b", re.IGNORECASE),
    re.compile(r"\bexplicit(?:\s+(?:sex|sexual|smut|porn(?:ography|ographic)?))?\b", re.IGNORECASE),
)
NEGATED_SEXUAL_CONTENT_TAG = re.compile(
    r"\b(?:no|without|non[- ]?)\s+(?:porn|pornography|pornographic|smut|"
    r"pwp|plot\s+what\s+plot|erotica?|sex|sexual|explicit)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------------------------------

def sexual_content_tag(additional_tags: str | None) -> str | None:
    """The first sexual-content Additional Tag, or None."""
    for tag in (additional_tags or "").split(";"):
        tag = tag.strip()
        if not tag or NEGATED_SEXUAL_CONTENT_TAG.search(tag):
            continue
        if any(pattern.search(tag) for pattern in SEXUAL_CONTENT_TAG_PATTERNS):
            return tag
    return None


def inclusion_reason(row: dict, english_only: bool = True, exclude_sexual_tags: bool = True) -> str | None:
    """None for a work to keep, otherwise a short exclusion reason."""
    if english_only and (row.get("language") or "").strip().casefold() != "english":
        return "non_english"
    if exclude_sexual_tags:
        matched = sexual_content_tag(row.get("additional_tags") or row.get("freeform_tags"))
        if matched:
            return f"sexual_tag:{matched}"
    return None


# ------------------------------------------------------------------------------------------
# Parsing (pure functions of HTML)
# ------------------------------------------------------------------------------------------

def _text(el):
    return el.get_text(strip=True) if el else None


def _label(el):
    """Icons carry their label in ``title`` (tooltip); fall back to visible text."""
    if el is None:
        return None
    return (el.get("title") or "").strip() or _text(el)


def _int(text):
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def _tag_group(blurb, li_class):
    return "; ".join(a.get_text(strip=True) for a in blurb.select(f"li.{li_class} a.tag"))


def parse_date(text):
    """AO3 listing dates look like '14 Jul 2026'. Returns ``(date, iso_string)``."""
    if not text:
        return None, None
    try:
        d = datetime.strptime(text.strip(), "%d %b %Y")
        return d, d.strftime("%Y-%m-%d")
    except ValueError:
        return None, text


def parse_chapter_date(text):
    if not text:
        return None, None
    cleaned = text.strip().strip("()")
    for date_format in ("%Y-%m-%d", "%d %b %Y"):
        try:
            d = datetime.strptime(cleaned, date_format)
            return d, d.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None, cleaned


def chapter_id_from(value):
    """AO3 chapter id from a chapter URL or HTML element id."""
    if not value:
        return None
    match = re.search(r"/chapters/(\d+)", value) or re.search(r"(?:^|[-_])chapter[-_]?(\d+)", value)
    return match.group(1) if match else None


def extract_chapter_index(html: str) -> list[dict]:
    """Chapter ids, titles and publication dates from a work's full-page chapter index."""
    soup = BeautifulSoup(html, "html.parser")
    chapters = []
    for number, item in enumerate(soup.select("ol.chapter.index > li"), start=1):
        link = item.select_one('a[href*="/chapters/"]') or item.select_one("a")
        if link is None:
            continue
        date_obj, chapter_date = parse_chapter_date(_text(item.select_one("span.datetime")))
        chapters.append({
            "chapter_id": chapter_id_from(link.get("href")),
            "chapter_number": number,
            "chapter_title": link.get_text(" ", strip=True) or None,
            "chapter_date": chapter_date,
            "_date_obj": date_obj,
        })
    return chapters


def _direct_chapter_body(chapter):
    """A chapter's prose element, excluding notes/prefaces."""
    for child in chapter.find_all("div", recursive=False):
        if "userstuff" in (child.get("class") or []):
            return child
    return None


def extract_chapter_texts(html: str) -> list[dict]:
    """``[{chapter_id, story_text}, ...]`` from a full-work page.

    Note: text is joined with ``"\\n"`` between every HTML node, so inline spans (italics) end
    up on their own lines and the hidden "Chapter Text" landmark heading is included. The AO3
    cleaner handles both; the format is kept so new scrapes match existing data.
    """
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.select("#chapters > div.chapter")
    bodies = ([(c.get("id"), _direct_chapter_body(c)) for c in containers] if containers
              else [(b.get("id"), b) for b in soup.select("#chapters > div.userstuff")])
    return [
        {"chapter_id": chapter_id_from(element_id), "story_text": body.get_text("\n", strip=True) or None}
        for element_id, body in bodies if body is not None
    ]


def parse_blurb(blurb) -> tuple[dict, datetime | None]:
    """One ``<li class="work blurb group">`` into a metadata dict and its listing date."""
    work_id = (blurb.get("id") or "").replace("work_", "")
    heading = blurb.select_one("h4.heading")
    title, authors = None, []
    if heading:
        title = _text(heading.select_one('a[href^="/works/"]'))
        authors = [a.get_text(strip=True) for a in heading.select("a[rel=author]")]
    authors = authors or ["Anonymous"]

    date_obj, listing_date = parse_date(_text(blurb.select_one("p.datetime")))
    chapters_posted = chapters_expected = None
    chapters_text = _text(blurb.select_one("dd.chapters"))
    if chapters_text and "/" in chapters_text:
        posted, expected = chapters_text.split("/", 1)
        chapters_posted = _int(posted)
        chapters_expected = _int(expected) if expected.strip() != "?" else None

    freeforms = _tag_group(blurb, "freeforms")
    row = {
        "work_id": work_id,
        "title": title,
        "authors": "; ".join(authors),
        "fandoms": "; ".join(a.get_text(strip=True) for a in blurb.select("h5.fandoms a.tag")),
        # Substring class matches: AO3 icon spans use compound classes like "rating-teen-and-up".
        "rating": _label(blurb.select_one('span[class*="rating-"]')),
        "warnings": _tag_group(blurb, "warnings"),
        "category": _label(blurb.select_one('span[class*="category-"]')),
        "status": _label(blurb.select_one("span.iswip") or blurb.select_one('span[class*="complete-"]')),
        "listing_date": listing_date,
        "relationships": _tag_group(blurb, "relationships"),
        "characters": _tag_group(blurb, "characters"),
        "additional_tags": freeforms,
        "freeform_tags": freeforms,
        "language": _text(blurb.select_one("dd.language")),
        "word_count": _int(_text(blurb.select_one("dd.words"))),
        "chapters_posted": chapters_posted,
        "chapters_expected": chapters_expected,
        "comments": _int(_text(blurb.select_one("dd.comments"))),
        "kudos": _int(_text(blurb.select_one("dd.kudos"))),
        "bookmarks": _int(_text(blurb.select_one("dd.bookmarks"))),
        "hits": _int(_text(blurb.select_one("dd.hits"))),
    }
    return row, date_obj


def chapter_rows(work_row: dict, chapters: list[dict], chapter_texts: list[dict] | None = None) -> list[dict]:
    """Combine work metadata, chapter dates and chapter prose into CSV rows."""
    text_by_id = {t["chapter_id"]: t["story_text"] for t in (chapter_texts or []) if t.get("chapter_id")}
    text_by_position = [t.get("story_text") for t in (chapter_texts or [])]
    rows = []
    for position, chapter in enumerate(chapters):
        row = work_row.copy()
        row.update({
            "chapter_id": chapter.get("chapter_id"),
            "chapter_number": chapter["chapter_number"],
            "chapter_title": chapter.get("chapter_title"),
            "chapter_date": chapter.get("chapter_date"),
        })
        if chapter_texts is not None:
            story_text = text_by_id.get(chapter.get("chapter_id"))
            # Position fallback for older full-work markup without chapter ids.
            if story_text is None and position < len(text_by_position):
                story_text = text_by_position[position]
            row["story_text"] = story_text
            row["chapter_word_count"] = len(re.findall(r"\b\w+\b", story_text)) if story_text else None
        rows.append(row)
    return rows


def _chapter_text_for_row(row: dict, chapter_texts: list[dict]) -> str | None:
    wanted = str(row.get("chapter_id") or "").strip()
    if wanted:
        for chapter in chapter_texts:
            if chapter.get("chapter_id") == wanted:
                return chapter.get("story_text")
    try:
        position = int(row.get("chapter_number")) - 1
    except (TypeError, ValueError):
        return None
    return chapter_texts[position].get("story_text") if 0 <= position < len(chapter_texts) else None


# ------------------------------------------------------------------------------------------
# Scraper
# ------------------------------------------------------------------------------------------

@dataclass
class ScrapeSettings:
    fandom_tag: str
    start_year: int = 2016
    end_year: int = 2026
    output: Path | None = None
    min_delay: float = 4.0
    max_delay: float = 7.0
    stop_streak: int = 0
    max_pages: int = 300
    include_text: bool = True
    text_retries: int = 3
    english_only: bool = True
    exclude_sexual_tags: bool = True

    def __post_init__(self) -> None:
        if self.output is None:
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", self.fandom_tag).strip("_").lower()
            suffix = "_chapters_with_text.csv" if self.include_text else "_chapters_metadata.csv"
            self.output = Path("data/raw/ao3") / f"ao3_{safe}{suffix}"
        self.output = Path(self.output)

    @property
    def fieldnames(self) -> list[str]:
        return FIELDNAMES if self.include_text else METADATA_FIELDNAMES


class AO3Scraper:
    """Resumable, rate-limited AO3 scraper for one fandom tag."""

    def __init__(self, settings: ScrapeSettings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.session.cookies.update(COOKIES)

    # -- HTTP --------------------------------------------------------------------------------

    def _pause(self) -> None:
        time.sleep(random.uniform(self.settings.min_delay, self.settings.max_delay))

    def _get(self, url: str, params: dict | None, what: str, max_retries: int = 3):
        """GET with retries for network errors / 429. Returns ``(response | None, blocked)``."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as exc:
                logger.warning("[network error] %s: %s (attempt %d/%d)", what, exc, attempt, max_retries)
                time.sleep(10 * attempt)
                continue
            if response.status_code == 200:
                return response, False
            if response.status_code == 429:
                logger.warning("[429 rate-limited] %s: waiting %ds", what, 60 * attempt)
                time.sleep(60 * attempt)
                continue
            if response.status_code in BLOCKED_STATUSES:
                logger.error("[HTTP %d] AO3/Cloudflare appears to be flagging this as automated "
                             "traffic. Stopping -- wait, raise --min-delay/--max-delay, then re-run "
                             "the same command to resume.", response.status_code)
                return None, True
            logger.warning("[HTTP %d] unexpected status for %s", response.status_code, what)
            return None, False
        return None, False

    def fetch_listing(self, page: int):
        """One listing page, newest works first (so pagination walks backwards in time)."""
        url = f"{BASE}/tags/{quote(self.settings.fandom_tag, safe='')}/works"
        params = {"page": page, "work_search[sort_column]": "created_at",
                  "work_search[sort_direction]": "desc"}
        return self._get(url, params, f"listing page {page}")

    def _fetch_with_content(self, url: str, params: dict | None, what: str, extract):
        """Fetch and extract, retrying pages that come back 200 but empty (interstitials)."""
        retries = self.settings.text_retries
        for attempt in range(1, retries + 1):
            response, blocked = self._get(url, params, what, retries)
            if response is None:
                return [], blocked
            content = extract(response.text)
            if content and (extract is not extract_chapter_texts
                            or any(c.get("story_text") for c in content)):
                return content, False
            if attempt < retries:
                logger.info("[empty %s] retrying (attempt %d/%d)", what, attempt, retries)
                self._pause()
        logger.warning("[empty %s] leaving it out after %d attempts", what, retries)
        return [], False

    def fetch_chapter_index(self, work_id: str):
        return self._fetch_with_content(f"{BASE}/works/{work_id}/navigate", None,
                                        f"chapter index of work {work_id}", extract_chapter_index)

    def fetch_chapter_texts(self, work_id: str):
        return self._fetch_with_content(f"{BASE}/works/{work_id}", {"view_full_work": "true"},
                                        f"text of work {work_id}", extract_chapter_texts)

    # -- main loop ---------------------------------------------------------------------------

    def scrape(self) -> None:
        s = self.settings
        s.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(f"{s.output}.checkpoint")
        done = Path(f"{s.output}.done")
        validate_output_schema(s.output, s.fieldnames)

        if done.exists():
            logger.info("%s exists -- a previous run already reached the end of this tag. Delete it "
                        "to re-run (e.g. after changing --start-year).", done)
            return

        start_page = _load_checkpoint(checkpoint) + 1
        if start_page > 1:
            logger.info("Resuming from page %d (checkpoint %s)", start_page, checkpoint)

        write_header = not s.output.exists()
        old_streak = written = pages = 0
        page, last_page = start_page, start_page - 1
        reached_end = blocked = False
        skipped = {"non_english": 0, "sexual_tag": 0, "unparseable_chapter_date": 0}

        with open(s.output, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=s.fieldnames)
            if write_header:
                writer.writeheader()

            while pages < s.max_pages:
                logger.info("Page %d ...", page)
                response, blocked = self.fetch_listing(page)
                if response is None:
                    break
                soup = BeautifulSoup(response.text, "html.parser")
                blurbs = soup.select("li.work.blurb.group")
                if not blurbs:
                    if soup.select_one("p.notice, div#no_results, #main h3.heading"):
                        logger.info("No more results -- reached the end of this fandom tag.")
                        reached_end = True
                    else:
                        logger.error("No work blurbs on this page: AO3's markup may have changed or "
                                     "this is a pagination-depth limit. Inspect the page by hand.")
                    break

                # Hold a page's rows until every work on it was fetched: if AO3 blocks us midway,
                # the checkpoint stays on the previous page and a resume cannot duplicate output.
                page_rows = []
                for blurb in blurbs:
                    row, date_obj = parse_blurb(blurb)
                    year = date_obj.year if date_obj else None
                    if year is not None:
                        old_streak = old_streak + 1 if year < s.start_year else 0

                    reason = inclusion_reason(row, s.english_only, s.exclude_sexual_tags)
                    if reason:
                        skipped[reason.split(":", 1)[0]] += 1
                        continue
                    # listing_date is the work's latest revision: if it is before the period no
                    # chapter can be in range; if it is later, older chapters may still qualify.
                    if year is not None and year < s.start_year:
                        continue

                    self._pause()
                    chapters, blocked = self.fetch_chapter_index(row["work_id"])
                    if blocked:
                        break
                    eligible = []
                    for chapter in chapters:
                        date = chapter.get("_date_obj")
                        if date is None:
                            skipped["unparseable_chapter_date"] += 1
                        elif s.start_year <= date.year <= s.end_year:
                            eligible.append(chapter)
                    if not eligible:
                        continue

                    texts = None
                    if s.include_text:
                        self._pause()
                        texts, blocked = self.fetch_chapter_texts(row["work_id"])
                        if blocked:
                            break
                    page_rows.extend(chapter_rows(row, eligible, texts))

                if blocked:
                    break
                writer.writerows(page_rows)
                handle.flush()
                written += len(page_rows)
                _save_checkpoint(checkpoint, page)
                last_page = page
                pages += 1

                if s.stop_streak and old_streak >= s.stop_streak:
                    logger.info("Seen %d consecutive works older than %d -- stopping.", old_streak, s.start_year)
                    reached_end = True
                    break
                page += 1
                self._pause()

        if reached_end:
            done.write_text("done\n")
        logger.info("Wrote %d chapter rows to %s (%d pages). Skipped %d non-English, %d sexual-tagged "
                    "works and %d chapters with unparseable dates.", written, s.output, pages,
                    skipped["non_english"], skipped["sexual_tag"], skipped["unparseable_chapter_date"])
        if not reached_end:
            logger.info("Checkpoint at page %d; re-run the same command to continue.", last_page)

    def retry_empty_story_text(self) -> None:
        """Re-fetch blank ``story_text`` cells of an existing CSV (atomic replace on success)."""
        s = self.settings
        validate_output_schema(s.output, FIELDNAMES)
        csv.field_size_limit(sys.maxsize)
        fd, temporary = tempfile.mkstemp(prefix=".ao3_text_retry_", suffix=".csv", dir=s.output.parent)
        os.close(fd)
        retried = recovered = 0
        active_work, active_texts = None, []
        try:
            with open(s.output, newline="", encoding="utf-8") as source, \
                    open(temporary, "w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDNAMES)
                writer.writeheader()
                for row in csv.DictReader(source):
                    text = row.get("story_text")
                    if not text or text.strip().lower() == "nan":
                        retried += 1
                        if row["work_id"] != active_work:
                            active_work = row["work_id"]
                            self._pause()
                            active_texts, blocked = self.fetch_chapter_texts(active_work)
                            if blocked:
                                logger.error("Stopping without changing the CSV.")
                                return
                        story_text = _chapter_text_for_row(row, active_texts)
                        if story_text:
                            row["story_text"] = story_text
                            row["chapter_word_count"] = len(re.findall(r"\b\w+\b", story_text))
                            recovered += 1
                    writer.writerow(row)
            os.replace(temporary, s.output)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        logger.info("Retried %d empty story_text rows; recovered %d.", retried, recovered)


def validate_output_schema(path: Path, fieldnames: list[str]) -> None:
    """Refuse to append rows with a different CSV schema."""
    if not Path(path).exists():
        return
    with open(path, newline="", encoding="utf-8") as handle:
        existing = next(csv.reader(handle), [])
    if existing != fieldnames:
        raise ValueError(f"{path} has a different CSV schema; choose a new --output path.")


def filter_existing_csv(source: Path, output: Path, english_only: bool = True,
                        exclude_sexual_tags: bool = True) -> None:
    """Write a filtered copy of an existing AO3 CSV without contacting AO3."""
    csv.field_size_limit(sys.maxsize)
    fd, temporary = tempfile.mkstemp(prefix=".ao3_filtered_", suffix=".csv", dir=Path(output).parent)
    os.close(fd)
    kept = 0
    skipped = {"non_english": 0, "sexual_tag": 0}
    try:
        with open(source, newline="", encoding="utf-8") as src, open(temporary, "w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            if not reader.fieldnames or "language" not in reader.fieldnames:
                raise ValueError(f"{source} has no header or no language column.")
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                reason = inclusion_reason(row, english_only, exclude_sexual_tags)
                if reason:
                    skipped[reason.split(":", 1)[0]] += 1
                    continue
                writer.writerow(row)
                kept += 1
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    logger.info("Filtered CSV written to %s: kept %d; removed %d non-English and %d sexual-tagged works.",
                output, kept, skipped["non_english"], skipped["sexual_tag"])


def _load_checkpoint(path: Path) -> int:
    if path.exists():
        content = path.read_text().strip()
        return int(content) if content else 0
    return 0


def _save_checkpoint(path: Path, page: int) -> None:
    path.write_text(str(page))


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fandom-tag", required=True,
                        help='Exact AO3 fandom tag, e.g. "The Hobbit - J. R. R. Tolkien". Check the '
                             "spelling at archiveofourown.org/tags/<name> first.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default: data/raw/ao3/ao3_<tag>_chapters_with_text.csv)")
    parser.add_argument("--min-delay", type=float, default=4.0, help="Minimum seconds between requests")
    parser.add_argument("--max-delay", type=float, default=7.0, help="Maximum seconds between requests")
    parser.add_argument("--stop-streak", type=int, default=0,
                        help="Stop after this many consecutive works older than --start-year (0 = off)")
    parser.add_argument("--max-pages", type=int, default=300,
                        help="Listing pages per run; re-run to continue from the checkpoint")
    parser.add_argument("--text-retries", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true", help="Do not fetch chapter text")
    parser.add_argument("--all-languages", action="store_true", help="Keep non-English works")
    parser.add_argument("--include-sexual-tags", action="store_true",
                        help="Keep works tagged as pornographic/explicit (excluded by default)")
    parser.add_argument("--retry-empty-text", action="store_true",
                        help="Only re-fetch blank story_text cells of an existing --output CSV")
    parser.add_argument("--filter-existing", action="store_true",
                        help="Only filter an existing --output CSV locally (no requests)")
    parser.add_argument("--filtered-output", type=Path, default=None)


def run(args: argparse.Namespace) -> None:
    settings = ScrapeSettings(
        fandom_tag=args.fandom_tag, start_year=args.start_year, end_year=args.end_year,
        output=args.output, min_delay=args.min_delay, max_delay=args.max_delay,
        stop_streak=args.stop_streak, max_pages=args.max_pages,
        include_text=not args.metadata_only, text_retries=args.text_retries,
        english_only=not args.all_languages, exclude_sexual_tags=not args.include_sexual_tags,
    )
    if args.filter_existing:
        output = args.filtered_output or settings.output.with_name(
            f"{settings.output.stem}_english_non_explicit{settings.output.suffix}")
        filter_existing_csv(settings.output, output, settings.english_only, settings.exclude_sexual_tags)
    elif args.retry_empty_text:
        AO3Scraper(settings).retry_empty_story_text()
    else:
        AO3Scraper(settings).scrape()
