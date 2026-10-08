"""Extract ACL Anthology papers from the Semantic Scholar S2ORC release.

Two phases:

* **extract** -- list the S2ORC shards, download each once, keep the ACL papers of every
  requested year in the same pass and append them to ``Dataset_Raw_<year>.csv``. Finished shards
  are recorded in ``_done_shards.txt``, so a crash loses at most one shard of work.
* **backfill** -- fill title/authors/year/venue/citation count from the Graph API, normalise
  the text and write ``Dataset_<year>.csv`` (the input of the ``acl`` loader).

Requires a Semantic Scholar API key in the ``S2_API_KEY`` environment variable (or
``SEMANTIC_SCHOLAR_API_KEY``). Never hard-code the key.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ..logging_utils import get_logger

logger = get_logger(__name__)

DATASET_URL = "https://api.semanticscholar.org/datasets/v1/release/{release}/dataset/s2orc"
GRAPH_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields={fields}"
GRAPH_FIELDS = "title,authors,year,venue,citationCount,externalIds"

TARGET_KEYWORDS = {
    "Introduction": ["introduction"],
    "Methods": ["method", "methodology", "approach", "experimental", "setup"],
    "Results": ["result", "results", "findings"],
    "Discussion": ["discussion", "conclusion", "limitations"],
}


def api_key_from_env() -> str:
    key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not key:
        raise SystemExit("Set S2_API_KEY (see .env.example) before running the ACL acquisition.")
    return key


# ------------------------------------------------------------------------------------------
# ACL id <-> year
# ------------------------------------------------------------------------------------------

def year_to_venue_pattern(year: int) -> re.Pattern | None:
    """Regex matching only ACL ids of ``year``.

    Old scheme (2000-2019): ``<Letter><YY>-<number>``, e.g. ``C00-1031``.
    New scheme (2020-2026): ``20<YY>.<slug>``, e.g. ``2020.acl-main.1``.
    """
    yy = year - 2000
    if 0 <= yy <= 19:
        return re.compile(rf"^[A-Z]{yy:02d}-\d+")
    if 20 <= yy <= 26:
        return re.compile(rf"^20{yy:02d}\..+")
    return None


def parse_acl_id_year(acl_id: str | None) -> int | None:
    """The year an ACL id encodes (inverse of :func:`year_to_venue_pattern`)."""
    if not acl_id:
        return None
    match = re.match(r"^20(\d{2})\.", acl_id) or re.match(r"^[A-Z](\d{2})-\d+", acl_id)
    return 2000 + int(match.group(1)) if match else None


# ------------------------------------------------------------------------------------------
# Per-paper extraction
# ------------------------------------------------------------------------------------------

def _parse_annotation(value):
    """S2ORC annotations are JSON-encoded lists of ``{start, end}`` spans."""
    if value is None:
        raise TypeError("missing annotation")
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)  # safe: literals only, never executes code


def extract_sections(paper: dict) -> dict | None:
    """Introduction/Methods/Results/Discussion text, or None if the record lacks annotations."""
    try:
        content = paper["content"]
        text = content["text"]
        annotations = content["annotations"]
        headers = sorted(_parse_annotation(annotations["sectionheader"]), key=lambda h: int(h["start"]))
        paragraphs = _parse_annotation(annotations["paragraph"])
    except (KeyError, TypeError, ValueError, SyntaxError):
        return None

    extracted = {name: "" for name in TARGET_KEYWORDS}
    for i, header in enumerate(headers):
        start, end = int(header["start"]), int(header["end"])
        title = text[start:end].lower()
        section = next((name for name, keywords in TARGET_KEYWORDS.items()
                        if any(k in title for k in keywords)), None)
        if section is None:
            continue
        boundary = int(headers[i + 1]["start"]) if i + 1 < len(headers) else len(text)
        parts = [text[int(p["start"]):int(p["end"])] for p in paragraphs
                 if int(p["start"]) >= end and int(p["end"]) <= boundary]
        extracted[section] += "\n".join(parts) + "\n"
    return {name: value.strip() for name, value in extracted.items()}


def extract_abstract(paper: dict) -> str:
    try:
        spans = _parse_annotation(paper["content"]["annotations"]["abstract"])
        text = paper["content"]["text"]
        return "\n".join(text[int(s["start"]):int(s["end"])] for s in spans)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return ""


def process_shard(shard_path: Path, patterns: dict[int, re.Pattern]) -> dict[int, list[dict]]:
    """ACL papers per year from one ``.gz`` shard (papers need >= 1 non-empty section)."""
    records: dict[int, list[dict]] = {year: [] for year in patterns}
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            paper = json.loads(line)
            acl_id = (paper.get("externalids") or {}).get("acl")
            if not acl_id:
                continue
            year = next((y for y, pattern in patterns.items() if pattern.match(acl_id)), None)
            if year is None:
                continue
            sections = extract_sections(paper)
            if not sections or not any(sections.values()):
                continue
            records[year].append({
                "id": acl_id, "title": paper.get("title", ""),
                "authors": str(paper.get("authors", [])), "year": year,
                "venue": paper.get("venue", ""), "citation_count": None,
                "abstract": extract_abstract(paper),
                "introduction": sections["Introduction"], "methods": sections["Methods"],
                "results": sections["Results"], "discussion": sections["Discussion"],
            })
    return records


def normalize_text(text: str) -> str:
    """Flatten whitespace, drop spaces before punctuation, NFKC-normalise."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,?!'])", r"\1", text)
    return unicodedata.normalize("NFKC", text)


# ------------------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------------------

class ExpiredURLError(Exception):
    """A presigned shard URL returned 400/403; long runs outlive them, so refresh and retry."""


def shard_filename(url: str) -> str:
    name = url.split("?")[0].split("/")[-1]
    return name if name.endswith(".gz") else name + ".gz"


class S2ORCClient:
    def __init__(self, api_key: str, release_id: str, session=requests) -> None:
        self.api_key = api_key
        self.release_id = release_id
        self.session = session

    @property
    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def shard_map(self, max_attempts: int = 10) -> dict[str, str]:
        """Stable shard filename -> current (presigned) URL."""
        url = DATASET_URL.format(release=self.release_id)
        for attempt in range(1, max_attempts + 1):
            response = self.session.get(url, headers=self.headers)
            if response.status_code == 200:
                files = response.json().get("files", [])
                if not files:
                    raise RuntimeError("Release returned zero shard files -- check release id / API key.")
                return {shard_filename(u): u for u in files}
            logger.warning("Shard listing failed (HTTP %d), attempt %d/%d",
                           response.status_code, attempt, max_attempts)
            time.sleep(2 ** min(attempt, 6))
        raise RuntimeError("Could not list S2ORC shards")

    def download(self, url: str, cache_dir: Path, timeout: int = 120, max_retries: int = 3) -> Path:
        path = cache_dir / shard_filename(url)
        if path.is_file():
            return path
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, stream=True, timeout=timeout)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (400, 403):
                raise ExpiredURLError(f"{path.name}: HTTP {response.status_code}")
            response.raise_for_status()
            partial = path.with_suffix(path.suffix + ".part")
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
            os.replace(partial, path)
            return path
        raise RuntimeError(f"{path.name}: download failed after {max_retries} retries: {last_error}")

    def backfill_metadata(self, frame: pd.DataFrame, batch_size: int = 500, max_retries: int = 5,
                          pause: float = 1.0) -> pd.DataFrame:
        """Fill authors/year/venue/citation_count in place from the Graph API (by ACL id)."""
        url = GRAPH_BATCH_URL.format(fields=GRAPH_FIELDS)
        ids = frame.index.tolist()
        for start in tqdm(range(0, len(ids), batch_size), desc="Metadata batches"):
            batch = ids[start:start + batch_size]
            response = None
            for attempt in range(max_retries):
                response = self.session.post(url, headers=self.headers, json={"ids": [f"ACL:{i}" for i in batch]})
                if response.status_code != 429:
                    break
                time.sleep(pause * 2 ** attempt)
            if response is None or response.status_code != 200:
                logger.warning("Metadata batch %d-%d failed (HTTP %s)", start, start + len(batch),
                               getattr(response, "status_code", "?"))
                continue
            for paper in response.json():
                if paper is None:
                    continue
                acl_id = (paper.get("externalIds") or {}).get("ACL")
                if not acl_id or acl_id not in frame.index:
                    continue
                frame.at[acl_id, "authors"] = ", ".join(a.get("name") for a in paper.get("authors", []))
                frame.at[acl_id, "year"] = paper.get("year")
                frame.at[acl_id, "venue"] = paper.get("venue")
                frame.at[acl_id, "citation_count"] = paper.get("citationCount")
            time.sleep(pause)
        return frame


# ------------------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------------------

def _process_one_shard(task: tuple) -> tuple[str, dict[int, list[dict]]]:
    filename, url, patterns, cache_dir, api_key, release_id = task
    client = S2ORCClient(api_key, release_id)
    try:
        try:
            path = client.download(url, cache_dir)
        except ExpiredURLError:
            fresh = client.shard_map().get(filename)
            if fresh is None:
                logger.warning("%s gone after URL refresh, skipping", filename)
                return filename, {y: [] for y in patterns}
            path = client.download(fresh, cache_dir)
    except Exception as exc:  # one bad shard must not kill a multi-hour run
        logger.warning("%s: download failed (%s), skipping", filename, exc)
        return filename, {y: [] for y in patterns}
    records = process_shard(path, patterns)
    path.unlink(missing_ok=True)
    return filename, records


class ACLExtractor:
    """Resumable two-phase S2ORC -> ACL dataset builder."""

    def __init__(self, years: list[int], output_dir: Path, cache_dir: Path, release_id: str,
                 api_key: str, max_workers: int = 16) -> None:
        self.years = years
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.release_id = release_id
        self.api_key = api_key
        self.max_workers = max_workers
        self.client = S2ORCClient(api_key, release_id)

    @property
    def manifest(self) -> Path:
        return self.output_dir / "_done_shards.txt"

    def raw_path(self, year: int) -> Path:
        return self.output_dir / f"Dataset_Raw_{year}.csv"

    def final_path(self, year: int) -> Path:
        return self.output_dir / f"Dataset_{year}.csv"

    def extract(self, limit_shards: int | None = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        patterns = {}
        for year in self.years:
            pattern = year_to_venue_pattern(year)
            if pattern is None:
                logger.warning("No ACL id convention known for %d, skipping", year)
            else:
                patterns[year] = pattern

        shard_map = self.client.shard_map()
        names = list(shard_map)[:limit_shards] if limit_shards else list(shard_map)
        done = set(self.manifest.read_text().split()) if self.manifest.exists() else set()
        todo = [name for name in names if name not in done]
        logger.info("%d shard(s), %d done, %d remaining, years %s", len(names), len(done), len(todo), sorted(patterns))

        tasks = [(n, shard_map[n], patterns, self.cache_dir, self.api_key, self.release_id) for n in todo]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for filename, records in tqdm(pool.map(_process_one_shard, tasks), total=len(tasks), desc="Shards"):
                for year, rows in records.items():
                    if rows:
                        path = self.raw_path(year)
                        pd.DataFrame(rows).set_index("id").to_csv(path, mode="a", header=not path.exists())
                with open(self.manifest, "a") as handle:
                    handle.write(filename + "\n")

    def backfill(self) -> None:
        for year in self.years:
            raw = self.raw_path(year)
            if not raw.exists():
                logger.warning("%s not found, skipping %d", raw, year)
                continue
            frame = pd.read_csv(raw, index_col="id")
            if frame.empty:
                continue
            for column in ["title", "authors", "venue", "year", "citation_count"]:
                frame[column] = frame[column].astype("object")
            self.client.backfill_metadata(frame)
            for column in ["title", "abstract", "introduction", "methods", "results", "discussion"]:
                frame[column] = frame[column].fillna("").astype(str).map(normalize_text)
            frame.to_csv(self.final_path(year))
            logger.info("%d: %d papers -> %s", year, len(frame), self.final_path(year))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--years", default="2000-2026", help='Years, e.g. "2000-2026" or "2000,2019,2024"')
    parser.add_argument("--release-id", default="2026-07-07", help="S2ORC release id")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/acl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache/s2orc"))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit-shards", type=int, default=None, help="Only process the first N shards (testing)")
    parser.add_argument("--phase", choices=["extract", "backfill", "all"], default="all")


def _parse_year_list(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(","):
        start, _, end = part.strip().partition("-")
        years.update(range(int(start), int(end or start) + 1))
    return sorted(years)


def run(args: argparse.Namespace) -> None:
    extractor = ACLExtractor(_parse_year_list(args.years), args.output_dir, args.cache_dir,
                             args.release_id, api_key_from_env(), args.max_workers)
    if args.phase in ("extract", "all"):
        extractor.extract(args.limit_shards)
    if args.phase in ("backfill", "all"):
        extractor.backfill()
