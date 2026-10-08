"""Cleaning for AO3 chapters.

Removes scraping and formatting artifacts before the analysis window is cut: chapters
duplicated across fandom files, podfics, the scraped ``Chapter Text`` page heading, author
notes and metadata headers, scene-break lines, URLs/emoji and markdown emphasis.

Known scraper artifact that is deliberately *kept*: the scraper puts a line break around every
inline (italic) span, so a sentence like ``ascetically *pleasing*.`` arrives as
``ascetically\\npleasing\\n.``. Lone ``.``, ``"`` or ``,`` lines are therefore pieces of real
sentences and are never treated as separators.
"""

from __future__ import annotations

import re

import pandas as pd

from ..corpus import TEXT
from .base import CorpusCleaner, DocumentFilter, Step, TextStep, register_cleaner

# Podfics: works whose "text" is an audio recording's links and file sizes. "Podfic Welcome"
# and "Podfic Available" mark ordinary text works and are deliberately not matched.
_PODFIC_TAG = re.compile(
    r"^(?:Podfic|Podfic & Podficced Works|Embedded Audio|Multivoice and Collaborative Podfic"
    r"|Podfic Length:.*|Audio Format:.*|Podfic Read by.*)$",
    re.IGNORECASE,
)
_PODFIC_TEXT = re.compile(r"(?i)right click to save|\.mp3\b|\bm4b\b")

# The AO3 page heading scraped in front of most chapters.
_CHAPTER_TEXT = re.compile(r"^\s*Chapter Text\s*")

# Author notes, disclaimers and metadata headers. At the start of a chapter every leading line
# that is a header ("Author's notes:", "Summary: ...", "Chapter 3", "Rating: G") is dropped; a
# label with nothing after it ("Author's notes:") also takes the paragraph that follows it.
_HEADER_LINE = re.compile(
    r"^\s*(?:a/n|an|author'?s?'? notes?|notes?|disclaimer|summary|rating|warnings?|pairings?"
    r"|characters|prompt|posted|title|beta(?:'?d)?(?: by)?|dedication|word count|current mood"
    r"|current music|fandom|spoilers?)\s*[:\-–—]\s*(.*)$",
    re.IGNORECASE,
)
_NUMBER_WORD = (
    r"(?:\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ]?[a-z]+)?)"
)
# "Chapter 23", "CHAPTER 7", "Part Three: The Forging", "Prologue" -- but not "Part of him knew."
_HEADING_LINE = re.compile(
    rf"^\s*(?:(?:chapter|ch\.|part)\s*{_NUMBER_WORD}\b|prologue\b|epilogue\b|interlude\b)"
    r"[^\n]{0,60}$",
    re.IGNORECASE,
)
# Anywhere in the text: paragraphs that open with an author-note marker.
_NOTE_PARAGRAPH = re.compile(r"(?im)^\s*(?:a/n|author'?s?'? notes?|disclaimer)\b[^\n]*(?:\n|$)")
# Trailing sign-offs.
_SIGN_OFF = re.compile(
    r"(?im)^\s*(?:tbc|to be continued|thanks? (?:you )?for reading|please (?:review|comment"
    r"|leave (?:a )?(?:kudos|comment|review)))\b[^\n]*(?:\n|$)"
)

# Scene-break lines: three or more separator symbols ("***", "~~~", "* * *", "-----") or
# decorative letter patterns ("~o~", "oOo", "-o0o-"). Lines holding sentence punctuation or
# quote marks are kept (see the module docstring).
_SEPARATOR_SYMBOL = "[^\\w\\s.,;:!?\"'“”‘’()\\[\\]…]"
_SCENE_BREAK_LINE = re.compile(
    rf"^(?:{_SEPARATOR_SYMBOL}\s*){{3,}}$"
    rf"|^(?=.*(?:{_SEPARATOR_SYMBOL}|oOo|OoO|xXx|XxX))(?:{_SEPARATOR_SYMBOL}*[oOxX0]+{_SEPARATOR_SYMBOL}*)+$"
)

# URLs, e-mails, emoji and decorative symbols (with their variation selectors / joiners).
_URL = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:)\]]*(?:\s|$))")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_EMOJI = re.compile(
    r"[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿⃐-⃿︎️‍]"
)

# Markdown emphasis: keep the word, drop the asterisks ("*actually*" -> "actually").
_MARKDOWN = re.compile(r"(?<![\w*])\*{1,2}(?=[^\s*])([^*\n]{1,80}?)(?<=[^\s*])\*{1,2}(?![\w*])")

_BLANK_RUNS = re.compile(r"[ \t]{2,}")
_NEWLINE_RUNS = re.compile(r"\n{2,}")


def is_podfic(tags: str, text: str) -> bool:
    """True for audio works, from their AO3 tags or (fallback) download links in the text."""
    if any(_PODFIC_TAG.match(tag.strip()) for tag in tags.split(";")):
        return True
    return bool(_PODFIC_TEXT.search(text[:3000]))


def remove_chapter_heading(text: str) -> str:
    return _CHAPTER_TEXT.sub("", text)


def remove_leading_headers(text: str) -> str:
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or _HEADING_LINE.match(line):
            i += 1
            continue
        header = _HEADER_LINE.match(line)
        if not header:
            break
        i += 1
        # An empty label ("Author's notes:") owns the next paragraph unless that is a header too.
        if not header.group(1).strip() and i < len(lines) and not _HEADER_LINE.match(lines[i]):
            i += 1
    return "\n".join(lines[i:])


def remove_notes(text: str) -> str:
    text = remove_leading_headers(text)
    text = _NOTE_PARAGRAPH.sub("", text)
    return _SIGN_OFF.sub("", text)


def remove_scene_breaks(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n") if not _SCENE_BREAK_LINE.match(line.strip())
    )


def remove_urls_emoji(text: str) -> str:
    return _EMOJI.sub("", _EMAIL.sub("", _URL.sub("", text)))


def remove_markdown(text: str) -> str:
    return _MARKDOWN.sub(r"\1", text)


def tidy(text: str) -> str:
    text = _BLANK_RUNS.sub(" ", text)
    return _NEWLINE_RUNS.sub("\n", text).strip()


def _tags(documents: pd.DataFrame) -> pd.Series:
    columns = [c for c in ("additional_tags", "freeform_tags") if c in documents.columns]
    if not columns:
        return pd.Series("", index=documents.index)
    return documents[columns].fillna("").astype(str).agg(";".join, axis=1)


@register_cleaner
class AO3Cleaner(CorpusCleaner):
    """Cleaner for AO3 chapters.

    Args:
        dedupe_on: column identifying the same chapter across fandom files.
        disable: step keys to skip.
    """

    name = "ao3"

    def __init__(self, dedupe_on: str = "chapter_id", disable: list[str] | None = None) -> None:
        self.dedupe_on = dedupe_on
        super().__init__(disable)

    def _not_podfic(self, documents: pd.DataFrame) -> pd.Series:
        tags = _tags(documents)
        flags = [is_podfic(t, s) for t, s in zip(tags, documents[TEXT])]
        return ~pd.Series(flags, index=documents.index)

    def steps(self) -> list[Step]:
        return [
            DocumentFilter("duplicates", "duplicate chapters (same chapter_id)",
                           lambda docs: ~docs[self.dedupe_on].duplicated(keep="first")),
            DocumentFilter("podfics", "podfic chapters", self._not_podfic),
            TextStep("chapter_heading", "'Chapter Text' heading", remove_chapter_heading),
            TextStep("notes", "author notes / headers / sign-offs", remove_notes),
            TextStep("scene_breaks", "scene-break lines", remove_scene_breaks),
            TextStep("urls_emoji", "URLs, e-mails, emoji", remove_urls_emoji),
            TextStep("markdown", "markdown asterisks", remove_markdown),
            TextStep("tidy", "tidy whitespace", tidy),
        ]
