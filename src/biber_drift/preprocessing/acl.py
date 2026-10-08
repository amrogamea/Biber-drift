"""Cleaning for ACL papers extracted from S2ORC.

PDF extraction leaves artifacts that inflate or distort Biber feature counts: sentences
glued together without a space, citations, URLs, footnote markers, table cells read as
prose, licence footers and arXiv stamps. Each regular expression below was checked against
the full ACL corpus; see the comments for what it deliberately leaves alone.
"""

from __future__ import annotations

import re

from ..corpus import TEXT
from .base import CorpusCleaner, DocumentFilter, Step, TextStep, register_cleaner

# --------------------------------------------------------------------------------------------
# Sentences glued together without a space: "corpus.This" -> "corpus. This".
# --------------------------------------------------------------------------------------------

_END = r"""([.!?]["')\]]*)"""  # sentence terminator plus any closing quotes/brackets

# After an ordinary word ending ("corpus.", "2003).", "sys1.") any capital starts a new
# sentence, including acronyms: "language.NLU" -> "language. NLU".
_AFTER_WORD = re.compile(r"(?<=[^\sA-Z.])" + _END + r"(?=[A-Z])")

# After a capital ("PEFT.", "U.S.") only split before a real word ("We", "The") or a
# standalone "A"/"I", so abbreviations like "U.S.A." and "TWITCH.TV" stay intact.
_AFTER_CAPITAL = re.compile(r"(?<=[A-Z])" + _END + r"(?=[A-Z][a-z]|[AI](?![\w.]))")

# Only tokens containing a glued terminator are passed to the Python-level check below.
_CANDIDATE = re.compile(r"""\S*[.!?]["')\]]*[A-Z]\S*""")

# Tokens where ".X" belongs to a URL, e-mail, file/identifier or arXiv category.
_SKIP_TOKEN = re.compile(
    r"://|www\.|@|_|arXiv|github|huggingface"
    r"|\.(?:com|org|net|edu|io|ai|gov)\b"
    r"|\b(?:cs|stat|math|eess|econ|physics|q-bio|q-fin)\.[A-Z]{2}\b"
)


def add_space_after_sentence_end(text: str) -> str:
    """'corpus.This' -> 'corpus. This', '2003).Such' -> '2003). Such'."""

    def fix_token(match: re.Match) -> str:
        token = match.group(0)
        if _SKIP_TOKEN.search(token):
            return token
        token = _AFTER_WORD.sub(r"\1 ", token)
        return _AFTER_CAPITAL.sub(r"\1 ", token)

    return _CANDIDATE.sub(fix_token, text)


_WHITESPACE = re.compile(r"\s{2,}")


def collapse_whitespace(text: str) -> str:
    """Joining empty IMRaD sections leaves runs of spaces; collapse them and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------------------------
# Boilerplate: licence footers (current and 2008-era wording), arXiv stamps, "\n" escapes.
# --------------------------------------------------------------------------------------------

_LICENCE = re.compile(
    r"This work is licen[cs]ed under a Creative Commons.{0,120}?Licen[cs]e\."
    r"(?:\s*Page numbers and proceedings footer are added by the organi[sz]ers\.)?"
    r"(?:\s*Licen[cs]e details:\s*https?:?\s*//\s*\S+)?"
    r"|(?:(?:©|\bc)\s*⃝?\s*)?(?:(?:19|20)\d{2}\.\s*)?Licensed under the Creative Commons"
    r".{0,150}?licen[cs]e(?:\s*\([^()]{0,100}\))?\.(?:\s*Some rights reserved\.)?",
    re.IGNORECASE,
)
_ARXIV_STAMP = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[[a-z\-]+\.[A-Z]{2}\])?"
    r"(?:\s*\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})?"
)
_ESCAPED_NEWLINE = re.compile(r"\\n")


def remove_boilerplate(text: str) -> str:
    return _ESCAPED_NEWLINE.sub(" ", _ARXIV_STAMP.sub("", _LICENCE.sub("", text)))


# --------------------------------------------------------------------------------------------
# URLs and e-mails; a trailing sentence full stop is left in place. Also catches URLs split by
# extraction spaces: "http:// chasen. aist-nara. ac. jp/".
# --------------------------------------------------------------------------------------------

_URL = re.compile(r"(?:https?://\s?|www\.)\S+?(?:\.\s(?=[a-z])\S+?)*(?=[.,;:)\]]*(?:\s|$))")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def remove_urls(text: str) -> str:
    return _EMAIL.sub("", _URL.sub("", text))


# --------------------------------------------------------------------------------------------
# Citations. A parenthesised group is split on ";" and every part that looks like an
# author-year reference ("Devlin et al., 2019", "e.g., Smith and Jones 2020a") is dropped, so
# "(BERT; Devlin et al., 2019)" becomes "(BERT)". Narrative citations keep the author as the
# sentence subject and only lose the year: "Devlin et al. (2019) show" -> "Devlin et al. show".
# --------------------------------------------------------------------------------------------

_YEAR = r"(?:19|20)\d{2}[a-z]?"
_PARENS = re.compile(r"\(([^()]{1,400})\)")
_CITATION_PART = re.compile(
    r"^\s*(?:(?:e\.g\.|i\.e\.|see(?: also)?|cf\.|also)[,\s]*)?"
    r"[A-ZÀ-ɏ][^;]*?,?\s*" + _YEAR + r"(?:\s*,\s*" + _YEAR + r")*"
    r"(?:,?\s*(?:inter alia|among others|for example|p+\.\s*\d+(?:[-–]\d+)?))?\s*$"
)
_YEAR_ONLY = re.compile(r"^\s*" + _YEAR + r"(?:\s*[,;]\s*" + _YEAR + r")*\s*$")
_NUMERIC_CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def remove_citations(text: str) -> str:
    def fix_parens(match: re.Match) -> str:
        inner = match.group(1)
        if _YEAR_ONLY.match(inner):
            return ""
        parts = inner.split(";")
        kept = [part for part in parts if not _CITATION_PART.match(part)]
        if len(kept) == len(parts):
            return match.group(0)
        return f"({';'.join(kept).strip()})" if kept else ""

    return _NUMERIC_CITATION.sub("", _PARENS.sub(fix_parens, text))


# Footnote numbers glued to punctuation: "available.3" -> "available."
_FOOTNOTE = re.compile(r"(?<=[a-z][.,;:])\d{1,2}\b")

# Affiliation / footnote symbols.
_SYMBOLS = re.compile("[†‡§⇤∗]")

# Table cells extracted as prose: five or more numbers in a row.
_NUMBER = r"[-+−]?\d+(?:[.,]\d+)?%?"
_NUMBER_RUN = re.compile(r"(?:" + _NUMBER + r"\s+){4,}" + _NUMBER + r"(?!\S)")

# Tidy-up after removals.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_EMPTY_PARENS = re.compile(r"\(\s*[,;]?\s*\)")


def tidy(text: str) -> str:
    text = _EMPTY_PARENS.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


@register_cleaner
class ACLCleaner(CorpusCleaner):
    """Cleaner for S2ORC ACL papers.

    Args:
        min_words: drop papers with fewer whitespace-separated words after cleaning.
        disable: step keys to skip.
    """

    name = "acl"

    def __init__(self, min_words: int = 200, disable: list[str] | None = None) -> None:
        self.min_words = min_words
        super().__init__(disable)

    def steps(self) -> list[Step]:
        return [
            TextStep("sentence_spacing", "missing space after sentence end", add_space_after_sentence_end),
            TextStep("whitespace", "collapse whitespace", collapse_whitespace),
            TextStep("boilerplate", "boilerplate (licence, arXiv stamp, \\n)", remove_boilerplate),
            TextStep("urls", "URLs and e-mails", remove_urls),
            TextStep("citations", "citations", remove_citations),
            TextStep("footnotes", "footnote markers", lambda t: _FOOTNOTE.sub("", t)),
            TextStep("symbols", "symbol markers", lambda t: _SYMBOLS.sub("", t)),
            TextStep("number_runs", "table number runs", lambda t: _NUMBER_RUN.sub(" ", t)),
            TextStep("tidy", "tidy spaces / punctuation", tidy),
            DocumentFilter("duplicates", "duplicate texts",
                           lambda docs: ~docs[TEXT].duplicated(keep="first")),
            DocumentFilter("short", f"documents under {self.min_words} words",
                           lambda docs: docs[TEXT].str.split().str.len() >= self.min_words),
        ]
