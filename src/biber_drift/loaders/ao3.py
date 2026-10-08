"""Archive of Our Own (AO3) chapters, as written by ``biber-drift acquire-ao3``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..corpus import DOC_ID, TEXT, YEAR, Corpus
from ..logging_utils import get_logger
from .base import CorpusLoader, register_loader

logger = get_logger(__name__)

# Columns kept from the scraper output; everything else (kudos, hits, ...) is dropped on read.
_KEEP_COLUMNS = {
    "work_id", "chapter_id", "chapter_number", "chapter_title", "chapter_date",
    "chapter_word_count", "title", "authors", "fandoms", "rating", "language",
    "additional_tags", "freeform_tags", "word_count",
}


@register_loader
class AO3Loader(CorpusLoader):
    """One document per posted chapter.

    Every CSV in ``path`` (or the single CSV ``path``) is read in chunks. The publication year
    comes from ``chapter_date`` (the date shown in the work's chapter index), and ``doc_id`` is
    ``<file name>:<chapter_id>``. A work tagged with several fandoms appears once per fandom
    file; the AO3 cleaner removes those copies by ``chapter_id``.

    Args:
        language: keep only works in this language (``None`` keeps all).
        max_chars: keep only the first ``max_chars`` characters of each chapter. Only the start
            of a chapter reaches the 400-word window, so this caps memory without changing
            results (10,000 characters is ~4x the window).
        chunksize: rows read per chunk.
        pattern: file glob when ``path`` is a directory.
    """

    name = "ao3"

    def __init__(self, path, years=None, language: str | None = "English",
                 max_chars: int | None = 10_000, chunksize: int = 20_000,
                 date_column: str = "chapter_date", text_column: str = "story_text",
                 pattern: str = "*.csv") -> None:
        super().__init__(path, years)
        self.language = language
        self.max_chars = max_chars
        self.chunksize = chunksize
        self.date_column = date_column
        self.text_column = text_column
        self.pattern = pattern

    def load(self) -> Corpus:
        frames = [self._read_file(path) for path in self.csv_files(self.pattern)]
        frame = pd.concat(frames, ignore_index=True)

        duplicated = frame[DOC_ID].duplicated()
        if duplicated.any():
            logger.info("Dropping %d repeated rows (same file and chapter_id)", int(duplicated.sum()))
            frame = frame.loc[~duplicated]

        corpus = Corpus(frame.set_index(DOC_ID), name="ao3")
        logger.info("Loaded %d AO3 chapters from %d year(s)", len(corpus), len(corpus.years))
        return corpus

    def _read_file(self, path: Path) -> pd.DataFrame:
        wanted = _KEEP_COLUMNS | {self.date_column, self.text_column}
        parts = []
        reader = pd.read_csv(path, usecols=lambda c: c in wanted, chunksize=self.chunksize,
                             low_memory=False)
        for chunk in reader:
            parts.append(self._prepare_chunk(chunk, path.name))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        logger.info("  %s: %d chapters", path.name, len(frame))
        return frame

    def _prepare_chunk(self, chunk: pd.DataFrame, source_file: str) -> pd.DataFrame:
        missing = {"chapter_id", self.date_column, self.text_column} - set(chunk.columns)
        if missing:
            raise ValueError(f"{source_file} is missing required column(s): {sorted(missing)}")

        chunk = chunk.dropna(subset=[self.text_column])
        dates = pd.to_datetime(chunk[self.date_column], errors="coerce")
        chunk = chunk.loc[dates.notna()].copy()
        chunk[YEAR] = dates.loc[dates.notna()].dt.year.astype(int)

        if self.years is not None:
            chunk = chunk.loc[chunk[YEAR].isin(self.years)]
        if self.language and "language" in chunk.columns:
            chunk = chunk.loc[chunk["language"].fillna("").str.strip().str.casefold()
                              == self.language.casefold()]

        text = chunk.pop(self.text_column).astype(str)
        if self.max_chars:
            text = text.map(lambda s: _prefix(s, self.max_chars))
        chunk[TEXT] = text

        chapter_ids = pd.to_numeric(chunk["chapter_id"], errors="coerce").astype("Int64").astype(str)
        chunk["chapter_id"] = chapter_ids
        chunk["source_file"] = source_file
        chunk[DOC_ID] = source_file + ":" + chapter_ids
        return chunk


def _prefix(text: str, max_chars: int) -> str:
    """The first ``max_chars`` characters, cut back to the last whole word."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(maxsplit=1)[0]
