"""The :class:`Corpus` data structure passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

TEXT = "text"
YEAR = "year"
DOC_ID = "doc_id"
ANALYSIS_TEXT = "analysis_text"
ANALYSIS_WORD_COUNT = "analysis_word_count"


class CorpusSchemaError(ValueError):
    """Raised when a document table does not follow the corpus schema."""


@dataclass
class Corpus:
    """A table of documents with a fixed schema.

    ``documents`` is indexed by a unique ``doc_id`` and has at least an integer ``year`` and a
    string ``text`` column. Loaders may add metadata columns (venue, tags, ...). Preprocessing
    adds ``analysis_text`` -- the exact text sent to the feature extractor -- and its word count.
    """

    documents: pd.DataFrame
    name: str = "corpus"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        frame = self.documents
        if frame.index.name != DOC_ID:
            raise CorpusSchemaError(f"Corpus index must be named '{DOC_ID}', got {frame.index.name!r}")
        if not frame.index.is_unique:
            dupes = frame.index[frame.index.duplicated()].unique()[:5].tolist()
            raise CorpusSchemaError(f"doc_id values must be unique, e.g. duplicated: {dupes}")
        missing = {YEAR, TEXT} - set(frame.columns)
        if missing:
            raise CorpusSchemaError(f"Corpus is missing required column(s): {sorted(missing)}")
        if not pd.api.types.is_integer_dtype(frame[YEAR]):
            raise CorpusSchemaError("Column 'year' must hold integers")

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def years(self) -> list[int]:
        return sorted(int(y) for y in self.documents[YEAR].unique())

    @property
    def texts(self) -> pd.Series:
        """The text used for feature extraction: ``analysis_text`` if present, else ``text``."""
        column = ANALYSIS_TEXT if ANALYSIS_TEXT in self.documents else TEXT
        return self.documents[column]

    def counts_by_year(self) -> pd.Series:
        return self.documents[YEAR].value_counts().sort_index().rename("documents")

    def filter_years(self, years: Iterable[int] | None) -> "Corpus":
        if years is None:
            return self
        keep = self.documents[YEAR].isin(list(years))
        return self.replace(self.documents.loc[keep])

    def replace(self, documents: pd.DataFrame) -> "Corpus":
        """A new corpus with the same name and different documents."""
        return Corpus(documents, name=self.name)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _arrow_safe(self.documents).to_parquet(path)

    @classmethod
    def load(cls, path: str | Path, name: str = "corpus") -> "Corpus":
        frame = pd.read_parquet(path)
        frame.index.name = DOC_ID
        return cls(frame, name=name)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, name: str = "corpus") -> "Corpus":
        """Build a corpus from a frame with ``doc_id``/``year``/``text`` columns or index."""
        frame = frame.copy()
        if frame.index.name != DOC_ID:
            if DOC_ID not in frame.columns:
                raise CorpusSchemaError(f"Frame needs a '{DOC_ID}' column or index")
            frame = frame.set_index(DOC_ID)
        frame.index = frame.index.astype(str)
        frame[YEAR] = frame[YEAR].astype(int)
        frame[TEXT] = frame[TEXT].fillna("").astype(str)
        return cls(frame, name=name)


def _arrow_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type object columns to strings so parquet can store them."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string")
    return frame
