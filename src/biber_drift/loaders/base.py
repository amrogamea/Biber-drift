"""Loader interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..corpus import Corpus

LOADERS: dict[str, type["CorpusLoader"]] = {}


def register_loader(cls: type["CorpusLoader"]) -> type["CorpusLoader"]:
    """Class decorator: make a loader available under ``cls.name`` in configs."""
    LOADERS[cls.name] = cls
    return cls


def create_loader(name: str, path: str | Path, years: list[int] | None = None,
                  options: dict[str, Any] | None = None) -> "CorpusLoader":
    try:
        loader_cls = LOADERS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(LOADERS)}") from None
    return loader_cls(path, years=years, **(options or {}))


class CorpusLoader(ABC):
    """Reads raw files from disk and returns a :class:`Corpus` with the standard schema."""

    name: ClassVar[str]

    def __init__(self, path: str | Path, years: list[int] | None = None) -> None:
        self.path = Path(path)
        self.years = years
        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

    def csv_files(self, pattern: str = "*.csv") -> list[Path]:
        """``path`` itself if it is a file, otherwise the matching files inside it."""
        if self.path.is_file():
            return [self.path]
        files = sorted(self.path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' under {self.path}")
        return files

    @abstractmethod
    def load(self) -> Corpus:
        """Read the raw data and return a corpus (``doc_id`` index, ``year``, ``text``)."""
