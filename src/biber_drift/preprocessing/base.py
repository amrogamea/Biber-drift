"""Building blocks for dataset-specific cleaners.

A cleaner is an ordered list of steps. A :class:`TextStep` rewrites every document's text;
a :class:`DocumentFilter` drops whole documents. Running the cleaner returns the cleaned
corpus and a :class:`CleaningReport` with one row per step (documents changed or dropped,
characters removed), so every cleaning decision is visible in the run outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import pandas as pd

from ..corpus import TEXT, Corpus
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextStep:
    """Rewrite each document's text with ``func``."""

    key: str
    description: str
    func: Callable[[str], str]


@dataclass(frozen=True)
class DocumentFilter:
    """Keep only documents for which ``keep(documents)`` is True."""

    key: str
    description: str
    keep: Callable[[pd.DataFrame], pd.Series]


Step = TextStep | DocumentFilter


@dataclass
class StepResult:
    step: str
    description: str
    documents_changed: int = 0
    documents_dropped: int = 0
    characters_removed: int = 0
    """Negative when a step inserts characters (e.g. missing spaces)."""
    documents_remaining: int = 0
    kind: str = "filter"
    """``text`` for steps that rewrite text, ``filter`` for steps that drop documents."""


@dataclass
class CleaningReport:
    """Per-step summary of what a cleaner did."""

    rows: list[StepResult] = field(default_factory=list)

    def add(self, row: StepResult) -> None:
        self.rows.append(row)
        if row.kind == "filter":
            logger.info("  %-38s dropped %7d documents (%d left)",
                        row.description, row.documents_dropped, row.documents_remaining)
        else:
            logger.info("  %-38s changed %7d documents, removed %10d chars",
                        row.description, row.documents_changed, row.characters_removed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(row) for row in self.rows])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)


class CorpusCleaner(ABC):
    """Applies an ordered list of cleaning steps to a corpus.

    Args:
        disable: step keys to skip, e.g. ``["markdown"]``.
    """

    name: ClassVar[str]

    def __init__(self, disable: list[str] | None = None) -> None:
        self.disable = set(disable or [])
        unknown = self.disable - {step.key for step in self.steps()}
        if unknown:
            raise ValueError(f"Unknown {self.name} cleaning step(s) to disable: {sorted(unknown)}")

    @abstractmethod
    def steps(self) -> list[Step]:
        """The cleaning steps, in the order they are applied."""

    def clean(self, corpus: Corpus) -> tuple[Corpus, CleaningReport]:
        documents = corpus.documents.copy()
        # Plain Python strings: the steps use the ``re`` module, not pandas' regex engine.
        documents[TEXT] = documents[TEXT].astype(object)
        report = CleaningReport()
        logger.info("Cleaning %d documents (%s)", len(documents), self.name)

        for step in self.steps():
            if step.key in self.disable:
                continue
            if isinstance(step, TextStep):
                before = documents[TEXT]
                after = before.map(step.func)
                documents[TEXT] = after
                report.add(StepResult(
                    step=step.key,
                    description=step.description,
                    documents_changed=int((after != before).sum()),
                    characters_removed=int(before.str.len().sum() - after.str.len().sum()),
                    documents_remaining=len(documents),
                    kind="text",
                ))
            else:
                keep = step.keep(documents).astype(bool)
                documents = documents.loc[keep]
                report.add(StepResult(
                    step=step.key,
                    description=step.description,
                    documents_dropped=int((~keep).sum()),
                    documents_remaining=len(documents),
                ))
        return corpus.replace(documents), report


CLEANERS: dict[str, type[CorpusCleaner]] = {}


def register_cleaner(cls: type[CorpusCleaner]) -> type[CorpusCleaner]:
    CLEANERS[cls.name] = cls
    return cls


def create_cleaner(name: str, options: dict | None = None) -> CorpusCleaner:
    try:
        cleaner_cls = CLEANERS[name]
    except KeyError:
        raise ValueError(f"No cleaner for dataset '{name}'. Available: {sorted(CLEANERS)}") from None
    return cleaner_cls(**(options or {}))
