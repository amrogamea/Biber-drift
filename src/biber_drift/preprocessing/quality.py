"""Malformed-text diagnostics.

Per-document signals that extracted text is not clean English prose. Each metric gets a
threshold (around the 99th percentile on the ACL corpus) and documents above it are flagged.
Flags are reported per year because a cohort with many broken documents (e.g. OCR'd scans in
the year-2000 ACL papers) looks linguistically "different" for reasons unrelated to style.

| metric            | catches                                                              |
|-------------------|----------------------------------------------------------------------|
| punct_per_word    | punctuation per alphabetic word -- formulas, tables, symbol soup     |
| nonascii_per_1k   | non-ASCII characters per 1,000 characters                            |
| nonlatin_share    | share of letters outside the Latin script                            |
| invisible_chars   | bidi controls, zero-width / private-use glyphs, U+FFFD               |
| blob_chars        | characters in runs of 30+ single-character tokens (image data)       |
| garble_per_1k     | words with punctuation inside them (OCR errors) per 1,000 tokens     |
| oov_share         | share of distinct lowercase words seen in < ``min_vocab_docs`` docs  |
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_QUALITY_THRESHOLDS
from ..corpus import TEXT, Corpus
from ..logging_utils import get_logger

logger = get_logger(__name__)

_ALPHA_WORD = re.compile(r"\b[^\W\d_]+\b")
_LETTER = re.compile(r"[^\W\d_]")
_PUNCT = re.compile(r"[^\w\s]")
_NON_ASCII = re.compile(r"[^\x00-\x7F]")
NONLATIN_LETTER = re.compile(r"(?:(?![\u0000-ɏ])[^\W\d_])")
INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁯﻿-�]")
IMAGE_BLOB = re.compile(r"(?:(?<!\S)\S\s){30,}")
GARBLED_WORD = re.compile(r"[a-z][;:(\[\]~{}|\\^][a-z]")
_LOWER_WORD = re.compile(r"(?<![\w-])[a-z]{3,}(?![\w-])")


def _count(texts: pd.Series, pattern: re.Pattern) -> pd.Series:
    return texts.map(lambda s: len(pattern.findall(s)))


@dataclass
class QualityReport:
    """Per-document metrics and flags."""

    metrics: pd.DataFrame
    thresholds: dict[str, float]
    flags: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.flags = pd.DataFrame({
            f"flag_{metric}": self.metrics[metric] > threshold
            for metric, threshold in self.thresholds.items()
        })
        self.flags["flag_any"] = self.flags.any(axis=1)

    def summary(self) -> pd.DataFrame:
        counts = self.flags.sum().rename("documents").to_frame()
        counts["pct"] = (counts["documents"] / max(len(self.flags), 1) * 100).round(2)
        return counts

    def rate_by_year(self, years: pd.Series) -> pd.DataFrame:
        """Percentage of documents carrying each flag, per year."""
        rates = self.flags.groupby(years.loc[self.flags.index]).mean().mul(100).round(2)
        rates["documents"] = years.loc[self.flags.index].value_counts().sort_index()
        return rates

    def flagged(self, metrics: list[str]) -> pd.Series:
        """True for documents carrying any of the given flags."""
        if not metrics:
            return pd.Series(False, index=self.flags.index)
        return self.flags[[f"flag_{m}" for m in metrics]].any(axis=1)

    def save(self, directory: str | Path, years: pd.Series) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.metrics.join(self.flags).to_parquet(directory / "quality_metrics.parquet")
        self.summary().to_csv(directory / "quality_summary.csv")
        self.rate_by_year(years).to_csv(directory / "quality_flags_by_year.csv")


class QualityDiagnostics:
    """Computes the malformed-text metrics for a corpus."""

    def __init__(self, thresholds: dict[str, float] | None = None, min_vocab_docs: int = 20) -> None:
        self.thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
        self.min_vocab_docs = min_vocab_docs

    def compute(self, corpus: Corpus) -> QualityReport:
        text = corpus.documents[TEXT].astype(object)
        n_tokens = text.str.split().str.len().clip(lower=1)
        n_alpha = _count(text, _ALPHA_WORD).clip(lower=1)
        n_letters = _count(text, _LETTER).clip(lower=1)
        n_chars = text.str.len().clip(lower=1)

        # Corpus vocabulary: lowercase words used in at least ``min_vocab_docs`` documents.
        # Technical terms and inflections recur across documents; OCR errors and foreign-language
        # passages mostly do not.
        word_types = text.map(lambda s: set(_LOWER_WORD.findall(s)))
        frequency: collections.Counter = collections.Counter()
        for types in word_types:
            frequency.update(types)
        vocab = {w for w, c in frequency.items() if c >= self.min_vocab_docs}

        metrics = pd.DataFrame({
            "punct_per_word": _count(text, _PUNCT) / n_alpha,
            "nonascii_per_1k": _count(text, _NON_ASCII) / n_chars * 1000,
            "nonlatin_share": _count(text, NONLATIN_LETTER) / n_letters,
            "invisible_chars": _count(text, INVISIBLE),
            "blob_chars": text.map(lambda s: sum(len(m) for m in IMAGE_BLOB.findall(s))),
            "garble_per_1k": _count(text, GARBLED_WORD) / n_tokens * 1000,
            "oov_share": word_types.map(lambda t: sum(w not in vocab for w in t) / max(len(t), 1)),
        }, index=corpus.documents.index)

        report = QualityReport(metrics.astype(float), self.thresholds)
        logger.info("Quality flags: %d of %d documents carry at least one flag",
                    int(report.flags["flag_any"].sum()), len(metrics))
        return report

    @staticmethod
    def drop_flagged(corpus: Corpus, report: QualityReport, metrics: list[str]) -> Corpus:
        flagged = report.flagged(metrics)
        if flagged.any():
            logger.info("Dropping %d documents flagged for %s", int(flagged.sum()), metrics)
        return corpus.replace(corpus.documents.loc[~flagged.loc[corpus.documents.index]])
