"""Biber (1988) feature extraction with pybiber and spaCy."""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..corpus import DOC_ID
from ..logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_PREFIX = "f_"


class BiberTaggingError(RuntimeError):
    """pybiber returned an impossible value (e.g. f_23 >= 1000 per 1,000 tokens)."""


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # small spaCy models run without torch
        return False
    return bool(torch.cuda.is_available())


def _quiet_pybiber() -> None:
    """pybiber attaches INFO-level handlers to its loggers on import; keep only warnings."""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("pybiber"):
            logging.getLogger(name).setLevel(logging.WARNING)


class BiberTagger:
    """Tags texts with the 67 pybiber features, resumably.

    Texts are processed longest-first in batches; every finished batch is appended to an
    ndjson cache, so an interrupted run resumes where it stopped (already tagged ``doc_id``s
    are skipped). A fresh ``pybiber.CorpusProcessor`` is created per batch -- reusing one
    across batches produced inflated ``f_23_wh_clause`` counts.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", batch_size: int = 100,
                 n_process: int = 1, cache_dir: str | Path = ".cache", gpu: str = "auto") -> None:
        self.spacy_model = spacy_model
        self.batch_size = batch_size
        self.n_process = n_process
        self.cache_dir = Path(cache_dir)
        self.gpu = gpu

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"biber_{self.spacy_model}.ndjson"

    def tag(self, texts: pd.Series) -> pd.DataFrame:
        """Return a frame indexed by ``doc_id`` with one column per Biber feature."""
        import polars as pl
        import pybiber as pb
        import spacy

        _quiet_pybiber()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        pending = pl.DataFrame({
            "doc_id": texts.index.astype(str).tolist(),
            "text": texts.astype(str).tolist(),
        })
        # Longest texts first, so batches hold texts of similar length.
        pending = (pending.with_columns(pl.col("text").str.len_chars().alias("_length"))
                   .sort("_length", descending=True).drop("_length"))

        if self.cache_path.exists():
            done = pl.read_ndjson(self.cache_path)["doc_id"]
            before = pending.height
            pending = pending.filter(~pl.col("doc_id").is_in(done.to_list()))
            logger.info("Resuming: %d of %d documents already tagged", before - pending.height, before)

        if pending.height:
            self._select_device(spacy)
            logger.info("Loading spaCy model %s", self.spacy_model)
            nlp = spacy.load(self.spacy_model, disable=["ner"])
            starts = range(0, pending.height, self.batch_size)
            for start in tqdm(starts, desc="Biber tagging", unit="batch"):
                batch = pending[start:start + self.batch_size]
                with contextlib.redirect_stdout(io.StringIO()):  # pybiber prints timing lines
                    tokens = pb.CorpusProcessor().process_corpus(
                        batch, nlp_model=nlp, n_process=self.n_process,
                        batch_size=self.batch_size, show_progress=False,
                    )
                    tagged = pb.biber(tokens, force_ttr=True)
                self._check(tagged)
                with open(self.cache_path, "a", encoding="utf-8") as handle:
                    handle.write(tagged.write_ndjson())

        tagged = pl.read_ndjson(self.cache_path).to_pandas()
        tagged = tagged.drop_duplicates(DOC_ID, keep="last").set_index(DOC_ID)
        return tagged.loc[tagged.index.intersection(texts.index.astype(str))]

    def clear_cache(self) -> None:
        self.cache_path.unlink(missing_ok=True)

    def _select_device(self, spacy) -> None:
        """Put spaCy on a GPU when configured and available.

        ``auto`` uses a CUDA GPU only: on Apple silicon, en_core_web_trf through Metal (MPS) was
        about 4x SLOWER than the CPU in a test. ``require`` forces any GPU spaCy supports, MPS
        included.
        """
        if self.gpu == "off":
            logger.info("spaCy runs on the CPU (features.gpu: off)")
        elif self.gpu == "require":
            spacy.require_gpu()
            logger.info("spaCy runs on the GPU")
        elif _cuda_available():
            spacy.require_gpu()
            logger.info("spaCy runs on the CUDA GPU")
        else:
            logger.info("spaCy runs on the CPU (no CUDA GPU found)")

    @staticmethod
    def _check(tagged) -> None:
        import polars as pl

        bad = tagged.filter(pl.col("f_23_wh_clause") >= 1000)
        if bad.height:
            ids = bad["doc_id"].to_list()[:5]
            raise BiberTaggingError(
                f"f_23_wh_clause >= 1000 for doc_id(s) {ids}. Inspect those texts (often "
                "malformed extraction) and drop them before re-running; finished batches are cached."
            )


@dataclass
class FeatureMatrix:
    """Numeric Biber features, one row per document (index ``doc_id``)."""

    values: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return list(self.values.columns)

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, exclude: list[str] | tuple[str, ...] = ()) -> "FeatureMatrix":
        """Keep the ``f_*`` columns (sorted, minus ``exclude``); drop rows with missing values."""
        columns = sorted(c for c in frame.columns if c.startswith(FEATURE_PREFIX) and c not in exclude)
        values = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        complete = values.notna().all(axis=1)
        if (~complete).any():
            logger.info("Dropping %d documents with incomplete features", int((~complete).sum()))
        values = values.loc[complete]
        values.index = values.index.astype(str)
        values.index.name = DOC_ID
        return cls(values)

    def subset(self, doc_ids) -> "FeatureMatrix":
        """Rows for ``doc_ids`` (those present), in that order."""
        doc_ids = pd.Index(doc_ids).astype(str)
        return FeatureMatrix(self.values.loc[doc_ids.intersection(self.values.index, sort=False)])

    def standardized(self, reference: pd.DataFrame | None = None) -> pd.DataFrame:
        """z-scores against ``reference`` (default: these rows), population sd, zero sd -> 1."""
        reference = self.values if reference is None else reference
        mean = reference.mean(axis=0)
        std = reference.std(axis=0, ddof=0).replace(0, 1.0)
        return (self.values - mean) / std

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.values.to_parquet(path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureMatrix":
        values = pd.read_parquet(path)
        values.index.name = DOC_ID
        return cls(values)
