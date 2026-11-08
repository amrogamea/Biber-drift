"""Clean, filter, cut and Biber-tag the outputs of one generation run (``biber-drift paraphrase tag``).

For every model the check's flags decide which outputs are left out (``filters.drop``), the
cleaners strip boilerplate (``filters.clean``), and each output is cut like the originals -- at
the first sentence end after 400 words, never past 440 -- but kept whole when shorter. The
sample's originals (the windows the pipeline tagged) are tagged again in the same environment, so
that tagger versions or GPU vs CPU cannot pose as change. Under ``features/<name>/`` it writes:

- ``<key>.parquet``: Biber features, index ``item_id`` (``key`` = a model, or ``original``);
- ``<key>.items.parquet``: per text, kept or not and why, the cleaners applied, the check's flags
  and word counts (ids and numbers only, never text);
- ``<key>.json``: the tagging settings, package versions and device.

A finished ``<key>.parquet`` is skipped; an interrupted key resumes from its tagging cache.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ..config import ConfigError
from ..features.biber import BiberTagger
from ..logging_utils import get_logger
from ..preprocessing.windowing import TextWindower, count_words
from .check import SAMPLE_COLUMNS, flag_names, flag_outputs, read_outputs
from .clean import CLEANERS, clean_output
from .config import ParaphraseConfig, ParaphrasePaths
from .generate import stage_items

logger = get_logger(__name__)

ORIGINAL = "original"


class ParaphraseTagger:
    """Tags the originals and the cleaned outputs of one generation run, key by key.

    Args:
        stage: ``pilot`` or ``main``: which sample texts the run covers.
        name: the generation run's folder (default: the stage).
        keys: ``original`` and/or model keys or config indexes (default: the originals, then every
            model with an output file, in config order).
        limit: only the first ``limit`` texts per corpus, as given to ``generate --limit``; the
            features then go to ``features/<name>_limit<N>/``.
    """

    def __init__(self, config: ParaphraseConfig, stage: str = "main", name: str | None = None,
                 keys: Iterable[str] | None = None, limit: int | None = None) -> None:
        filters = config.filters
        unknown = (set(filters.drop) - set(flag_names())) | (set(filters.clean) - set(CLEANERS))
        if unknown:
            raise ConfigError(f"filters name unknown flags or cleaners: {sorted(unknown)}")
        self.config = config
        self.stage = stage
        self.name = name or stage
        self.limit = limit
        self.paths = ParaphrasePaths(config.run_dir)
        self.generations = self.paths.generations(self.name)
        # A quick test (--limit) gets its own folder: the full run skips keys whose file exists.
        self.directory = self.paths.features(f"{self.name}_limit{limit}" if limit else self.name)
        self.windower = TextWindower(filters.window_words, filters.window_cap)
        self.keys = [self._key(key) for key in keys] if keys else self._default_keys()

    def _key(self, key: str) -> str:
        if key == ORIGINAL:
            return key
        key = self.config.model_key(key) if key.isdigit() else key
        if not (self.generations / f"{key}.jsonl").exists():
            raise FileNotFoundError(f"No output file {key}.jsonl in {self.generations}")
        return key

    def _default_keys(self) -> list[str]:
        order = list(self.config.models)
        models = sorted((path.stem for path in self.generations.glob("*.jsonl")),
                        key=lambda key: (order.index(key) if key in order else len(order), key))
        return [ORIGINAL, *models]

    def features_path(self, key: str) -> Path:
        return self.directory / f"{key}.parquet"

    def items_path(self, key: str) -> Path:
        return self.directory / f"{key}.items.parquet"

    def prepare(self, key: str) -> tuple[pd.Series, pd.DataFrame]:
        """The texts to tag (index ``item_id``) and the record of every sample text for ``key``."""
        items = stage_items(self.paths, self.stage, self.limit, SAMPLE_COLUMNS)
        if key == ORIGINAL:
            record = items[["item_id", "corpus", "group", "year", "stratum"]].assign(
                kept=True, reason=None, words=items["text"].map(count_words))
            return items.set_index("item_id")["text"], record.reset_index(drop=True)

        joined = items.merge(read_outputs(self.generations / f"{key}.jsonl"), on="item_id", how="left")
        flags = flag_outputs(joined, self.config.check)
        filters = self.config.filters
        reasons, cleaned, applied = [], [], []
        for (_, flag), output, original in zip(flags.iterrows(), joined["output"], joined["text"]):
            reason = next((name for name in filters.drop if flag[name]), None)
            if not reason and flag["missing"]:      # nothing to clean, whatever filters.drop says
                reason = "missing"
            text, used = (None, []) if reason else clean_output(output, original, filters.clean)
            if not reason:
                text = self.windower.window(text)
                words = count_words(text)
                reason = "empty" if not words else "under_min_words" if words < filters.min_words else None
            reasons.append(reason)
            cleaned.append(None if reason else text)
            applied.append(",".join(used))
        record = flags.drop(columns=["finish_reason", "prompt_tokens", "output_tokens"]).assign(
            kept=[reason is None for reason in reasons], reason=reasons, cleaned=applied,
            words=[count_words(text) if text else 0 for text in cleaned])
        record["window_ratio"] = record["words"].where(record["kept"]) / record["words_original"]
        texts = pd.Series(cleaned, index=pd.Index(joined["item_id"], name="item_id"), name="text").dropna()
        return texts, record

    def run(self, make_tagger: Callable[[Path], Any] | None = None) -> list[Path]:
        """Tag every key not yet done. ``make_tagger(cache_dir)`` builds the tagger (default: a
        :class:`BiberTagger` with the ``tagging`` settings); tests pass a fake one."""
        settings = self.config.tagging
        make_tagger = make_tagger or (lambda cache: BiberTagger(
            settings.spacy_model, settings.batch_size, 1, cache, gpu=settings.gpu,
            mattr_window=settings.mattr_window))
        written = []
        for key in self.keys:
            path = self.features_path(key)
            if path.exists():
                logger.info("%s: %s exists, skipped", key, path)
                continue
            texts, record = self.prepare(key)
            self._log(key, record)
            self.directory.mkdir(parents=True, exist_ok=True)
            record.to_parquet(self.items_path(key), index=False)

            began = time.perf_counter()
            tagger = make_tagger(self.directory / "cache" / key)
            features = tagger.tag(texts)
            missing = texts.index.difference(features.index.astype(str))
            if len(missing):
                raise RuntimeError(f"{key}: {len(missing)} texts came back untagged, e.g. {list(missing[:3])}")
            features = features.loc[texts.index]
            features.index.name = "item_id"
            features.to_parquet(path)
            self._write_info(key, len(features), time.perf_counter() - began, features)
            tagger.clear_cache()
            logger.info("%s: %d texts tagged -> %s", key, len(features), path)
            written.append(path)
        return written

    def _write_info(self, key: str, texts: int, seconds: float, features: pd.DataFrame) -> None:
        info = {"key": key, "stage": self.stage, "generations": self.name, "texts": texts,
                "seconds": round(seconds, 1), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "tagging": vars(self.config.tagging), "filters": vars(self.config.filters),
                "f_43_missing": int(features["f_43_type_token"].isna().sum())
                if "f_43_type_token" in features else None,
                **_environment()}
        (self.directory / f"{key}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @staticmethod
    def _log(key: str, record: pd.DataFrame) -> None:
        dropped = record.loc[~record["kept"], "reason"].value_counts().to_dict()
        cleaned = record["cleaned"].str.split(",").explode().replace("", None).dropna().value_counts().to_dict() \
            if "cleaned" in record else {}
        kept = record[record["kept"]]
        logger.info("%s: %d of %d texts kept; dropped %s; cleaned %s; words median %d (5-95%%: %d-%d)",
                    key, len(kept), len(record), dropped or "none", cleaned or "none", kept["words"].median(),
                    kept["words"].quantile(0.05), kept["words"].quantile(0.95))


def _environment() -> dict[str, Any]:
    """Versions of the tagging packages and the device (whatever is importable)."""
    from importlib import metadata

    versions = {}
    for package in ("spacy", "en_core_web_trf", "spacy-curated-transformers", "pybiber", "polars", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    device = None
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {"versions": versions, "device": device or "cpu"}
