"""Draw the texts every model paraphrases: fixed, stratified samples from the prepared runs.

Each sample group takes ``per_year`` documents a year from one pipeline run -- their 400-440-word
analysis windows, which are already tagged -- holding the mix of one column (ACL venue tier, AO3
rating) at the same shares every year, and spreading AO3 draws over as many works as possible.
A pilot subset marks a few hundred of the drawn texts for trying the models first. The draw is
seeded and the sample keeps every document's id, so it can be rebuilt exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import ConfigError, ProjectConfig, load_config, save_config
from ..corpus import ANALYSIS_TEXT, ANALYSIS_WORD_COUNT, DOC_ID, YEAR
from ..features import FeatureMatrix
from ..logging_utils import get_logger
from .config import ParaphraseConfig, ParaphrasePaths, SampleGroupConfig

logger = get_logger(__name__)

CANON = "canon"

# AO3 fandom tags (separated by ";") of the two canons sampled. A chapter belongs to a canon when each
# of its tags names that canon or is a general Tolkien tag. Crossovers with other fandoms, chapters
# tagged with both canons and the later spin-offs are left out -- the Rings of Power series (from
# Sept 2022) and the War of the Rohirrim film (2024) would otherwise change the mix at the break.
_CANON_TAGS = {"lotr": re.compile(r"lord of the rings|\blotr\b", re.I),
               "hobbit": re.compile(r"hobbit", re.I)}
_SPIN_OFF_TAGS = re.compile(r"rings of power|rohirrim|\brpf\b|online|video game|lego|musical|"
                            r"60 seconds|war in the north|third age", re.I)
_GENERAL_TOLKIEN_TAGS = re.compile(r"^tolkien j\. r\. r\.|legendarium|^tolkien - fandom$", re.I)


def ao3_canon(fandoms: str | None) -> str | None:
    """``lotr``, ``hobbit`` or ``None`` for one chapter's fandom tags (see ``_CANON_TAGS``)."""
    canons = set()
    for tag in str(fandoms or "").split(";"):
        tag = tag.strip()
        if not tag or _GENERAL_TOLKIEN_TAGS.search(tag):
            continue
        if _SPIN_OFF_TAGS.search(tag):
            return None
        named = [canon for canon, pattern in _CANON_TAGS.items() if pattern.search(tag)]
        if len(named) != 1:  # another fandom, or one tag naming both canons
            return None
        canons.add(named[0])
    return canons.pop() if len(canons) == 1 else None


def allocate(n: int, shares: pd.Series, available: pd.Series) -> pd.Series:
    """Split ``n`` draws over strata in proportion to ``shares`` (largest remainder), never asking
    a stratum for more than it has available; a shortfall moves to the others, in proportion."""
    available = available.reindex(shares.index, fill_value=0).astype(int)
    quota = pd.Series(0, index=shares.index)
    remaining = min(n, int(available.sum()))
    open_strata = shares.index[available > 0]
    while remaining > 0 and len(open_strata):
        exact = shares.loc[open_strata] / shares.loc[open_strata].sum() * remaining
        add = np.floor(exact).astype(int)
        largest = (exact - add).sort_values(ascending=False).index[:remaining - int(add.sum())]
        add.loc[largest] += 1
        add = np.minimum(add, available.loc[open_strata] - quota.loc[open_strata])
        quota.loc[open_strata] += add
        remaining -= int(add.sum())
        open_strata = open_strata[(available.loc[open_strata] - quota.loc[open_strata]).to_numpy() > 0]
    return quota


def spread_pick(candidates: pd.DataFrame, k: int, rng: np.random.Generator,
                spread_over: str | None = None) -> pd.Index:
    """``k`` random documents. With ``spread_over``, one per group (e.g. work) until every group has
    one, then a second, and so on, so few documents share a group."""
    shuffled = candidates.iloc[rng.permutation(len(candidates))]
    if spread_over:
        rank = shuffled.groupby(spread_over, sort=False).cumcount().to_numpy()
        shuffled = shuffled.iloc[np.argsort(rank, kind="stable")]
    return shuffled.index[:k]


def even_split(n: int, parts: int) -> list[int]:
    """``n`` split into ``parts`` near-equal integers, larger ones first (50 over 3: 17, 17, 16)."""
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


class ParaphraseSampler:
    """Draws the paraphrase sample from prepared pipeline runs (see the module docstring)."""

    def __init__(self, config: ParaphraseConfig) -> None:
        self.config = config

    def run_dir(self, run: str) -> Path:
        return Path(self.config.output_dir) / run

    def draw(self) -> pd.DataFrame:
        cfg = self.config.sample
        if not cfg.groups:
            raise ConfigError("sample.groups is empty: nothing to draw")
        rng = np.random.default_rng(cfg.seed)
        documents: dict[str, pd.DataFrame] = {}
        parts = []
        for name, group in cfg.groups.items():
            if group.run not in documents:
                documents[group.run] = self._eligible(group.run)
            parts.append(self._draw_group(name, group, documents[group.run], rng))
        sample = pd.concat(parts, ignore_index=True)
        repeated = sample.duplicated(["corpus", DOC_ID])
        if repeated.any():
            raise ConfigError(f"{int(repeated.sum())} documents were drawn by two sample groups; "
                              "make the groups' 'where' filters exclusive")
        sample["pilot"] = self._pilot(sample, rng)
        return self._with_text(sample)

    def save(self, sample: pd.DataFrame, paths: ParaphrasePaths) -> None:
        paths.root.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(paths.sample, index=False)
        keys = ["corpus", "group", YEAR]
        counts = sample.groupby(keys).agg(documents=(DOC_ID, "size"), pilot=("pilot", "sum"),
                                          words=("words", "mean"))
        strata = sample.groupby(keys + ["stratum"]).size().unstack("stratum", fill_value=0)
        counts.join(strata).round({"words": 1}).to_csv(paths.sample_counts)
        totals = sample.groupby("group", sort=False).size().to_dict()
        logger.info("Sample: %d texts %s, %d in the pilot -> %s", len(sample), totals,
                    int(sample["pilot"].sum()), paths.sample)

    # ------------------------------------------------------------------------------------

    def _eligible(self, run: str) -> pd.DataFrame:
        """Metadata of the run's documents that have a complete feature row."""
        root = self.run_dir(run)
        corpus_path, features_path = root / "corpus" / "corpus.parquet", root / "features" / "biber_features.parquet"
        for path in (corpus_path, features_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found -- run the '{run}' pipeline (prepare, features) first")
        run_config = load_config(root / "config.yaml") if (root / "config.yaml").exists() else ProjectConfig()

        available = set(pq.read_schema(corpus_path).names)
        wanted = {YEAR}
        for name, group in self.config.sample.groups.items():
            if group.run != run:
                continue
            columns = set(group.where) | ({group.stratify, group.spread_over} - {None})
            if CANON in columns:
                columns = (columns - {CANON}) | {"fandoms"}
            missing = columns - available
            if missing:
                raise ConfigError(f"sample.groups.{name}: column(s) {sorted(missing)} not in {corpus_path}")
            wanted |= columns
        documents = pd.read_parquet(corpus_path, columns=sorted(wanted))
        documents.index = documents.index.astype(str)

        features = FeatureMatrix.from_frame(pd.read_parquet(features_path), run_config.features.exclude)
        untagged = documents.index.difference(features.values.index)
        if len(untagged):
            logger.info("%s: %d documents without complete features are not sampled", run, len(untagged))
        documents = documents.loc[documents.index.intersection(features.values.index, sort=False)]
        if "fandoms" in documents:
            documents[CANON] = documents["fandoms"].map(ao3_canon)
        return documents

    def _draw_group(self, name: str, group: SampleGroupConfig, documents: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
        pool = documents[documents[YEAR].isin(group.years)]
        for column, values in group.where.items():
            pool = pool[pool[column].isin(values)]
        if group.max_per_spread:  # a random max_per_spread documents of each work and year stay eligible
            shuffled = pool.iloc[rng.permutation(len(pool))]
            rank = shuffled.groupby([YEAR, group.spread_over], sort=False).cumcount()
            pool = shuffled[rank < group.max_per_spread].sort_index()
        if group.stratify:
            pool = pool[pool[group.stratify].notna()]
            shares = pool[group.stratify].value_counts(normalize=True)
            logger.info("%s: %s shares held every year: %s", name, group.stratify, shares.round(3).to_dict())

        chosen = []
        for year in group.years:
            candidates = pool[pool[YEAR] == year]
            if len(candidates) <= group.per_year:
                if len(candidates) < group.per_year:
                    logger.warning("%s %d: only %d eligible documents (per_year %d); all are taken",
                                   name, year, len(candidates), group.per_year)
                chosen.append(candidates.index)
            elif not group.stratify:
                chosen.append(spread_pick(candidates, group.per_year, rng, group.spread_over))
            else:
                quota = allocate(group.per_year, shares, candidates[group.stratify].value_counts())
                target = allocate(group.per_year, shares, pd.Series(group.per_year, index=shares.index))
                if not quota.equals(target):
                    logger.warning("%s %d: too few documents for the %s mix %s; drew %s",
                                   name, year, group.stratify, target.to_dict(), quota.to_dict())
                for stratum, k in quota.items():
                    if k:
                        members = candidates[candidates[group.stratify] == stratum]
                        chosen.append(spread_pick(members, int(k), rng, group.spread_over))
        ids = chosen[0].append(chosen[1:]) if chosen else pd.Index([])
        drawn = pool.loc[ids]
        return pd.DataFrame({
            "corpus": group.run, DOC_ID: drawn.index, "group": name, YEAR: drawn[YEAR].to_numpy(),
            "stratum": drawn[group.stratify].to_numpy() if group.stratify else pd.NA,
        })

    def _pilot(self, sample: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
        """Mark ``pilot.per_year`` texts per pilot year of each run, split evenly over its groups."""
        cfg = self.config.sample.pilot
        pilot = pd.Series(False, index=sample.index)
        for run, years in cfg.years.items():
            groups = [name for name, group in self.config.sample.groups.items() if group.run == run]
            for year in years:
                rows = sample[(sample["corpus"] == run) & (sample[YEAR] == year)]
                present = [name for name in groups if (rows["group"] == name).any()]
                if not present:
                    logger.warning("Pilot: %s has no sampled texts in %d", run, year)
                    continue
                for name, k in zip(present, even_split(cfg.per_year, len(present))):
                    members = rows.index[rows["group"] == name]
                    if len(members) < k:
                        logger.warning("Pilot: %s %d has %d texts (wanted %d)", name, year, len(members), k)
                    pilot[members[rng.permutation(len(members))[:k]]] = True
        return pilot

    def _with_text(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Add each document's analysis window (``text``) and its word count (``words``)."""
        parts = []
        for run, rows in sample.groupby("corpus", sort=False):
            path = self.run_dir(run) / "corpus" / "corpus.parquet"
            texts = pd.read_parquet(path, columns=[ANALYSIS_TEXT, ANALYSIS_WORD_COUNT],
                                    filters=[(DOC_ID, "in", rows[DOC_ID].tolist())])
            texts.index = texts.index.astype(str)
            parts.append(rows.join(texts, on=DOC_ID))
        sample = pd.concat(parts).rename(columns={ANALYSIS_TEXT: "text", ANALYSIS_WORD_COUNT: "words"})
        if sample["text"].isna().any():
            raise ValueError(f"{int(sample['text'].isna().sum())} sampled documents have no analysis window")
        sample.insert(0, "item_id", sample["corpus"] + "/" + sample[DOC_ID])
        return sample.sort_values(["corpus", "group", YEAR, DOC_ID]).reset_index(drop=True)


def draw_sample(config: ParaphraseConfig) -> pd.DataFrame:
    """Draw the sample and write ``sample.parquet``, ``sample_counts.csv`` and ``config.yaml``."""
    paths = ParaphrasePaths(config.run_dir)
    sampler = ParaphraseSampler(config)
    sample = sampler.draw()
    save_config(config, paths.config)
    sampler.save(sample, paths)
    return sample
