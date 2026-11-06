"""Compare each original with its paraphrases (``biber-drift paraphrase analyze``).

Raidar across years: if post-2023 texts already read like LLM output, a paraphrase should change
them less than older texts. For every model and kept output this measures how far the cleaned
paraphrase window lies from its original (``MEASURES``), then asks whether that change drops after
the break, beyond the earlier trend. Under ``analysis/<name>/`` it writes:

- ``texts/<model>.parquet``: per kept text, the measures with ids, stratum, lengths and check flags;
- ``differences/<model>.parquet``: per kept text, paraphrase minus original in z-scores, per
  feature and per Biber dimension (for ``features_yearly.csv`` and, later, the "LLM direction");
- ``yearly.csv``: yearly mean change per model, group and measure, with bootstrap intervals;
- ``features_yearly.csv``: the mean (absolute) z-change of every feature and dimension per model,
  group and year; descriptive only, no per-feature tests;
- ``trend/``: interrupted time series of the period means (ACL yearly, AO3 quarterly) with placebo
  breaks;
- ``regressions.csv``: the ``post`` effect in change ~ post + time + stratum (+ group) (+ log length
  ratio), per model (OLS, HC3 errors) and pooled over the ``main`` models (mixed model, random
  intercept per text);
- ``mmd/``: cohort MMD^2 between each group-year's originals and their paraphrases, on one fixed
  sample per group-year (the same texts for every model) and the run's sigma, with a paired null
  (original and paraphrase swap places within random pairs), a bootstrap over pairs, and yearly
  interrupted time series of MMD^2 and its effect (``analysis.mmd``; all kept texts only, since the
  sensitivity variant leaves no common fixed sample);
- ``scaler_<corpus>.csv`` and ``analysis.json``.

Every statistic is computed for all kept texts (variant ``all``) and again without the outputs the
check flagged as short, long or copied (``analysis.exclude_flags``). Means are weighted to each
group's overall stratum shares, so years with another venue or rating mix (ACL 2013: 171 / 59 / 27,
LotR 2009: 214 / 86) do not move the series. Less change after the break shows as a negative
``post`` effect for the distances and a positive one for ``overlap``.

The ruler is the corpus run's ``reference_scaler.csv``, except for ``f_43``, which here is the
moving-average type-token ratio and is scaled on the re-tagged originals of its corpus. Files
hold ids and numbers only: the texts for the edit distance are cut again in memory by
:meth:`ParaphraseTagger.prepare`, never written.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..analysis.cohort_mmd import representative_sample
from ..analysis.mmd import confidence_interval, paired_mmd2
from ..analysis.multiple_testing import P_ADJUST_METHODS, adjust_pvalues
from ..analysis.trend import InterruptedTimeSeries, break_time, document_times
from ..config import ConfigError
from ..features.dimensions import BiberDimensionScorer
from ..logging_utils import get_logger
from .config import ParaphraseConfig, ParaphrasePaths
from .generate import stage_items
from .tag import ORIGINAL, ParaphraseTagger

logger = get_logger(__name__)

MATTR = "f_43_type_token"
ALL = "all"
MEASURES = {
    "biber": "Euclidean distance of the z-scored Biber features (NaN without f_43)",
    "biber_no_f43": "the same without f_43, so outputs under 100 tokens count too",
    "dimensions": "Euclidean distance of the Dimension 1-6 scores (NaN without f_43)",
    "edit": "word-level edit distance / the original's words",
    "overlap": "share of the original's 8-word sequences kept (the check's; higher = less change)",
}
TEXT_COLUMNS = ["item_id", "corpus", "group", "year", "stratum", "words_original", "words",
                "length_ratio", "window_ratio", "overlap", "short", "long", "copy"]
_WORD = re.compile(r"\b\w+\b")
_EPS = 1e-9


def edit_distance(source: list[str], target: list[str]) -> int:
    """Levenshtein distance between two token sequences (insertion, deletion, substitution: 1 each)."""
    if not source or not target:
        return len(source) + len(target)
    codes: dict[str, int] = {}
    a = [codes.setdefault(token, len(codes)) for token in source]
    b = np.array([codes.setdefault(token, len(codes)) for token in target])
    offsets = np.arange(len(b) + 1)
    previous = offsets.copy()
    current = np.empty_like(previous)
    for i, token in enumerate(a, start=1):
        current[0] = i
        np.minimum(previous[1:] + 1, previous[:-1] + (b != token), out=current[1:])
        # insertions: current[j] = min over k <= j of current[k] + (j - k), one cumulative minimum
        previous = np.minimum.accumulate(current - offsets) + offsets
    return int(previous[-1])


def word_edit_rate(original: str, paraphrase: str) -> float:
    """Word-level edit distance (lower-cased words, punctuation ignored) per word of the original."""
    source, target = _WORD.findall(original.lower()), _WORD.findall(paraphrase.lower())
    return edit_distance(source, target) / len(source) if source else float("nan")


def corpus_scaler(run_scaler: pd.DataFrame, originals: pd.DataFrame) -> pd.DataFrame:
    """The run's ruler (``mean``, ``std`` per feature) with ``f_43`` refitted on ``originals``.

    The pipeline runs hold the plain type-token ratio; here ``f_43`` is the MATTR, so its mean and
    (population) sd come from the re-tagged originals of the corpus. The other features keep the
    run's values, the same ruler as the pipeline's MMD and trend analyses.
    """
    scaler = run_scaler[["mean", "std"]].astype(float).assign(source="run")
    if MATTR in scaler.index:
        values = originals[MATTR].dropna()
        scaler.loc[MATTR, ["mean", "std"]] = [values.mean(), values.std(ddof=0)]
        scaler.loc[MATTR, "source"] = f"originals ({len(values)})"
    scaler.loc[scaler["std"] == 0, "std"] = 1.0
    scaler.index.name = "feature"
    return scaler


def standardise(features: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """z-scores of the scaler's features (columns in the scaler's order)."""
    return (features[list(scaler.index)].astype(float) - scaler["mean"]) / scaler["std"]


def change_measures(original: pd.DataFrame, paraphrase: pd.DataFrame, scaler: pd.DataFrame,
                    scorer: BiberDimensionScorer) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per paraphrase (rows of ``paraphrase``, matched to ``original`` by index): the Biber, no-f_43
    and Dimension distances, and the signed z-differences per feature and dimension."""
    z_original = standardise(original.loc[paraphrase.index], scaler)
    z_paraphrase = standardise(paraphrase, scaler)
    difference = z_paraphrase - z_original
    dimensions = scorer.project(z_paraphrase) - scorer.project(z_original)
    measures = pd.DataFrame({
        "biber": _norm(difference),
        "biber_no_f43": _norm(difference.drop(columns=[MATTR], errors="ignore")),
        "dimensions": _norm(dimensions),
    }, index=paraphrase.index)
    return measures, pd.concat([difference, dimensions], axis=1)


def _norm(frame: pd.DataFrame) -> pd.Series:
    """Euclidean length of each row; NaN when any value is missing."""
    return np.sqrt((frame ** 2).sum(axis=1, skipna=False))


def stratified_means(frame: pd.DataFrame, by: list[str], columns: list[str]) -> pd.DataFrame:
    """Per ``by`` cell, each column's mean over strata weighted by the rows' ``weight`` (the stratum's
    share), renormalised over the strata the cell has values for; and ``<column>_n``, the values."""
    grouped = frame.groupby([*by, "stratum"], observed=True)
    means, counts = grouped[columns].mean(), grouped[columns].count()
    weights = (counts > 0).mul(grouped["weight"].first(), axis=0)
    level = by if len(by) > 1 else by[0]
    result = (means.fillna(0) * weights).groupby(level=level).sum() / \
        weights.groupby(level=level).sum().replace(0, np.nan)
    return pd.concat([result, counts.groupby(level=level).sum().add_suffix("_n")], axis=1)


def stratified_bootstrap(values: np.ndarray, strata: np.ndarray, weights: np.ndarray, resamples: int,
                         rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """The stratum-weighted mean of ``values`` (``weights`` = each row's stratum share), and the same
    mean for ``resamples`` draws in which every stratum is resampled with replacement on its own."""
    mean, draws, total = 0.0, np.zeros(resamples), 0.0
    for stratum in np.unique(strata):
        mask = strata == stratum
        cell, weight = values[mask], weights[mask][0]
        mean += weight * cell.mean()
        draws += weight * cell[rng.integers(0, len(cell), size=(resamples, len(cell)))].mean(axis=1)
        total += weight
    return mean / total, draws / total


class ParaphraseAnalysis:
    """Measures every model's change per text, then its course over the years.

    Args:
        stage, name: the generation run, as for ``tag`` (features in ``features/<name>/``).
        keys: model keys or config indexes (default: every model with a features file, in config order).
    """

    def __init__(self, config: ParaphraseConfig, stage: str = "main", name: str | None = None,
                 keys: Iterable[str] | None = None) -> None:
        if config.analysis.p_adjust not in P_ADJUST_METHODS:
            raise ConfigError(f"analysis.p_adjust must be one of {list(P_ADJUST_METHODS)}")
        self.config = config
        self.settings = config.analysis
        self.stage = stage
        self.tagger = ParaphraseTagger(config, stage, name, keys=[ORIGINAL])
        self.paths = ParaphrasePaths(config.run_dir)
        self.directory = self.paths.analysis(self.tagger.name)
        self.keys = [self._key(key) for key in keys] if keys else self._default_keys()
        if not self.keys:
            raise FileNotFoundError(f"No model features in {self.tagger.directory}; run `paraphrase tag` first")
        self.scorer = BiberDimensionScorer(include_seem_appear=False)

    def _key(self, key: str) -> str:
        key = self.config.model_key(key) if key.isdigit() else key
        if not self.tagger.features_path(key).exists():
            raise FileNotFoundError(f"No features {self.tagger.features_path(key)}; run `paraphrase tag` first")
        return key

    def _default_keys(self) -> list[str]:
        order = list(self.config.models)
        found = [path.name.removesuffix(".items.parquet")
                 for path in self.tagger.directory.glob("*.items.parquet")]
        found = [key for key in found if key != ORIGINAL and self.tagger.features_path(key).exists()]
        return sorted(found, key=lambda key: (order.index(key) if key in order else len(order), key))

    def texts_path(self, key: str) -> Path:
        return self.directory / "texts" / f"{key}.parquet"

    def differences_path(self, key: str) -> Path:
        return self.directory / "differences" / f"{key}.parquet"

    def role(self, key: str) -> str:
        model = self.config.models.get(key)
        return model.role if model else "main"

    # ---------------------------------------------------------------- per text

    def run(self) -> Path:
        began = time.perf_counter()
        originals = pd.read_parquet(self.tagger.features_path(ORIGINAL))
        sample = stage_items(self.paths, self.stage, None,
                             ["item_id", "corpus", "doc_id", "group", "year", "stratum"]).set_index("item_id")
        missing = sample.index.difference(originals.index)
        if len(missing):
            raise RuntimeError(f"{len(missing)} sample texts have no original features, e.g. {list(missing[:3])}")
        scalers = self.scalers(originals, sample)
        texts = None
        for key in self.keys:
            if self.texts_path(key).exists():
                logger.info("%s: %s exists, skipped", key, self.texts_path(key))
                continue
            if texts is None:
                texts = self.tagger.prepare(ORIGINAL)[0]
            self.measure(key, originals, texts, sample, scalers)

        frame = pd.concat([pd.read_parquet(self.texts_path(key)).assign(model=key, role=self.role(key))
                           for key in self.keys], ignore_index=True)
        frame = self._with_times(frame, sample)
        written = self.statistics(frame)
        written += self.cohorts(originals, sample, scalers)
        self._write_info(frame, scalers, time.perf_counter() - began)
        logger.info("Analysis of %d models written to %s (%s)", len(self.keys), self.directory,
                    ", ".join(path.name for path in written))
        return self.directory

    def scalers(self, originals: pd.DataFrame, sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Per corpus: its pipeline run's scaler with f_43 on the corpus's originals; saved alongside."""
        scalers = {}
        self.directory.mkdir(parents=True, exist_ok=True)
        for corpus, rows in sample.groupby("corpus", sort=False):
            path = Path(self.config.output_dir) / corpus / "results" / "reference_scaler.csv"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: the analysis uses the ruler of the pipeline run "
                                        f"'{corpus}' (laptop)")
            scaler = corpus_scaler(pd.read_csv(path, index_col="feature"), originals.loc[rows.index])
            scaler.to_csv(self.directory / f"scaler_{corpus}.csv")
            scalers[corpus] = scaler
            if MATTR in scaler.index:
                logger.info("%s: %d features on the run's scaler; f_43 (MATTR) mean %.4f, sd %.4f from %s",
                            corpus, len(scaler), *scaler.loc[MATTR, ["mean", "std", "source"]])
        return scalers

    def measure(self, key: str, originals: pd.DataFrame, texts: pd.Series, sample: pd.DataFrame,
                scalers: dict[str, pd.DataFrame]) -> Path:
        """The measures of one model's kept outputs -> ``texts/<key>.parquet`` and ``differences/<key>.parquet``."""
        began = time.perf_counter()
        paraphrases, record = self.tagger.prepare(key)
        features = pd.read_parquet(self.tagger.features_path(key))
        if set(features.index) != set(paraphrases.index):
            raise RuntimeError(f"{key}: the features hold {len(features)} texts, the filters keep "
                               f"{len(paraphrases)}; the outputs or filters changed since tagging, so tag again")
        measures, differences = [], []
        for corpus, rows in features.groupby(sample.loc[features.index, "corpus"], sort=False):
            values, difference = change_measures(originals, rows, scalers[corpus], self.scorer)
            measures.append(values)
            differences.append(difference)
        measures = pd.concat(measures)
        measures["edit"] = [word_edit_rate(texts[item], paraphrases[item]) for item in measures.index]

        kept = record[record["kept"]].set_index("item_id")
        table = kept[TEXT_COLUMNS[1:]].join(measures, how="inner").rename_axis("item_id").reset_index()
        for path in (self.texts_path(key), self.differences_path(key)):
            path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(differences).rename_axis("item_id").to_parquet(self.differences_path(key))
        table.to_parquet(self.texts_path(key), index=False)    # last: its existence marks the model done
        logger.info("%s: %d texts; median biber %.2f, biber_no_f43 %.2f, dimensions %.2f, edit %.2f, "
                    "overlap %.2f; %.0f s", key, len(table), *table[list(MEASURES)].median(),
                    time.perf_counter() - began)
        return self.texts_path(key)

    def _with_times(self, frame: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
        """Adds each text's ``period``, ``time`` (start of its period, in years), ``t`` (time since
        its group's break), ``post`` and its stratum's ``weight`` (share in its group)."""
        times = []
        for group, rows in sample.groupby("group", sort=False):
            series = self.config.series(group)
            if series.time_unit != "year":
                rows = rows.join(self._dates(rows["corpus"].iloc[0], series.date_column), on="doc_id")
            table = document_times(rows, series.time_unit, series.date_column)
            at = break_time(series.break_at, series.time_unit)
            times.append(table.assign(t=table["time"] - at, post=(table["time"] >= at - _EPS).astype(float)))
            if len(table) < len(rows):
                logger.warning("%s: %d texts have no %s and are left out", group, len(rows) - len(table),
                               series.date_column)
        times = pd.concat(times)
        shares = sample.groupby(["group", "stratum"]).size()
        shares = (shares / shares.groupby(level="group").transform("sum")).rename("weight")
        frame = frame.join(times, on="item_id", how="inner")
        return frame.join(shares, on=["group", "stratum"])

    def _dates(self, run: str, column: str) -> pd.Series:
        path = Path(self.config.output_dir) / run / "corpus" / "corpus.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: the dates ({column}) come from the pipeline run '{run}'")
        return pd.read_parquet(path, columns=[column])[column]

    # ---------------------------------------------------------------- statistics

    def variants(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """All kept texts, and the sensitivity check without the ``exclude_flags`` outputs."""
        flags = self.settings.exclude_flags
        variants = {ALL: frame}
        if flags:
            variants["without_" + "_".join(flags)] = frame[~frame[flags].any(axis=1)]
        return variants

    def statistics(self, frame: pd.DataFrame) -> list[Path]:
        rng = np.random.default_rng(self.settings.seed)
        yearly, series, its, placebos, regressions = [], [], [], [], []
        for variant, texts in self.variants(frame).items():
            yearly.append(self.yearly(texts, rng).assign(variant=variant))
            table, rows, fakes = self.trend(texts)
            series.append(table.assign(variant=variant))
            its.append(rows.assign(variant=variant))
            placebos.append(fakes.assign(variant=variant))
            regressions.append(self.regressions(texts).assign(variant=variant))

        its = pd.concat(its, ignore_index=True)
        if "level_p" in its:
            its["level_p_adjusted"] = its.groupby("variant")["level_p"].transform(
                lambda p: adjust_pvalues(p, self.settings.p_adjust))
        regressions = pd.concat(regressions, ignore_index=True)
        regressions["p_adjusted"] = regressions.groupby(["variant", "estimator", "length"])["p"].transform(
            lambda p: adjust_pvalues(p, self.settings.p_adjust))
        tables = {
            "yearly.csv": pd.concat(yearly, ignore_index=True),
            "features_yearly.csv": self.features_yearly(frame),
            "trend/series.csv": pd.concat(series, ignore_index=True),
            "trend/its.csv": its,
            "trend/its_placebos.csv": pd.concat(placebos, ignore_index=True),
            "regressions.csv": regressions,
        }
        written = []
        for name, table in tables.items():
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            first = ["variant"] if "variant" in table else []
            table[first + [c for c in table.columns if c not in first]].to_csv(path, index=False)
            written.append(path)
        self._log_summary(its, regressions)
        return written

    def yearly(self, texts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Stratum-weighted yearly means per model, group and measure, with bootstrap intervals."""
        rows = []
        for (model, group, year), cell in texts.groupby(["model", "group", "year"], sort=True):
            for measure in MEASURES:
                values = cell[[measure, "stratum", "weight"]].dropna()
                if values.empty:
                    continue
                mean, draws = stratified_bootstrap(values[measure].to_numpy(float), values["stratum"].to_numpy(),
                                                   values["weight"].to_numpy(float),
                                                   self.settings.bootstrap_resamples, rng)
                lower, upper = confidence_interval(draws, self.settings.confidence)
                rows.append({"model": model, "role": cell["role"].iloc[0], "group": group, "year": year,
                             "measure": measure, "texts": len(values), "mean": mean,
                             "ci_lower": lower, "ci_upper": upper})
        return pd.DataFrame(rows)

    def trend(self, texts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Interrupted time series of the stratum-weighted period means per model, group and measure."""
        measures = list(MEASURES)
        tables, rows, placebos = [], [], []
        for (model, group), cell in texts.groupby(["model", "group"], sort=True):
            series = self.config.series(group)
            table = stratified_means(cell, ["period"], measures)
            table.insert(0, "time", cell.groupby("period")["time"].first())
            table = table.sort_values("time")
            tables.append(table.reset_index().assign(model=model, group=group))
            its = InterruptedTimeSeries(break_time(series.break_at, series.time_unit), series.hac_lags,
                                        min_segment=series.min_segment, confidence=self.settings.confidence)
            for measure in measures:
                used = table[table[f"{measure}_n"] >= series.min_documents]
                row, fakes = its.fit(used["time"].to_numpy(float), used[measure].to_numpy(float))
                if row is None:
                    continue
                labels = {"model": model, "role": cell["role"].iloc[0], "group": group, "measure": measure}
                rows.append({**labels, **row})
                placebos.append(fakes.assign(**labels))
        return (pd.concat(tables, ignore_index=True), pd.DataFrame(rows),
                pd.concat(placebos, ignore_index=True) if placebos else pd.DataFrame())

    def regressions(self, texts: pd.DataFrame) -> pd.DataFrame:
        """The ``post`` effect per corpus and measure: OLS per model, and a mixed model over the
        ``main`` models with a random intercept per text; each with and without the length covariate."""
        rows, confidence = [], self.settings.confidence
        texts = texts.assign(log_length=np.log(texts[self.settings.length_covariate]))
        for corpus, data in texts.groupby("corpus", sort=True):
            controls = "post + t + C(stratum)" + (" + C(group)" if data["group"].nunique() > 1 else "")
            for length in (True, False):
                formula = controls + (" + log_length" if length else "")
                for measure in MEASURES:
                    labels = {"corpus": corpus, "measure": measure, "length": length}
                    used = data.dropna(subset=[measure, "log_length"] if length else [measure])
                    for model, cell in used.groupby("model", sort=False):
                        rows.append({**labels, "estimator": "ols", "model": model, "role": cell["role"].iloc[0],
                                     **_post_effect(cell, f"{measure} ~ {formula}", confidence)})
                    main = used[used["role"] == "main"]
                    if main["model"].nunique() > 1:
                        rows.append({**labels, "estimator": "mixed", "model": "main (pooled)", "role": "main",
                                     **_post_effect(main, f"{measure} ~ {formula} + C(model)", confidence,
                                                    groups="item_id")})
        return pd.DataFrame(rows)

    def features_yearly(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Stratum-weighted mean z-change (signed and absolute) of every feature and dimension per
        model, group and year, all kept texts. Descriptive only."""
        tables = []
        for key in self.keys:
            differences = pd.read_parquet(self.differences_path(key))
            columns = list(differences.columns)
            texts = frame[frame["model"] == key].join(differences, on="item_id")
            signed = stratified_means(texts, ["group", "year"], columns)[columns]
            absolute = stratified_means(texts.assign(**{c: texts[c].abs() for c in columns}),
                                        ["group", "year"], columns)[columns]
            table = pd.concat({"mean": signed.stack(), "mean_abs": absolute.stack()}, axis=1)
            table.index.names = ["group", "year", "feature"]
            tables.append(table.reset_index().assign(model=key)[["model", "group", "year", "feature",
                                                                 "mean", "mean_abs"]])
        return pd.concat(tables, ignore_index=True)

    # ---------------------------------------------------------------- cohort MMD

    def cohorts(self, originals: pd.DataFrame, sample: pd.DataFrame,
                scalers: dict[str, pd.DataFrame]) -> list[Path]:
        """Cohort MMD^2 per model and group-year -> ``mmd/`` (nothing without ``analysis.mmd.sizes``)."""
        settings = self.settings.mmd
        if not settings.sizes:
            logger.info("analysis.mmd.sizes is empty: no cohort MMD")
            return []
        rng = np.random.default_rng(self.settings.seed)
        # every model with features, not only those analysed now, so that --model leaves the cohorts as they are
        paraphrases = {key: pd.read_parquet(self.tagger.features_path(key)) for key in self._default_keys()}
        samples = self.cohort_samples(originals, paraphrases, sample, scalers, rng)
        if samples.empty:
            logger.warning("Cohort MMD: no group-year has enough texts for analysis.mmd.sizes")
            return []
        sigmas = {corpus: self._sigma(corpus) for corpus in samples["corpus"].unique()}
        rows = []
        for key in self.keys:
            for (group, year), cohort in samples.groupby(["group", "year"], sort=True):
                corpus, ids = cohort["corpus"].iloc[0], cohort["item_id"]
                x = standardise(originals.loc[ids], scalers[corpus]).to_numpy()
                y = standardise(paraphrases[key].loc[ids], scalers[corpus]).to_numpy()
                observed, null, boot = paired_mmd2(x, y, sigmas[corpus], settings.permutations,
                                                   self.settings.bootstrap_resamples, rng)
                lower, upper = confidence_interval(boot, self.settings.confidence)
                null_mean, null_sd = null.mean(), null.std(ddof=1)
                rows.append({"model": key, "role": self.role(key), "group": group, "year": year,
                             "texts": len(ids), "mmd2": observed, "ci_lower": lower, "ci_upper": upper,
                             "null_mean": null_mean, "null_sd": null_sd,
                             "effect": (observed - null_mean) / null_sd if null_sd else np.nan,
                             "p": (1 + np.count_nonzero(null >= observed)) / (1 + len(null)),
                             "sigma": sigmas[corpus]})
        table = pd.DataFrame(rows)
        table["p_adjusted"] = adjust_pvalues(table["p"], self.settings.p_adjust)
        its, placebos = self.cohort_trend(table)

        directory = self.directory / "mmd"
        directory.mkdir(parents=True, exist_ok=True)
        files = {"cohorts.csv": table, "samples.csv": samples[["group", "year", "item_id"]],
                 "its.csv": its, "its_placebos.csv": placebos}
        for name, frame in files.items():
            frame.to_csv(directory / name, index=False)
        mmd2 = its[its["measure"] == "mmd2"] if len(its) else its
        if len(mmd2):
            significant = mmd2["level_p_adjusted"] < 0.05
            logger.info("Cohort MMD: %d cohorts of %s texts x %d models; %d of %d MMD^2 series with an adjusted "
                        "level-change p < 0.05 (%d of them drops)", samples.groupby(["group", "year"]).ngroups,
                        settings.sizes, len(self.keys), int(significant.sum()), len(mmd2),
                        int((significant & (mmd2["level_change"] < 0)).sum()))
        return [directory / name for name in files]

    def cohort_samples(self, originals: pd.DataFrame, paraphrases: dict[str, pd.DataFrame], sample: pd.DataFrame,
                       scalers: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
        """One fixed sample per group-year that has a size: ``group``, ``year``, ``corpus``, ``item_id``.

        Eligible are the texts that every model kept with complete features (Llama 2 lacks ``f_43``
        for its outputs under 100 tokens), so every model is measured on the same texts; of them,
        the most representative draw of the originals' Dimension scores is kept, as in the pipeline.
        """
        settings = self.settings.mmd
        rows = []
        for corpus, items in sample.groupby("corpus", sort=True):
            scaler = scalers[corpus]
            eligible = items.index
            for frame in (originals, *paraphrases.values()):
                values = frame.loc[frame.index.intersection(eligible), list(scaler.index)]
                eligible = eligible.intersection(values.index[values.notna().all(axis=1)])
            eligible = eligible.sort_values()
            scores = self.scorer.project(standardise(originals.loc[eligible], scaler))
            for (group, year), cell in items.loc[eligible].groupby(["group", "year"], sort=True):
                size = settings.sizes.get(group)
                if size is None:
                    continue
                if len(cell) < size:
                    logger.warning("Cohort MMD: %s %s has %d texts that every model kept with complete "
                                   "features, fewer than %d; left out", group, year, len(cell), size)
                    continue
                picked = representative_sample(cell.index, size, rng, scores, settings.subsample_draws)
                rows += [(group, year, corpus, item_id) for item_id in picked]
        return pd.DataFrame(rows, columns=["group", "year", "corpus", "item_id"])

    def cohort_trend(self, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Yearly interrupted time series of each model's MMD^2 and effect per group, with placebos."""
        rows, placebos = [], []
        for (model, group), cell in table.groupby(["model", "group"], sort=True):
            series = self.config.series(group)
            # cohorts are years: the break is the first year wholly after it (2023Q1 -> 2023)
            at = float(math.ceil(break_time(series.break_at, series.time_unit) - _EPS))
            its = InterruptedTimeSeries(at, confidence=self.settings.confidence)
            cell = cell.sort_values("year")
            for measure in ("mmd2", "effect"):
                row, fakes = its.fit(cell["year"].to_numpy(float), cell[measure].to_numpy(float))
                if row is None:
                    continue
                labels = {"model": model, "role": cell["role"].iloc[0], "group": group, "measure": measure}
                rows.append({**labels, **row})
                placebos.append(fakes.assign(**labels))
        its = pd.DataFrame(rows)
        if len(its):
            its["level_p_adjusted"] = adjust_pvalues(its["level_p"], self.settings.p_adjust)
        return its, pd.concat(placebos, ignore_index=True) if placebos else pd.DataFrame()

    def _sigma(self, corpus: str) -> float:
        path = Path(self.config.output_dir) / corpus / "results" / "reference.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: the cohort MMD uses the sigma of the pipeline run '{corpus}'")
        return float(json.loads(path.read_text())["sigma"])

    def _log_summary(self, its: pd.DataFrame, regressions: pd.DataFrame) -> None:
        pooled = regressions[(regressions["variant"] == ALL) & (regressions["estimator"] == "mixed")
                             & regressions["length"]]
        for _, row in pooled.iterrows():
            logger.info("Pooled main models, %s, %s: post %+.4f (%.0f%% CI %+.4f to %+.4f), adjusted p %.3g",
                        row["corpus"], row["measure"], row["estimate"], self.settings.confidence,
                        row["ci_lower"], row["ci_upper"], row["p_adjusted"])
        main = its[its["variant"] == ALL]
        logger.info("Time series (all texts): %d of %d with an adjusted level-change p < 0.05",
                    int((main["level_p_adjusted"] < 0.05).sum()), len(main))

    def _write_info(self, frame: pd.DataFrame, scalers: dict[str, pd.DataFrame], seconds: float) -> None:
        info = {
            "stage": self.stage, "generations": self.tagger.name, "models": self.keys,
            "roles": {key: self.role(key) for key in self.keys}, "measures": MEASURES,
            "analysis": dataclasses.asdict(self.settings),
            "series": {group: dataclasses.asdict(self.config.series(group)) for group in sorted(frame["group"].unique())},
            "texts": {variant: texts.groupby("model").size().to_dict()
                      for variant, texts in self.variants(frame).items()},
            "f_43": {corpus: {"mean": scaler.loc[MATTR, "mean"], "std": scaler.loc[MATTR, "std"]}
                     for corpus, scaler in scalers.items() if MATTR in scaler.index},
            "seconds": round(seconds, 1),
        }
        (self.directory / "analysis.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def _post_effect(data: pd.DataFrame, formula: str, confidence: float, groups: str | None = None) -> dict:
    """The ``post`` coefficient of an OLS (HC3 errors) or, with ``groups``, a mixed model with a
    random intercept per group."""
    import statsmodels.formula.api as smf

    alpha = 1 - confidence / 100
    sizes = {"texts": data["item_id"].nunique(), "observations": len(data)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if groups is None:
                fit = smf.ols(formula, data).fit(cov_type="HC3")
                converged = True
            else:
                fit = smf.mixedlm(formula, data, groups=data[groups]).fit()
                converged = bool(fit.converged)
        except np.linalg.LinAlgError as exc:     # e.g. a covariate without variation: report, don't stop
            logger.warning("%s (%d rows) could not be fitted: %s", formula, len(data), exc)
            return {"estimate": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p": np.nan,
                    **sizes, "converged": False}
    ci = fit.conf_int(alpha)
    return {"estimate": fit.params["post"], "se": fit.bse["post"], "ci_lower": ci.loc["post", 0],
            "ci_upper": ci.loc["post", 1], "p": fit.pvalues["post"], **sizes, "converged": converged}
