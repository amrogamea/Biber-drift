"""The end-to-end pipeline: prepare -> features -> analyze -> report.

Each stage reads its inputs from and writes its outputs to the run directory
(``<output_dir>/<name>/``), so stages can be re-run independently -- e.g. re-run ``analyze``
with a different sample size without re-tagging the corpus.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis import (
    CohortMMDAnalysis, FeatureChangeAnalysis, FeatureReference, SampleSizeStability, VarianceAnalysis,
)
from .config import ConfigError, ProjectConfig, save_config
from .corpus import YEAR, Corpus
from .features import BiberDimensionScorer, BiberTagger, FeatureMatrix
from .loaders import create_loader
from .logging_utils import get_logger
from .preprocessing import (
    CleaningReport, QualityDiagnostics, StepResult, TextWindower, create_cleaner,
)

logger = get_logger(__name__)

STAGES = ("prepare", "features", "analyze", "report")
ERA = "era"
GROUP = "group"
ERA_GROUP = "era_group"


@dataclass(frozen=True)
class RunPaths:
    """On-disk layout of one run."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus" / "corpus.parquet"

    @property
    def reports(self) -> Path:
        return self.root / "corpus"

    @property
    def features(self) -> Path:
        return self.root / "features" / "biber_features.parquet"

    @property
    def features_info(self) -> Path:
        return self.root / "features" / "biber_features.json"

    @property
    def feature_cache(self) -> Path:
        return self.root / "features" / ".cache"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"


class Pipeline:
    """Runs the stages configured in a :class:`ProjectConfig`."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.paths = RunPaths(config.run_dir)

    # ------------------------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------------------------

    def run(self, stages: Iterable[str] = STAGES) -> None:
        stages = list(stages)
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown stage(s) {sorted(unknown)}; choose from {list(STAGES)}")
        save_config(self.config, self.paths.config)
        logger.info("Run directory: %s", self.paths.root.resolve())

        corpus = features = None
        for stage in STAGES:  # canonical order, whatever order they were given in
            if stage not in stages:
                continue
            logger.info("===== stage: %s =====", stage)
            if stage == "prepare":
                corpus = self.prepare()
            elif stage == "features":
                features = self.extract_features(corpus)
            elif stage == "analyze":
                self.analyze(corpus, features)
            elif stage == "report":
                self.report()

    # ------------------------------------------------------------------------------------
    # Stage 1: load, clean, diagnose, window
    # ------------------------------------------------------------------------------------

    def prepare(self) -> Corpus:
        cfg = self.config
        loader = create_loader(cfg.data.dataset, cfg.data.path, cfg.data.years, cfg.data.loader)
        corpus = loader.load()
        self.paths.reports.mkdir(parents=True, exist_ok=True)
        corpus.counts_by_year().to_csv(self.paths.reports / "documents_by_year_raw.csv")

        report = CleaningReport()
        if cfg.cleaning.enabled:
            corpus, report = create_cleaner(cfg.data.dataset, cfg.cleaning.options).clean(corpus)

        if cfg.quality.enabled:
            diagnostics = QualityDiagnostics(cfg.quality.thresholds, cfg.quality.min_vocab_docs)
            quality = diagnostics.compute(corpus)
            quality.save(self.paths.reports, corpus.documents[YEAR])
            if cfg.quality.drop_flags:
                before = len(corpus)
                corpus = diagnostics.drop_flagged(corpus, quality, cfg.quality.drop_flags)
                report.add(StepResult("quality", f"quality flags {cfg.quality.drop_flags}",
                                      documents_dropped=before - len(corpus),
                                      documents_remaining=len(corpus)))

        windower = TextWindower(cfg.window.target_words, cfg.window.hard_cap)
        corpus, window_step = windower.apply(corpus, cfg.window.min_words, cfg.window.enabled)
        report.add(window_step)

        report.save(self.paths.reports / "cleaning_report.csv")
        corpus.counts_by_year().to_csv(self.paths.reports / "documents_by_year.csv")
        corpus.save(self.paths.corpus)
        logger.info("Prepared corpus: %d documents -> %s", len(corpus), self.paths.corpus)
        return corpus

    # ------------------------------------------------------------------------------------
    # Stage 2: Biber features
    # ------------------------------------------------------------------------------------

    def extract_features(self, corpus: Corpus | None = None) -> FeatureMatrix:
        cfg = self.config.features
        corpus = corpus if corpus is not None else self._load_corpus()
        tagger = BiberTagger(cfg.spacy_model, cfg.batch_size, cfg.n_process, self.paths.feature_cache,
                             gpu=cfg.gpu)
        raw = tagger.tag(corpus.texts)
        self.paths.features.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(self.paths.features)
        self.paths.features_info.write_text(json.dumps(
            {"spacy_model": cfg.spacy_model, "documents": len(raw)}, indent=2))
        tagger.clear_cache()
        logger.info("Biber features for %d documents -> %s", len(raw), self.paths.features)
        return FeatureMatrix.from_frame(raw, cfg.exclude)

    # ------------------------------------------------------------------------------------
    # Stage 3: statistics
    # ------------------------------------------------------------------------------------

    def analyze(self, corpus: Corpus | None = None, features: FeatureMatrix | None = None) -> None:
        cfg = self.config
        corpus = corpus if corpus is not None else self._load_corpus()
        if features is None:
            self._check_feature_model()
            features = FeatureMatrix.from_frame(pd.read_parquet(self.paths.features), cfg.features.exclude)
        features = features.subset(corpus.documents.index)
        cohorts = self._cohort_table(corpus.documents.loc[features.values.index])
        results = self.paths.results
        results.mkdir(parents=True, exist_ok=True)
        logger.info("Analysing %d documents x %d features", len(features), len(features.feature_names))

        # ONE scaler and ONE kernel bandwidth for the whole run (as in the paper): every analysis
        # below measures with this reference, so their numbers share one ruler.
        reference = FeatureReference.fit(features.values, np.random.default_rng(cfg.mmd.seed),
                                         cfg.mmd.bandwidth_max_points)
        reference.save(results)
        logger.info("Reference: scaler and sigma %.4f fitted on all %d documents", reference.sigma,
                    reference.documents)

        if cfg.mmd.enabled:
            self._analyze_mmd(features, cohorts, reference)
        if cfg.stability.enabled:
            self._analyze_stability(features, cohorts, reference)
        if cfg.dimensions.enabled:
            self._analyze_dimensions(features, cohorts)
        if cfg.feature_change.enabled:
            self._analyze_feature_change(features, cohorts)
        if cfg.variance.enabled:
            self._analyze_variance(features, cohorts, reference)

    def _check_feature_model(self) -> None:
        """Refuse features tagged with a different spaCy model than the config names."""
        if not self.paths.features.exists():
            raise FileNotFoundError(f"{self.paths.features} not found -- run the 'features' stage first")
        if self.paths.features_info.exists():
            tagged_with = json.loads(self.paths.features_info.read_text()).get("spacy_model")
            wanted = self.config.features.spacy_model
            if tagged_with and tagged_with != wanted:
                raise ConfigError(f"The features were tagged with {tagged_with} but the config asks for "
                                  f"{wanted}: re-run the 'features' stage (or set features.spacy_model).")

    def _dimension_scorer(self) -> BiberDimensionScorer:
        cfg = self.config.dimensions
        return BiberDimensionScorer(cfg.standardization, cfg.include_seem_appear)

    def _sampling_dimensions(self, features: FeatureMatrix) -> pd.DataFrame | None:
        """Dimension scores used to draw representative fixed samples (paper: 1988-normed dims)."""
        if self.config.mmd.subsample != "representative":
            return None
        try:
            return self._dimension_scorer().score(features.values)
        except ValueError as exc:
            logger.warning("Representative sampling unavailable (%s); using random fixed samples", exc)
            return None

    def _cohort_table(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Per document: year, era (if the year is a key year) and group (if configured)."""
        cfg = self.config.analysis
        table = pd.DataFrame({YEAR: documents[YEAR].astype(int)}, index=documents.index)
        # Nullable strings, so a missing era or group propagates to era_group as <NA>.
        table[ERA] = table[YEAR].astype(str).astype("string").where(table[YEAR].isin(cfg.eras))
        if cfg.group_column:
            if cfg.group_column not in documents.columns:
                raise ValueError(f"analysis.group_column '{cfg.group_column}' is not a corpus column")
            table[GROUP] = documents[cfg.group_column].astype("string")
            table[ERA_GROUP] = table[ERA] + " - " + table[GROUP]
        return table

    def _group_order(self, cohorts: pd.DataFrame) -> list[str]:
        order = self.config.analysis.group_order
        present = set(cohorts[GROUP].dropna())
        return [g for g in order if g in present] if order else sorted(present)

    def _analyze_mmd(self, features: FeatureMatrix, cohorts: pd.DataFrame,
                     reference: FeatureReference) -> None:
        cfg = self.config
        years = cohorts[YEAR]
        keep = years.isin(cfg.mmd.years) if cfg.mmd.years else pd.Series(True, index=years.index)
        analysis = CohortMMDAnalysis(cfg.mmd)
        dims = self._sampling_dimensions(features)
        by_year = analysis.run(features.values.loc[keep], years.loc[keep], dimension_scores=dims,
                               reference=reference, pool=features.values)
        by_year.save(self.paths.results, "mmd_by_year")

        if cfg.mmd.era_group_cohorts:
            labels = cohorts[ERA_GROUP].dropna()
            order = [f"{era} - {group}" for era in cfg.analysis.eras for group in self._group_order(cohorts)]
            n = cfg.mmd.era_group_sample_size
            era_group = CohortMMDAnalysis(dataclasses.replace(cfg.mmd, sample_size=n, min_cohort_documents=n))
            result = era_group.run(features.values.loc[labels.index], labels, order=order,
                                   dimension_scores=dims, with_feature_shifts=False,
                                   reference=reference, pool=features.values)
            result.save(self.paths.results, "mmd_by_era_group")

    def _analyze_stability(self, features: FeatureMatrix, cohorts: pd.DataFrame,
                           reference: FeatureReference) -> None:
        cfg = self.config
        pooled = cfg.stability.scaling == "pooled"
        values = reference.transform(features.values) if pooled else features.values
        stability = SampleSizeStability(cfg.stability, cfg.mmd.confidence)

        tables = []
        for grouping in cfg.stability.groupings:
            if grouping == "all":
                groups = {f"{self.config.name} (all documents)": values.to_numpy()}
            else:
                labels = cohorts[grouping].dropna()
                if grouping == ERA:
                    order = [str(e) for e in cfg.analysis.eras]
                elif grouping == GROUP:
                    order = self._group_order(cohorts)
                else:
                    order = [f"{e} - {g}" for e in cfg.analysis.eras for g in self._group_order(cohorts)]
                groups = {label: values.loc[labels.index[labels == label]].to_numpy()
                          for label in order if (labels == label).any()}
            tables.append(stability.run(groups, grouping, sigma=reference.sigma if pooled else None))
        pd.concat(tables, ignore_index=True).to_csv(
            self.paths.results / "sample_size_stability.csv", index=False)

    def _analyze_dimensions(self, features: FeatureMatrix, cohorts: pd.DataFrame) -> None:
        cfg = self.config.dimensions
        keep = cohorts[YEAR].isin(cfg.years) if cfg.years else pd.Series(True, index=cohorts.index)
        if cfg.violin_by == ERA:
            keep &= cohorts[ERA].notna()
        # corpus: z-scores against the documents compared; biber1988: against Biber's 1988 norms.
        scorer = self._dimension_scorer()
        scores = scorer.score(features.values.loc[keep])
        scores = scores.join(cohorts.loc[keep])
        scores.to_parquet(self.paths.results / "biber_dimensions.parquet")
        dimension_columns = scorer.dimensions
        scores.groupby(YEAR)[dimension_columns].mean().to_csv(
            self.paths.results / "biber_dimensions_mean_by_year.csv")
        if GROUP in scores:
            scores.groupby(GROUP)[dimension_columns].mean().to_csv(
                self.paths.results / "biber_dimensions_mean_by_group.csv")

    def _analyze_feature_change(self, features: FeatureMatrix, cohorts: pd.DataFrame) -> None:
        cfg = self.config
        years = cfg.feature_change.years
        if not years or len(years) < 2:
            logger.warning("feature_change needs >= 2 years (set feature_change.years or analysis.eras); skipped")
            return
        analysis = FeatureChangeAnalysis(years, cfg.feature_change.absolute_weight,
                                         cfg.feature_change.relative_weight)
        directory = self.paths.results / "feature_change"
        analysis.run(features.values, cohorts[YEAR]).save(directory)
        if cfg.feature_change.by_group:
            for group in self._group_order(cohorts):
                members = cohorts.index[cohorts[GROUP] == group]
                try:
                    analysis.run(features.values.loc[members], cohorts[YEAR], group=group).save(directory)
                except ValueError as exc:
                    logger.warning("Feature change skipped for group %s: %s", group, exc)

    def _analyze_variance(self, features: FeatureMatrix, cohorts: pd.DataFrame,
                          reference: FeatureReference) -> None:
        cfg = self.config
        # Standardise first, on the run's reference (the full corpus), so every feature counts equally.
        standardized = reference.transform(features.values)

        def analysis(default_n: int) -> VarianceAnalysis:
            return VarianceAnalysis(cfg.variance.sample_size or default_n, cfg.variance.resamples,
                                    cfg.mmd.confidence, cfg.variance.seed)

        years = cohorts[YEAR]
        by_year = {str(y): years.index[years == y] for y in sorted(years.unique())}
        tables = [analysis(cfg.mmd.sample_size).run(standardized, by_year, "year")]
        if cfg.mmd.era_group_cohorts:
            labels = cohorts[ERA_GROUP].dropna()
            order = [f"{e} - {g}" for e in cfg.analysis.eras for g in self._group_order(cohorts)]
            groups = {label: labels.index[labels == label] for label in order if (labels == label).any()}
            tables.append(analysis(cfg.mmd.era_group_sample_size).run(standardized, groups, "era_group"))
        pd.concat(tables, ignore_index=True).to_csv(self.paths.results / "variance.csv", index=False)

    # ------------------------------------------------------------------------------------
    # Stage 4: figures
    # ------------------------------------------------------------------------------------

    def report(self) -> None:
        from . import visualization as viz

        cfg = self.config
        figures, results, reports = self.paths.figures, self.paths.results, self.paths.reports
        name = cfg.name

        counts_path = reports / "documents_by_year.csv"
        if counts_path.exists():
            counts = pd.read_csv(counts_path, index_col=0).iloc[:, 0]
            viz.save_figure(viz.plot_cohort_sizes(counts, f"{name}: documents per year"),
                            figures / "documents_by_year.png")

        quality_path = reports / "quality_metrics.parquet"
        if quality_path.exists():
            metrics = pd.read_parquet(quality_path)
            viz.save_figure(viz.plot_quality_metrics(metrics, cfg.quality.thresholds),
                            figures / "quality_metrics.png")
            rates = pd.read_csv(reports / "quality_flags_by_year.csv", index_col=0)
            viz.save_figure(viz.plot_flag_rates(rates), figures / "quality_flags_by_year.png")

        matrix_path = results / "mmd_by_year_matrix.csv"
        if matrix_path.exists():
            matrix = pd.read_csv(matrix_path, index_col=0)
            viz.save_figure(viz.plot_mmd_heatmap(matrix, f"{name}: pairwise Biber-feature MMD² across years"),
                            figures / "mmd_by_year_heatmap.png")
            shifts = pd.read_csv(results / "mmd_by_year_feature_shifts.csv")
            if not shifts.empty:
                first, last = int(matrix.index[0]), int(matrix.index[-1])
                viz.save_figure(viz.plot_feature_shifts(shifts, first, last),
                                figures / f"feature_shifts_{first}_{last}.png")

        era_group_path = results / "mmd_by_era_group_matrix.csv"
        if era_group_path.exists():
            matrix = pd.read_csv(era_group_path, index_col=0)
            viz.save_figure(viz.plot_mmd_heatmap(matrix, f"{name}: MMD² across era x {cfg.analysis.group_column}"),
                            figures / "mmd_by_era_group_heatmap.png")

        stability_path = results / "sample_size_stability.csv"
        if stability_path.exists():
            table = pd.read_csv(stability_path)
            for grouping, sub in table.groupby("grouping", sort=False):
                viz.save_figure(viz.plot_stability(sub, cfg.mmd.confidence),
                                figures / f"sample_size_stability_{grouping}.png")

        dimensions_path = results / "biber_dimensions.parquet"
        if dimensions_path.exists():
            self._report_dimensions(pd.read_parquet(dimensions_path), viz)

        variance_path = results / "variance.csv"
        if variance_path.exists():
            table = pd.read_csv(variance_path)
            for grouping, sub in table.groupby("grouping", sort=False):
                viz.save_figure(viz.plot_variance(sub, cfg.mmd.confidence,
                                                  f"{name}: total variance of standardised features ({grouping})"),
                                figures / f"variance_by_{grouping}.png")
        logger.info("Figures written to %s", figures)

    def _report_dimensions(self, scores: pd.DataFrame, viz) -> None:
        cfg = self.config
        dimensions = [c for c in scores.columns if c.startswith("Dim")]
        figures = self.paths.figures
        scale = "Biber 1988 scale" if cfg.dimensions.standardization == "biber1988" else "corpus z-scores"
        if cfg.dimensions.violin_by == YEAR:
            order = sorted(scores[YEAR].unique())
            x, title = YEAR, f"{cfg.name}: Biber dimensions by year ({order[0]}-{order[-1]}), {scale}"
        else:
            order = [str(e) for e in cfg.analysis.eras if str(e) in set(scores[ERA].astype(str))]
            x, title = ERA, f"{cfg.name}: Biber dimensions by era ({' / '.join(order)}), {scale}"
        viz.save_figure(
            viz.plot_dimension_violins(scores, x, order, dimensions, title=title),
            figures / f"biber_dimensions_by_{x}.png",
        )
        if GROUP in scores and cfg.analysis.group_column:
            groups = self._group_order(scores)
            viz.save_figure(
                viz.plot_dimension_violins(scores.dropna(subset=[GROUP]), GROUP, groups, dimensions, ncols=4,
                                           title=f"{cfg.name}: Biber dimensions by {cfg.analysis.group_column}, {scale}"),
                figures / "biber_dimensions_by_group.png",
            )
            if cfg.analysis.eras:
                eras = [str(e) for e in cfg.analysis.eras]
                subset = scores.dropna(subset=[GROUP, ERA])
                viz.save_figure(
                    viz.plot_dimension_violins(subset, ERA, eras, dimensions, hue=GROUP, hue_order=groups,
                                               ncols=4, title=f"{cfg.name}: Biber dimensions by era x "
                                               f"{cfg.analysis.group_column}"),
                    figures / "biber_dimensions_by_era_group.png",
                )

    # ------------------------------------------------------------------------------------

    def _load_corpus(self) -> Corpus:
        if not self.paths.corpus.exists():
            raise FileNotFoundError(f"{self.paths.corpus} not found -- run the 'prepare' stage first")
        return Corpus.load(self.paths.corpus, name=self.config.name)
