"""Typed configuration for a pipeline run.

A run is described by one YAML file (see ``configs/``). The file is parsed into the nested
dataclasses below; unknown keys raise :class:`ConfigError` so a typo cannot silently fall
back to a default. Any value can be overridden from the command line with ``--set
section.key=value`` (the value is parsed as YAML, so ``--set mmd.sample_size=300`` gives an
int and ``--set data.years=[2019,2024]`` a list).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for invalid or unknown configuration values."""


def parse_years(value: Any) -> list[int] | None:
    """Normalise a year specification.

    Accepts ``None``, a list of years, a ``"2009-2024"`` range string, or a mapping with
    ``start``/``end`` (both inclusive).
    """
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        start, sep, end = value.partition("-")
        if not sep:
            return [int(start)]
        return list(range(int(start), int(end) + 1))
    if isinstance(value, dict):
        unknown = set(value) - {"start", "end"}
        if unknown:
            raise ConfigError(f"Year range only takes 'start' and 'end', got {sorted(unknown)}")
        return list(range(int(value["start"]), int(value["end"]) + 1))
    if isinstance(value, (list, tuple, range)):
        return sorted({int(year) for year in value})
    raise ConfigError(f"Cannot interpret {value!r} as a list of years")


def parse_sample_sizes(value: Any) -> list[int]:
    """Normalise sample sizes: a list, or a mapping with ``start``/``stop``/``step`` (inclusive)."""
    if isinstance(value, dict):
        unknown = set(value) - {"start", "stop", "step"}
        if unknown:
            raise ConfigError(f"Sample sizes only take start/stop/step, got {sorted(unknown)}")
        return list(range(int(value["start"]), int(value["stop"]) + 1, int(value.get("step", 50))))
    if isinstance(value, (list, tuple, range)):
        return sorted({int(n) for n in value})
    raise ConfigError(f"Cannot interpret {value!r} as sample sizes")


# Malformed-text thresholds, set around the 99th percentile of each metric on the ACL corpus.
DEFAULT_QUALITY_THRESHOLDS: dict[str, float] = {
    "punct_per_word": 0.35,
    "nonascii_per_1k": 10.0,
    "nonlatin_share": 0.01,
    "invisible_chars": 0.0,
    "blob_chars": 0.0,
    "garble_per_1k": 5.0,
    "oov_share": 0.10,
}


@dataclass
class DataConfig:
    """Where the raw corpus lives and how to read it."""

    dataset: str = "ao3"
    """Registered loader name (``ao3`` or ``acl``)."""
    path: str = "data/raw"
    """CSV file or directory of CSV files."""
    years: list[int] | None = None
    """Keep only documents from these years (``None`` keeps every year)."""
    loader: dict[str, Any] = field(default_factory=dict)
    """Loader-specific keyword arguments (see the loader classes)."""

    def __post_init__(self) -> None:
        self.years = parse_years(self.years)


@dataclass
class CleaningConfig:
    """Dataset-specific text cleaning (see :mod:`biber_drift.preprocessing`)."""

    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the dataset's cleaner, e.g. ``min_words`` or ``disable``."""


@dataclass
class QualityConfig:
    """Malformed-text diagnostics run on the cleaned text."""

    enabled: bool = True
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUALITY_THRESHOLDS))
    min_vocab_docs: int = 20
    """A word counts as known if it occurs in at least this many documents."""
    drop_flags: list[str] = field(default_factory=list)
    """Drop documents carrying any of these flags (metric names); empty keeps everything."""

    def __post_init__(self) -> None:
        unknown = set(self.thresholds) - set(DEFAULT_QUALITY_THRESHOLDS)
        if unknown:
            raise ConfigError(f"Unknown quality metrics: {sorted(unknown)}")
        self.thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **self.thresholds}
        bad = set(self.drop_flags) - set(DEFAULT_QUALITY_THRESHOLDS)
        if bad:
            raise ConfigError(f"Unknown quality flags in drop_flags: {sorted(bad)}")


@dataclass
class WindowConfig:
    """Fixed-length text windows, so feature rates are comparable across documents."""

    enabled: bool = True
    target_words: int = 400
    hard_cap: int = 440
    min_words: int | None = None
    """Drop documents whose window is shorter than this (defaults to ``target_words``)."""

    def __post_init__(self) -> None:
        if self.hard_cap < self.target_words:
            raise ConfigError("window.hard_cap must be >= window.target_words")
        if self.min_words is None:
            self.min_words = self.target_words


@dataclass
class FeatureConfig:
    """Biber feature extraction with pybiber + spaCy."""

    spacy_model: str = "en_core_web_sm"
    batch_size: int = 100
    n_process: int = 1
    gpu: str = "auto"
    """``auto``: use a CUDA GPU when there is one, else the CPU; ``require``: use any GPU spaCy
    supports (also Apple MPS, which was slower than the CPU for en_core_web_trf); ``off``: CPU."""
    exclude: list[str] = field(default_factory=lambda: ["f_62_split_infinitive"])
    """Features left out of every analysis (f_62 is almost always zero)."""

    def __post_init__(self) -> None:
        if self.gpu not in {"auto", "require", "off"}:
            raise ConfigError("features.gpu must be 'auto', 'require' or 'off'")


@dataclass
class AnalysisConfig:
    """Cohort definitions shared by the analyses."""

    eras: list[int] = field(default_factory=list)
    """Key years for era-based summaries (e.g. pre-deep-learning / pre-LLM / post-LLM)."""
    group_column: str | None = None
    """Optional second factor, e.g. ``venue_tier`` for ACL."""
    group_order: list[str] | None = None

    def __post_init__(self) -> None:
        self.eras = parse_years(self.eras) or []


@dataclass
class MMDConfig:
    """Pairwise RBF-kernel MMD^2 between cohorts, on fixed samples as in Nieth et al.

    Every cohort is represented by ONE fixed sample of ``sample_size`` documents (the paper's
    600-document subsample). The observed MMD^2 of a pair is computed on the two fixed samples
    in full; bootstrap and permutation test resample those same samples. Cohorts smaller than
    ``sample_size`` are dropped, so every pair is compared at the same n.
    """

    enabled: bool = True
    years: list[int] | None = None
    """Years compared pairwise (``None`` = every year in the corpus)."""
    sample_size: int = 600
    subsample: str = "representative"
    """``representative``: of ``subsample_draws`` random draws keep the one whose Biber-dimension
    distributions are closest (mean Wasserstein distance) to the full cohort, as the paper's
    ``03_subsample_data.py`` does; ``random``: one random draw."""
    subsample_draws: int = 1000
    bootstrap_resamples: int = 500
    permutations: int = 1000
    min_cohort_documents: int | None = None
    """Cohorts below this size are dropped (defaults to ``sample_size``, the minimum for a fixed
    sample). A within-cohort baseline additionally needs ``2 * sample_size`` documents."""
    confidence: float = 95.0
    bandwidth_max_points: int = 5000
    """Documents sub-sampled for the median-heuristic bandwidth of the run's reference."""
    era_group_cohorts: bool = False
    """Also compare every (era, group) cohort pair; needs ``analysis.eras`` and ``group_column``."""
    era_group_sample_size: int | None = None
    """Fixed-sample size for the (era, group) cohorts, which are smaller than years (defaults to
    ``sample_size``). MMD^2 values are only comparable within one sample size."""
    seed: int = 42

    def __post_init__(self) -> None:
        self.years = parse_years(self.years)
        if self.era_group_sample_size is None:
            self.era_group_sample_size = self.sample_size
        if self.subsample not in {"representative", "random"}:
            raise ConfigError("mmd.subsample must be 'representative' or 'random'")
        if self.min_cohort_documents is None:
            self.min_cohort_documents = self.sample_size
        if self.min_cohort_documents < self.sample_size:
            raise ConfigError("mmd.min_cohort_documents cannot be smaller than mmd.sample_size")


@dataclass
class StabilityConfig:
    """Within-cohort MMD^2 as a function of sample size (the 'elbow' plot)."""

    enabled: bool = True
    groupings: list[str] = field(default_factory=lambda: ["all"])
    """Any of ``all`` (whole corpus), ``era``, ``group``, ``era_group``."""
    sample_sizes: list[int] = field(default_factory=lambda: list(range(100, 1001, 50)))
    resamples: int = 100
    scaling: str = "pooled"
    """``pooled``: the run's one scaler and sigma, fitted on the full corpus -- the same ruler as the
    cohort comparisons (the paper's choice); ``per_group``: every panel standardised and its sigma
    estimated on its own documents (curves not comparable with the comparisons)."""
    seed: int = 42

    def __post_init__(self) -> None:
        self.sample_sizes = parse_sample_sizes(self.sample_sizes)
        allowed = {"all", "era", "group", "era_group"}
        if set(self.groupings) - allowed:
            raise ConfigError(f"stability.groupings must be a subset of {sorted(allowed)}")
        if self.scaling not in {"pooled", "per_group"}:
            raise ConfigError("stability.scaling must be 'pooled' or 'per_group'")


@dataclass
class DimensionConfig:
    """Biber (1988) dimension scores and violin plots."""

    enabled: bool = True
    years: list[int] | None = None
    """Documents scored and plotted (``None`` = every year)."""
    violin_by: str = "year"
    """``year``: one violin per year; ``era``: one violin per era in ``analysis.eras``."""
    standardization: str = "corpus"
    """``corpus``: z-scores against the documents scored; ``biber1988``: against Biber's 1988
    norms, as in Nieth et al. (scores on Biber's scale, comparable with the paper)."""
    include_seem_appear: bool = True
    """Keep Dimension 7 (SEEM / APPEAR, one feature). The paper reports Dimensions 1-6."""

    def __post_init__(self) -> None:
        self.years = parse_years(self.years)
        if self.violin_by not in {"year", "era"}:
            raise ConfigError("dimensions.violin_by must be 'year' or 'era'")
        if self.standardization not in {"corpus", "biber1988"}:
            raise ConfigError("dimensions.standardization must be 'corpus' or 'biber1988'")


@dataclass
class VarianceConfig:
    """Total variance of the standardised features per cohort (the paper's variance analysis).

    The total variance is the sum over features of each feature's variance after z-scoring
    against the pooled corpus, so every feature counts equally. Besides the value on all of a
    cohort's documents, it is resampled on ``sample_size`` documents (without replacement) to give
    comparable-at-equal-n means and confidence intervals.
    """

    enabled: bool = True
    sample_size: int | None = None
    """Documents per resample. ``None`` follows the MMD analysis: ``mmd.sample_size`` for years,
    ``mmd.era_group_sample_size`` for (era, group) cohorts."""
    resamples: int = 500
    seed: int = 42


@dataclass
class FeatureChangeConfig:
    """Absolute / relative / combined change of every feature between key years."""

    enabled: bool = True
    years: list[int] | None = None
    """Years compared (consecutive pairs plus first->last); defaults to ``analysis.eras``."""
    by_group: bool = False
    """Also run the comparison inside each level of ``analysis.group_column``."""
    absolute_weight: float = 0.5
    relative_weight: float = 0.5
    top_k: int = 15

    def __post_init__(self) -> None:
        self.years = parse_years(self.years)


@dataclass
class ProjectConfig:
    """Everything needed to run the pipeline once."""

    name: str = "run"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    mmd: MMDConfig = field(default_factory=MMDConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    dimensions: DimensionConfig = field(default_factory=DimensionConfig)
    feature_change: FeatureChangeConfig = field(default_factory=FeatureChangeConfig)
    variance: VarianceConfig = field(default_factory=VarianceConfig)

    def __post_init__(self) -> None:
        if self.feature_change.years is None:
            self.feature_change.years = list(self.analysis.eras)
        if self.mmd.era_group_cohorts and not (self.analysis.eras and self.analysis.group_column):
            raise ConfigError("mmd.era_group_cohorts needs analysis.eras and analysis.group_column")
        needs_group = {"group", "era_group"} & set(self.stability.groupings)
        if needs_group and not self.analysis.group_column:
            raise ConfigError(f"stability.groupings {sorted(needs_group)} need analysis.group_column")
        needs_eras = {"era", "era_group"} & set(self.stability.groupings)
        if needs_eras and not self.analysis.eras:
            raise ConfigError(f"stability.groupings {sorted(needs_eras)} need analysis.eras")
        if self.dimensions.violin_by == "era" and not self.analysis.eras:
            raise ConfigError("dimensions.violin_by: era needs analysis.eras")
        if self.feature_change.by_group and not self.analysis.group_column:
            raise ConfigError("feature_change.by_group needs analysis.group_column")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, values: Any, where: str) -> Any:
    """Recursively build dataclass ``cls`` from a mapping, rejecting unknown keys."""
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ConfigError(f"'{where or 'config'}' must be a mapping, got {type(values).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where or 'config'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        path = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, path)
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive, fields are validated above
        raise ConfigError(f"Invalid values in '{where or 'config'}': {exc}") from exc


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides to a nested mapping (values parsed as YAML)."""
    values = copy.deepcopy(values)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"Override must look like 'section.key=value', got {override!r}")
        parts = key.strip().split(".")
        node = values
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot set {key!r}: '{part}' is not a section")
        node[parts[-1]] = yaml.safe_load(raw)
    return values


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ProjectConfig:
    """Load a YAML config file (optional) and apply command-line overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
    values = apply_overrides(values, overrides or [])
    return _build(ProjectConfig, values, "")


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Write the fully resolved config next to the run outputs, for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
