"""Configuration of the paraphrase test (``configs/paraphrase.yaml``).

The test spans both corpora and several LLMs, so it has its own config next to the pipeline runs
it reads from: which texts to paraphrase (``sample``), how every model is prompted and decoded
(``generation``) and which open-weight models run (``models``). Loading works as for the pipeline:
unknown keys raise :class:`ConfigError`, and ``--set section.key=value`` overrides any value.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import PERIODS_PER_YEAR, TIME_UNITS, ConfigError, _build, apply_overrides, parse_years

MODEL_ROLES = ("main", "separate", "check")

# Smoke tests, 2026-11-04/05. "Do not shorten ...": with only "approximate length", five of seven
# models condensed the texts to about half. "Sentence by sentence" made gpt-oss alternate copied
# sentences with their paraphrases. "In your own words": without it, the length clause made Qwen3
# return AO3 chapters almost unchanged.
DEFAULT_PROMPT = ("Paraphrase the following text in your own words. Keep its meaning, register and length: "
                  "do not shorten, summarise or leave anything out. Return only the paraphrased text.")


@dataclass
class SampleGroupConfig:
    """One fixed sample: ``per_year`` documents a year from one prepared pipeline run."""

    run: str
    """Pipeline run the documents come from: ``<output_dir>/<run>/`` with its corpus and features."""
    years: list[int] | None = None
    per_year: int = 300
    """Documents drawn per year; a year with fewer eligible documents contributes all of them."""
    where: dict[str, list[Any]] = field(default_factory=dict)
    """Keep documents whose column holds one of the listed values, e.g. ``{rating: [General Audiences]}``.
    Runs with AO3 fandom tags also offer ``canon`` (``lotr`` / ``hobbit``, see :func:`ao3_canon`)."""
    stratify: str | None = None
    """Column whose mix is held at the same shares every year (its pooled shares over ``years``)."""
    spread_over: str | None = None
    """Column grouping related documents (``work_id``): the draw covers as many groups as it can."""
    max_per_spread: int | None = None
    """At most this many documents per ``spread_over`` value and year, even when a stratum runs short."""

    def __post_init__(self) -> None:
        self.years = parse_years(self.years)
        if not self.years:
            raise ConfigError("Every sample group needs years")
        if self.per_year < 1:
            raise ConfigError("Sample groups need per_year >= 1")
        if self.max_per_spread is not None and not self.spread_over:
            raise ConfigError("max_per_spread needs spread_over")
        if not isinstance(self.where, dict):
            raise ConfigError("A sample group's 'where' maps columns to lists of values")
        self.where = {column: list(values) if isinstance(values, (list, tuple)) else [values]
                      for column, values in self.where.items()}


@dataclass
class PilotConfig:
    """A small subset of the sample to try every model on before the main run."""

    per_year: int = 50
    """Texts per corpus and pilot year, split evenly over the corpus's sample groups."""
    years: dict[str, list[int]] = field(default_factory=dict)
    """Pilot years per run, e.g. ``{acl: [2019, 2021, 2023, 2024]}``."""

    def __post_init__(self) -> None:
        self.years = {run: parse_years(years) or [] for run, years in self.years.items()}


@dataclass
class SampleConfig:
    """The texts every model paraphrases (see :mod:`biber_drift.paraphrase.sample`)."""

    seed: int = 0
    groups: dict[str, SampleGroupConfig] = field(default_factory=dict)
    """Sample groups by name, drawn in this order."""
    pilot: PilotConfig = field(default_factory=PilotConfig)

    def __post_init__(self) -> None:
        self.groups = {name: spec if isinstance(spec, SampleGroupConfig)
                       else _build(SampleGroupConfig, spec, f"sample.groups.{name}")
                       for name, spec in self.groups.items()}
        unknown = set(self.pilot.years) - {group.run for group in self.groups.values()}
        if unknown:
            raise ConfigError(f"sample.pilot.years names runs without a sample group: {sorted(unknown)}")


@dataclass
class GenerationConfig:
    """Prompt and decoding, the same for every model."""

    prompt: str = DEFAULT_PROMPT
    """The user message is this instruction, a blank line and the text: no system prompt, year or metadata."""
    temperature: float = 0.0
    """0 = greedy decoding: each model's most likely rewrite, reproducible."""
    max_new_tokens: int = 1200
    """About twice the longest window, so outputs are not cut off."""
    seed: int = 0
    max_model_len: int = 4096
    """Context reserved per sequence: the prompt (about 700 tokens) plus the new tokens."""
    gpu_memory_utilization: float = 0.9
    chunk_size: int = 1000
    """Texts per vLLM call. Each finished chunk is appended to the output file, so a stopped job resumes."""

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ConfigError("generation.temperature must be between 0 and 2")
        if self.chunk_size < 1 or self.max_new_tokens < 1:
            raise ConfigError("generation.chunk_size and generation.max_new_tokens must be >= 1")


@dataclass
class ModelConfig:
    """One open-weight instruct model, run with vLLM from a local Hugging Face snapshot."""

    repo: str
    """Hugging Face repository, e.g. ``Qwen/Qwen3-8B``."""
    family: str | None = None
    role: str = "main"
    """In the analysis: ``main`` counts towards "most families"; ``separate`` is reported on its own
    (Qwen3: it copies most texts, decided 2026-11-05); ``check`` is a robustness check (Llama 2)."""
    revision: str | None = None
    """Branch, tag or commit to download and run (``None`` = main); the commit used is recorded."""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Chat-template switches, e.g. ``enable_thinking: false`` (Qwen3), ``reasoning_effort: low`` (gpt-oss)."""
    current_date: str | None = None
    """``YYYY-MM-DD`` that replaces the day the chat template writes into the prompt (gpt-oss:
    "Current date: ..."), so that sessions on different days render the same prompt."""
    answer_channel: str | None = None
    """Harmony-format models (gpt-oss) answer after reasoning: keep only this channel (``final``)."""
    stop_tokens: list[str] = field(default_factory=list)
    """Extra tokens that end generation (gpt-oss: ``<|return|>``, ``<|call|>``)."""
    max_new_tokens: int | None = None
    """Overrides ``generation.max_new_tokens``, e.g. to leave room for gpt-oss's reasoning."""
    tensor_parallel_size: int = 1
    """GPUs per model (2 for 70B models on 80 GB cards)."""
    engine: dict[str, Any] = field(default_factory=dict)
    """Extra ``vllm.LLM`` arguments, e.g. ``tokenizer_mode: hf`` or ``limit_mm_per_prompt: {image: 0}``."""

    def __post_init__(self) -> None:
        if self.role not in MODEL_ROLES:
            raise ConfigError(f"{self.repo}: role must be one of {list(MODEL_ROLES)}, not {self.role!r}")
        if isinstance(self.current_date, date):     # YAML reads an unquoted 2026-11-05 as a date
            self.current_date = self.current_date.isoformat()
        if self.current_date is not None:
            try:
                date.fromisoformat(self.current_date)
            except (TypeError, ValueError):
                raise ConfigError(f"{self.repo}: current_date must be YYYY-MM-DD, "
                                  f"not {self.current_date!r}") from None


@dataclass
class CheckConfig:
    """Diagnostics of the model outputs (``biber-drift paraphrase check``).

    The check only counts; what gets cleaned or dropped is decided after the pilot. Every text flag
    compares the output with its original, so a chapter that itself opens with "Sure," or quotes
    is not counted as a chatty or wrapped output.
    """

    head_chars: int = 300
    """Refusals are looked for in the first characters of the output only (dialogue later on is fine)."""
    loop_ngram: int = 8
    """Word n-gram length of the repetition check."""
    loop_repeats: int = 3
    """An output loops when one n-gram occurs at least this often, and more often than in the original."""
    short_ratio: float = 0.5
    """Provisional: an output with fewer words than this share of the original's counts as short."""
    long_ratio: float = 1.5
    """Provisional: an output with more words than this share of the original's counts as long."""
    copy_ngram: int = 8
    """Word n-gram length of the overlap measure: the share of the original's n-grams kept in the output."""
    copy_share: float = 0.8
    """Provisional: an output that keeps at least this share of the original's n-grams counts as a copy."""
    examples_seed: int = 0
    """Seed of the examples shown in the terminal (``--show``)."""

    def __post_init__(self) -> None:
        if self.loop_ngram < 2 or self.loop_repeats < 2:
            raise ConfigError("check.loop_ngram and check.loop_repeats must be >= 2")
        if not 0 < self.short_ratio < 1 < self.long_ratio:
            raise ConfigError("check needs 0 < short_ratio < 1 < long_ratio")
        if self.copy_ngram < 2 or not 0 < self.copy_share <= 1:
            raise ConfigError("check needs copy_ngram >= 2 and 0 < copy_share <= 1")


@dataclass
class FilterConfig:
    """What happens to the outputs before tagging (``biber-drift paraphrase tag``).

    Frozen on 2026-11-05 from the main run's check, before any comparison between years. ``short``, ``long`` and ``copy`` stay in on purpose: they are amounts of change, the
    thing measured, so dropping them would filter on the result (a sensitivity check can).
    """

    drop: list[str] = field(default_factory=lambda: ["missing", "cut_off", "refusal", "think", "template", "loop"])
    """Check flags that mean the output is no paraphrase; these texts are left out (<= 0.5% per model)."""
    clean: list[str] = field(default_factory=lambda: ["opener", "closer", "quotes", "markdown"])
    """Boilerplate stripped from the output, which then stays (see ``paraphrase/clean.py``)."""
    window_words: int = 400
    window_cap: int = 440
    """Outputs are cut like the originals: at the first sentence end after ``window_words`` words,
    never past ``window_cap``; shorter outputs are kept whole."""
    min_words: int = 0
    """No 400-word minimum for paraphrases: it would keep 6-54% of the outputs, a share that varies
    by year (highest in ACL 2024). The type-token ratio is made length-robust instead (``tagging``)."""

    def __post_init__(self) -> None:
        if not 0 < self.window_words <= self.window_cap or self.min_words < 0:
            raise ConfigError("filters need 0 < window_words <= window_cap and min_words >= 0")


@dataclass
class TaggingConfig:
    """Biber tagging of the cleaned outputs and of the originals, in the same environment."""

    spacy_model: str = "en_core_web_trf"
    batch_size: int = 100
    gpu: str = "auto"
    """``auto``: a CUDA GPU when there is one, else the CPU; ``require``; ``off``."""
    mattr_window: int | None = 100
    """``f_43_type_token`` is the moving-average type-token ratio over windows of this many
    alphabetic tokens, for originals and paraphrases alike, since many paraphrases are shorter than
    the originals (NaN under one window). Computed per text: pybiber's own MATTR depends on the
    batch. ``null``: the plain ratio, as in the pipeline runs (400-440 words)."""

    def __post_init__(self) -> None:
        if self.gpu not in {"auto", "require", "off"}:
            raise ConfigError("tagging.gpu must be 'auto', 'require' or 'off'")
        if self.batch_size < 1 or (self.mattr_window is not None and self.mattr_window < 2):
            raise ConfigError("tagging needs batch_size >= 1 and mattr_window >= 2 (or null)")


@dataclass
class SeriesConfig:
    """How one sample group's change is followed over time: its periods and the break."""

    time_unit: str = "year"
    """``year``, or ``quarter`` / ``month`` from ``date_column`` of the group's pipeline run."""
    date_column: str | None = None
    """Corpus column with each document's date, joined by ``doc_id`` (``chapter_date`` for AO3)."""
    break_at: int | str = 2023
    """First post-break period: ``2023``, ``"2023Q1"`` or ``"2022-12"``."""
    min_documents: int = 30
    """Periods with fewer kept texts are left out of the time series."""
    hac_lags: int | None = None
    """Newey-West lags of the time series (default: one year, i.e. 0 / 4 / 12 for years, quarters, months)."""
    min_segment: int | None = None
    """Periods required on each side of a placebo break (default: 2 years / 4 quarters / 12 months)."""

    def __post_init__(self) -> None:
        if self.time_unit not in TIME_UNITS:
            raise ConfigError(f"analysis.series: time_unit must be one of {list(TIME_UNITS)}")
        dated = self.time_unit != "year"
        if dated and not self.date_column:
            raise ConfigError(f"analysis.series: time_unit {self.time_unit} needs date_column")
        if self.hac_lags is None:
            self.hac_lags = PERIODS_PER_YEAR[self.time_unit] if dated else 0
        if self.min_segment is None:
            self.min_segment = PERIODS_PER_YEAR[self.time_unit] if dated else 2


@dataclass
class CohortConfig:
    """Cohort MMD^2 per year between the originals and their paraphrases (``analysis.mmd``)."""

    sizes: dict[str, int] = field(default_factory=dict)
    """Texts per cohort by sample group; a group without a size gets no MMD. Every group-year gets
    one fixed sample, the same texts for every model, from the texts that every model kept with
    complete features. MMD^2 values compare only at the same size."""
    subsample_draws: int = 1000
    """The fixed sample is the most representative of this many random draws: the originals'
    Dimension 1-6 distributions closest to those of all eligible texts, as in the pipeline."""
    permutations: int = 1000
    """Null draws: original and paraphrase swap places within randomly chosen pairs."""

    def __post_init__(self) -> None:
        if any(size < 2 for size in self.sizes.values()) or self.subsample_draws < 1 or self.permutations < 1:
            raise ConfigError("analysis.mmd needs sizes >= 2, subsample_draws >= 1 and permutations >= 1")


@dataclass
class AnalysisConfig:
    """How much each model changes the texts, and whether that drops after the break
    (``biber-drift paraphrase analyze``)."""

    series: dict[str, SeriesConfig] = field(default_factory=dict)
    """Time series per sample group (default: yearly with a 2023 break)."""
    exclude_flags: list[str] = field(default_factory=lambda: ["short", "long", "copy"])
    """Sensitivity check: every statistic again without the outputs carrying one of these check flags."""
    length_covariate: str = "length_ratio"
    """Item column whose log enters the regressions (output words / original words); each
    regression is also fitted without it."""
    bootstrap_resamples: int = 1000
    """Resamples of the yearly means' intervals (texts are drawn again within each stratum)."""
    confidence: float = 95.0
    p_adjust: str = "fdr_bh"
    """``fdr_bh``, ``holm`` or ``none``: across models x groups x measures (time series) or corpora x
    models x measures (regressions), within each variant."""
    mmd: CohortConfig = field(default_factory=CohortConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        # p_adjust is checked by the analysis: importing it here would need statsmodels in the vLLM env
        self.series = {group: spec if isinstance(spec, SeriesConfig)
                       else _build(SeriesConfig, spec, f"analysis.series.{group}")
                       for group, spec in self.series.items()}
        if self.bootstrap_resamples < 1 or not 50 <= self.confidence < 100:
            raise ConfigError("analysis needs bootstrap_resamples >= 1 and a confidence in percent (e.g. 95)")


@dataclass
class ParaphraseConfig:
    """Everything the paraphrase test needs."""

    name: str = "paraphrase"
    output_dir: str = "outputs"
    sample: SampleConfig = field(default_factory=SampleConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    """Models by key, in order: a Slurm array index picks the model at that position."""
    check: CheckConfig = field(default_factory=CheckConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self) -> None:
        self.models = {key: spec if isinstance(spec, ModelConfig) else _build(ModelConfig, spec, f"models.{key}")
                       for key, spec in self.models.items()}
        unknown = (set(self.analysis.series) | set(self.analysis.mmd.sizes)) - set(self.sample.groups)
        if unknown:
            raise ConfigError(f"analysis.series or analysis.mmd.sizes name groups without a sample group: "
                              f"{sorted(unknown)}")

    def series(self, group: str) -> SeriesConfig:
        """The time-series settings of one sample group (yearly, break 2023, unless configured)."""
        return self.analysis.series.get(group) or SeriesConfig()

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def model_key(self, key_or_index: str) -> str:
        """A model key, given the key itself or its position in ``models`` (``"0"`` = the first)."""
        keys = list(self.models)
        if key_or_index in self.models:
            return key_or_index
        if key_or_index.isdigit() and int(key_or_index) < len(keys):
            return keys[int(key_or_index)]
        raise ConfigError(f"Unknown model {key_or_index!r}: give a key or an index 0-{len(keys) - 1} of {keys}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ParaphrasePaths:
    """On-disk layout of the paraphrase test (``<output_dir>/<name>/``)."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def sample(self) -> Path:
        return self.root / "sample.parquet"

    @property
    def sample_counts(self) -> Path:
        return self.root / "sample_counts.csv"

    def generations(self, name: str) -> Path:
        """Folder of one generation run (``pilot``, ``main``, or a named check)."""
        return self.root / "generations" / name

    def check(self, name: str) -> Path:
        """Diagnostics of one generation run: ids, flags and numbers only, never texts."""
        return self.generations(name) / "check"

    def features(self, name: str) -> Path:
        """Biber features of one generation run's cleaned outputs and of their originals."""
        return self.root / "features" / name

    def analysis(self, name: str) -> Path:
        """The comparison of one generation run's originals and outputs: ids and numbers only."""
        return self.root / "analysis" / name


def load_paraphrase_config(path: str | Path | None = None,
                           overrides: list[str] | None = None) -> ParaphraseConfig:
    """Load the paraphrase config (optional YAML file) and apply command-line overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
    values = apply_overrides(values, overrides or [])
    return _build(ParaphraseConfig, values, "")
