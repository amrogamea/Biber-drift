# Biber-drift

Measure how the **linguistic style of a register drifts over time**, using Biber's (1988)
multidimensional feature set and kernel two-sample tests.

The project ships two case studies:

| dataset | documents | question |
|---|---|---|
| **ACL** (S2ORC) | ACL Anthology papers, 2000–2026, abstract + introduction | Has academic NLP writing changed since deep learning and LLMs (2000 / 2019 / 2024), and does it differ by venue tier (A\* / Conference / Workshop)? |
| **AO3** | Archive of Our Own fan-fiction chapters, 2009–2026 (Tolkien fandoms) | Has fan-fiction prose changed year over year? |

Each document is turned into the 67 [pybiber](https://github.com/browndw/pybiber) features,
standardised, and cohorts of documents (years, eras, venue tiers) are compared with an
RBF-kernel **maximum mean discrepancy (MMD²)**, with within-cohort baselines, bootstrap
confidence intervals and permutation tests.

---

## Quick start

```bash
git clone <this repository> && cd biber-drift
# 1. put your data under data/raw/ (see data/README.md), or collect it -- see "Getting data"
# 2. adjust the settings at the top of run.sh (or the YAML config)
./run.sh                    # AO3 (default)
DATASET=acl ./run.sh        # ACL
```

On first use `run.sh` creates `.venv/`, installs the package and downloads the spaCy model.
Everything after `./run.sh` is passed to the CLI:

```bash
./run.sh --stages analyze,report            # re-run the statistics only
./run.sh --set mmd.sample_size=300          # override any config value
DATA_PATH=/path/to/csvs RUN_NAME=ao3_v2 ./run.sh
```

Results are written to `outputs/<run name>/` (see [Outputs](#outputs)).

### Without the script

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_trf
biber-drift run --config configs/ao3.yaml
```

Requires Python ≥ 3.10. Both configs tag with `en_core_web_trf` (the paper's tagger): about 3
400-word windows per second on a laptop CPU -- roughly 13 h for the AO3 corpus and 3 h for ACL --
and much faster on a CUDA GPU (`features.gpu: auto` uses one when present). For a quick look,
`--set features.spacy_model=en_core_web_sm` is ~6x faster but its features are not comparable.

---

## Pipeline

```
 raw CSVs ──► prepare ──────────────► features ─────────► analyze ─────────────► report
             load (loader)            spaCy + pybiber      cohort MMD²            figures
             clean (cleaner)          (resumable cache)    sample-size stability
             quality diagnostics                           Biber dimensions
             400-word window                               feature-change ranking
                                                           interrupted time series
```

| stage | what happens | writes |
|---|---|---|
| `prepare` | Load the raw data into a `Corpus` (unique `doc_id`, `year`, `text`); apply the dataset's cleaning steps; compute malformed-text diagnostics; cut each text to the analysis window. | `corpus/` |
| `features` | Tag every text with the 67 Biber features (pybiber + spaCy). Finished batches are cached, so an interrupted run resumes. | `features/` |
| `analyze` | Pairwise MMD² between years (and era × group cohorts), the sample-size "elbow", Biber dimension scores, the per-feature change ranking, and interrupted time series. | `results/` |
| `report` | Figures for all of the above. | `figures/` |

Every stage reads its inputs from the run directory, so stages can be re-run independently.

### Cleaning

Each cleaner reports, per step, how many documents it changed or dropped
(`corpus/cleaning_report.csv`). Steps can be switched off with
`cleaning.options.disable: [<step>]`.

* **ACL** (`acl`): missing space after a sentence end (`corpus.This`), whitespace, licence
  footers / arXiv stamps, URLs and e-mails, author-year and numeric citations, footnote markers,
  affiliation symbols, table cells read as prose, duplicate papers, papers under 200 words.
  Each paper is represented by its abstract + introduction, cut to the same 400–440-word window
  as AO3 (the paper uses windowed introductions): full papers grow from a median 804 words (2000)
  to 1,301 (2024) and change their section mix, which would leak into the rates and the
  type/token ratio.
* **AO3** (`ao3`): chapters duplicated across fandom files, podfics, the scraped
  `Chapter Text` heading, author notes / metadata headers / chapter headings, scene-break lines,
  URLs and emoji, markdown asterisks.

### Quality diagnostics

Seven per-document signals of malformed text (punctuation per word, non-ASCII and non-Latin
characters, invisible characters, leaked image data, OCR-garbled words, corpus-relative rare-word
rate). Flags are reported per year (`corpus/quality_flags_by_year.csv`) and, optionally, dropped
with `quality.drop_flags`. In the ACL corpus 23% of the year-2000 papers carry a flag (most of
them OCR'd scans) against 1-4% in every other year -- a year-correlated artifact worth
controlling for. The thresholds were chosen by inspecting flagged documents near the 99th
percentile of each metric; they are heuristics, not validated cut-offs.

### Statistics

The comparison follows Nieth et al. (2026):

* **One ruler** -- ONE scaler (feature means / sds) and ONE kernel bandwidth are fitted on all
  documents of the run and reused by every analysis: the year and era x group comparisons, their
  baselines, the sample-size curves and the variance analysis (`results/reference.json`). MMD²
  depends on both, so this keeps every number comparable -- as in the paper.
* **Kernel** -- `k(x, y) = exp(-‖x − y‖² / 2σ²)`, σ from the median heuristic
  `σ = √(median ‖xᵢ − xⱼ‖² / 2)` on (up to 5,000) of the run's standardised documents.
* **Fixed samples** -- every cohort is represented by ONE fixed sample of `sample_size` documents
  (the paper's 600-document subsample): the most representative of `subsample_draws` random draws
  (closest Biber-dimension distributions to the full cohort), or one random draw. Cohorts smaller
  than `sample_size` are skipped, so every pair is compared at the same n. The samples are saved.
* **MMD²** -- the biased (V-statistic) estimator, observed on the two fixed samples in full.
* **Bootstrap CI** -- with-replacement resamples of the fixed samples, each cohort resampled
  independently (the paper draws the same indices for both samples because its human and model
  texts are paired; cohorts of different years are not). Duplicates raise the biased MMD², so the
  bootstrap distribution sits above the observed value: read its width, not its location.
* **Permutation test** -- on the fixed samples; one-sided p-value with the +1 correction.
* **Baselines** -- MMD² between two *disjoint* random halves at the same n, from the whole corpus
  (the paper's human-human baseline) and from each cohort with ≥ 2n documents. Each pair gets an
  effect size `(MMD² − baseline mean) / baseline sd` and is flagged when it exceeds the baselines'
  CI upper bounds.
* **Sample-size stability** -- within-cohort MMD² against sample size; choose `sample_size`
  past the elbow.
* **Total variance** -- the sum of per-feature variances of the *standardised* features per cohort
  (raw rates would be dominated by nouns and prepositions), resampled at n.
* **Biber dimensions** -- Biber (1988), Table 9 loadings; z-scores against the corpus or, as in the
  paper, against Biber's 1988 norms (`dimensions.standardization: biber1988`, Dimensions 1-6).
* **Feature change** -- for key years (`feature_change.years`, e.g. 2009 → 2019 → 2025) the
  absolute and relative change of every feature, min-max scaled and combined into one ranking
  score. `figures/feature_shifts_<years>.png` shows the largest per-feature shifts for each step and
  the full span, on one scale, coloured by whether the feature's mean rose or fell.
* **Multiple testing** -- every family of tests (all year pairs, all series of the interrupted time
  series) gets adjusted p-values (`analysis.p_adjust`: Benjamini-Hochberg FDR by default, or Holm)
  next to the raw ones, e.g. `p_adjusted` in `mmd_by_year_pairwise.csv`.

Beyond the paper, one analysis tests *when* and *how* the style changed around a break
(`trend.break_at`, e.g. ChatGPT's release):

* **Interrupted time series** (`trend`) -- one series per standardised feature,
  Biber dimension and the total variance: the mean per year, or per month for dated documents
  (Newey-West errors). The segmented regression `y = b0 + b1·t + b2·post (+ b3·post·t)` gives the jump
  at the break beyond the pre-break trend (`b2`) and the change of slope (`b3`, with ≥ 3 post-break
  periods); placebo breaks inside the pre-break period show how often such a jump appears by chance.
  `post_departure` (the mean gap between the fitted post-break line and the pre-break trend
  continued) measures how much a series changed, jumps and bends alike.
  `figures/trend_series_dimensions.png` shows total variance and all seven Biber dimensions;
  `figures/trend_series_features.png` the `trend.feature_panels` (8) features with the largest
  |`post_departure`|.

---

## Getting data

The analysis expects the CSV formats produced by the two acquisition commands (described in
[`data/README.md`](data/README.md)); any data in the same format works.

**AO3** -- one CSV per fandom tag, one row per chapter:

```bash
biber-drift acquire-ao3 --fandom-tag "The Hobbit - J. R. R. Tolkien" --start-year 2009 --end-year 2026
```

The scraper is deliberately slow (4–7 s between requests), stops as soon as AO3 signals
automated traffic, and resumes from its checkpoint when re-run with the same arguments.

**ACL** -- papers from the Semantic Scholar S2ORC release (needs an API key):

```bash
cp .env.example .env        # then put your key in .env
set -a; source .env; set +a
biber-drift acquire-acl --years 2000-2026 --output-dir data/raw/acl
```

---

## Configuration

A run is one YAML file (`configs/ao3.yaml`, `configs/acl.yaml`). Unknown keys are rejected, so a
typo cannot silently fall back to a default. Any key can be overridden with
`--set section.key=value` (values are parsed as YAML).

| section | key settings |
|---|---|
| `data` | `dataset` (`ao3`/`acl`), `path`, `years`, loader options |
| `cleaning` | `enabled`, `options` (`min_words`, `disable`) |
| `quality` | `thresholds`, `min_vocab_docs`, `drop_flags` |
| `window` | `enabled`, `target_words` (400), `hard_cap` (440) |
| `features` | `spacy_model`, `batch_size`, `n_process`, `gpu` (`auto`/`require`/`off`), `exclude` |
| `analysis` | `eras` (key years), `group_column` (e.g. `venue_tier`), `group_order`, `p_adjust` (`fdr_bh`/`holm`/`none`) |
| `mmd` | `years`, `sample_size`, `subsample` (`representative`/`random`), `subsample_draws`, `bootstrap_resamples`, `permutations`, `min_cohort_documents`, `era_group_cohorts`, `era_group_sample_size`, `seed` |
| `stability` | `groupings` (`all`/`era`/`group`/`era_group`), `sample_sizes`, `resamples`, `scaling` (`pooled`: the run's ruler; `per_group`) |
| `dimensions` | `years`, `violin_by` (`year`/`era`), `standardization` (`corpus`/`biber1988`), `include_seem_appear` |
| `feature_change` | `years`, `by_group`, `absolute_weight`, `relative_weight` |
| `variance` | `enabled`, `sample_size` (default: the MMD sample sizes), `resamples` |
| `trend` | `enabled`, `time_unit` (`year`/`month`), `date_column`, `break_at` (`2023` or `"2022-12"`), `years`, `min_documents`, `series`, `include_seem_appear`, `feature_panels`, `hac_lags`, `min_segment` |

Years can be written as a list (`[2009, 2019, 2024]`), a range (`{start: 2009, end: 2024}`) or a
string (`"2009-2024"`). The fully resolved config is saved as `outputs/<name>/config.yaml`.

---

## Outputs

```
outputs/<name>/
├── config.yaml                          resolved configuration of the run
├── corpus/
│   ├── corpus.parquet                   cleaned documents + analysis windows
│   ├── cleaning_report.csv              per cleaning step: documents changed / dropped
│   ├── documents_by_year[_raw].csv
│   ├── quality_metrics.parquet          per-document malformed-text metrics and flags
│   └── quality_flags_by_year.csv
├── features/biber_features.parquet      67 Biber features per document (+ .json: the spaCy model)
├── results/
│   ├── reference.json, reference_scaler.csv   the run's scaler and kernel bandwidth
│   ├── mmd_by_year_{pairwise,baselines,pooled_baseline,samples,matrix,interpretation,feature_shifts}.csv
│   ├── mmd_by_era_group_*.csv           (if mmd.era_group_cohorts)
│   ├── variance.csv                     total variance of standardised features per cohort
│   ├── sample_size_stability.csv
│   ├── biber_dimensions.parquet         Biber dimension scores per document
│   ├── feature_change/<all|group=...>/  means, changes, scaled tables, combined scores
│   └── trend/                           series, its, its_placebos
└── figures/                             heatmaps, elbow plots, dimension violins, variance, diagnostics,
                                         interrupted time series
```

---

## Project structure

```
biber-drift/
├── run.sh                       one-command entry point
├── configs/                     ao3.yaml, acl.yaml, acl_venue_tiers.csv
├── data/                        your data (not committed); format description
├── src/biber_drift/
│   ├── config.py                typed configuration (dataclasses) + YAML loading
│   ├── corpus.py                Corpus: the document table passed between stages
│   ├── pipeline.py              Pipeline: prepare -> features -> analyze -> report
│   ├── cli.py                   `biber-drift` command
│   ├── loaders/                 CorpusLoader registry: AO3Loader, ACLLoader, VenueTierClassifier
│   ├── preprocessing/           CorpusCleaner (ACL, AO3), QualityDiagnostics, TextWindower
│   ├── features/                BiberTagger, FeatureMatrix, BiberDimensionScorer
│   ├── analysis/                RBFKernelMMD, CohortMMDAnalysis, SampleSizeStability,
│   │                            FeatureChangeAnalysis
│   ├── visualization.py         figures
│   └── acquisition/             AO3 scraper, S2ORC ACL extractor
└── tests/                       pytest suite (synthetic data, no model download needed)
```

### Adding a dataset

Subclass `CorpusLoader` (and optionally `CorpusCleaner`), register both with
`@register_loader` / `@register_cleaner` under the same `name`, import the module in
`loaders/__init__.py` / `preprocessing/__init__.py`, and set `data.dataset: <name>` in a config.

### Using the package from Python

```python
from biber_drift.config import load_config
from biber_drift.pipeline import Pipeline

config = load_config("configs/ao3.yaml", ["mmd.sample_size=300"])
Pipeline(config).run(["analyze", "report"])
```

The building blocks are usable on their own, e.g.

```python
from biber_drift.analysis import RBFKernelMMD
mmd = RBFKernelMMD.from_median_heuristic(pooled, rng)
mmd.mmd2(x, y); mmd.baseline(x, sample_size=200, resamples=500, rng=rng)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests replace spaCy/pybiber with synthetic features, so they run in seconds and need no model
download. They run on every push via GitHub Actions (`.github/workflows/tests.yml`).

## Data use

Do not commit or redistribute the collected texts: AO3 works belong to their authors, and the
S2ORC release has its own licence terms. Respect AO3's Terms of Service and rate limits when
collecting data. `data/` and `outputs/` are git-ignored.

## References

* Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.
* Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel
  two-sample test. *Journal of Machine Learning Research*, 13, 723–773.
* Nieth et al. (2026) -- the MMD-on-Biber-features setup and dimension loadings followed here.
* Brown, D. W. *pybiber*: Biber feature extraction for Python.
