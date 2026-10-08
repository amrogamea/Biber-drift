"""End-to-end runs on synthetic data, with the spaCy/pybiber tagger replaced by fake features."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from biber_drift.config import load_config  # noqa: E402
from biber_drift.features import BiberTagger  # noqa: E402
from biber_drift.pipeline import Pipeline  # noqa: E402

from .conftest import fake_features  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SMALL_RUN = [
    "mmd.sample_size=10", "mmd.bootstrap_resamples=5", "mmd.permutations=5",
    "mmd.min_cohort_documents=20", "mmd.subsample_draws=5", "stability.sample_sizes=[5, 10]",
    "stability.resamples=3", "quality.min_vocab_docs=2", "variance.resamples=5",
]


@pytest.fixture
def fake_tagger(monkeypatch):
    """Replace Biber tagging with year-dependent synthetic features."""

    def tag(self, texts):
        corpus = pd.read_parquet(self.cache_dir.parent.parent / "corpus" / "corpus.parquet")
        return fake_features(list(texts.index), corpus.loc[texts.index, "year"].to_numpy())

    monkeypatch.setattr(BiberTagger, "tag", tag)


def test_ao3_pipeline_end_to_end(ao3_dir, tmp_path, fake_tagger):
    config = load_config(CONFIGS / "ao3.yaml", SMALL_RUN + [
        f"output_dir={tmp_path}", f"data.path={ao3_dir}", "data.years=[2015, 2019, 2024]",
        "analysis.eras=[2015, 2019, 2024]", "feature_change.years=null", "dimensions.years=null",
    ])
    Pipeline(config).run()

    root = tmp_path / "ao3"
    report = pd.read_csv(root / "corpus" / "cleaning_report.csv").set_index("step")
    assert report.loc["duplicates", "documents_dropped"] == 10
    assert report.loc["podfics", "documents_dropped"] == 3
    corpus = pd.read_parquet(root / "corpus" / "corpus.parquet")
    assert len(corpus) == 117
    assert not corpus["analysis_text"].str.contains("Chapter Text|Author's notes").any()
    assert (corpus["analysis_word_count"] >= 400).all()

    for name in ["mmd_by_year_pairwise.csv", "mmd_by_year_matrix.csv", "mmd_by_year_samples.csv",
                 "mmd_by_year_pooled_baseline.csv", "sample_size_stability.csv", "variance.csv",
                 "biber_dimensions.parquet", "feature_change/all/combined_scores.csv"]:
        assert (root / "results" / name).exists(), name
    samples = pd.read_csv(root / "results" / "mmd_by_year_samples.csv")
    assert samples.groupby("cohort").size().eq(10).all()                  # fixed samples of n
    dims = pd.read_parquet(root / "results" / "biber_dimensions.parquet")
    assert [c for c in dims if c.startswith("Dim")][-1] == "Dim6_Online_Informational_Elaboration"
    figures = {p.name for p in (root / "figures").glob("*.png")}
    assert {"mmd_by_year_heatmap.png", "biber_dimensions_by_year.png", "quality_metrics.png",
            "sample_size_stability_all.png", "feature_shifts_2015_2024.png", "variance_by_year.png"} <= figures


def test_acl_pipeline_with_venue_tiers(acl_dir, tmp_path, fake_tagger):
    config = load_config(CONFIGS / "acl.yaml", SMALL_RUN + [
        f"output_dir={tmp_path}", f"data.path={acl_dir}", "data.loader={}",
        "cleaning.options={min_words: 100}", "mmd.min_cohort_documents=8", "mmd.sample_size=4",
        "mmd.era_group_sample_size=4", "stability.sample_sizes=[2, 4]",
    ])
    Pipeline(config).run()

    corpus = pd.read_parquet(tmp_path / "acl" / "corpus" / "corpus.parquet")
    assert (corpus["analysis_word_count"].between(400, 440)).all()       # abstract + intro window
    assert not corpus["analysis_text"].str.contains("Devlin et al").any()
    results = tmp_path / "acl" / "results"
    variance = pd.read_csv(results / "variance.csv")
    assert set(variance["grouping"]) == {"year", "era_group"}
    assert (results / "mmd_by_era_group_matrix.csv").exists()
    assert (results / "feature_change" / "group=A*" / "combined_scores.csv").exists()
    stability = pd.read_csv(results / "sample_size_stability.csv")
    assert set(stability["grouping"]) == {"group", "era", "era_group"}

    # Option A: one ruler -- every analysis uses the run's single scaler and sigma.
    sigma = json.loads((results / "reference.json").read_text())["sigma"]
    documents = len(corpus)
    for table in ["mmd_by_year_pairwise", "mmd_by_era_group_pairwise"]:
        assert pd.read_csv(results / f"{table}.csv")["sigma"].eq(sigma).all(), table
    assert stability["sigma"].eq(sigma).all()
    for prefix in ["mmd_by_year", "mmd_by_era_group"]:
        pooled = pd.read_csv(results / f"{prefix}_pooled_baseline.csv")
        assert pooled["documents"].iloc[0] == documents          # baseline from the whole corpus
    figures = {p.name for p in (tmp_path / "acl" / "figures").glob("*.png")}
    assert {"biber_dimensions_by_era.png", "biber_dimensions_by_era_group.png",
            "mmd_by_era_group_heatmap.png"} <= figures


def test_stages_can_run_separately(acl_dir, tmp_path, fake_tagger):
    config = load_config(CONFIGS / "acl.yaml", SMALL_RUN + [
        f"output_dir={tmp_path}", f"data.path={acl_dir}", "data.loader={}",
        "cleaning.options={min_words: 100}", "mmd.min_cohort_documents=8", "mmd.sample_size=4",
        "stability.enabled=false", "mmd.era_group_cohorts=false",
    ])
    Pipeline(config).run(["prepare", "features"])
    Pipeline(config).run(["analyze"])
    assert (tmp_path / "acl" / "results" / "mmd_by_year_matrix.csv").exists()
    assert not (tmp_path / "acl" / "figures").exists()


def test_analyze_refuses_features_from_another_spacy_model(acl_dir, tmp_path, fake_tagger):
    from biber_drift.config import ConfigError

    overrides = SMALL_RUN + [f"output_dir={tmp_path}", f"data.path={acl_dir}", "data.loader={}",
                             "cleaning.options={min_words: 100}", "mmd.min_cohort_documents=8",
                             "mmd.sample_size=4", "mmd.era_group_cohorts=false", "stability.enabled=false"]
    Pipeline(load_config(CONFIGS / "acl.yaml", overrides + ["features.spacy_model=en_core_web_sm"])
             ).run(["prepare", "features"])
    with pytest.raises(ConfigError, match="en_core_web_sm"):
        Pipeline(load_config(CONFIGS / "acl.yaml", overrides)).run(["analyze"])
