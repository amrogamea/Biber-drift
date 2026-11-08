"""The comparison of originals and paraphrases (no spaCy model, LLM or GPU needed)."""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biber_drift.config import ConfigError
from biber_drift.features import BiberDimensionScorer
from biber_drift.paraphrase import ParaphrasePaths, ParaphraseTagger, load_paraphrase_config
from biber_drift.paraphrase.analysis import (
    MATTR, MEASURES, ParaphraseAnalysis, change_measures, corpus_scaler, edit_distance, stratified_bootstrap,
    stratified_means, word_edit_rate,
)

from .conftest import BIBER_FEATURES, prose

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MARK = "changed"           # the fake paraphrases swap words for this one; the fake tagger counts it


def brute_edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))
    return row[-1]


def test_edit_distance_matches_the_textbook_recursion(rng):
    assert edit_distance([], list("abc")) == 3 and edit_distance(list("kitten"), list("sitting")) == 3
    for _ in range(200):
        a, b = (list(rng.choice(list("abcde"), size=rng.integers(0, 12))) for _ in range(2))
        assert edit_distance(a, b) == brute_edit_distance(a, b)
    assert word_edit_rate("The elf, walked home.", "the ELF walked home!") == 0.0     # case and punctuation
    assert word_edit_rate("The elf walked home.", "The dwarf walked.") == pytest.approx(2 / 4)


def test_stratified_means_hold_each_groups_stratum_shares(rng):
    # three A texts (mean 1) and one B text (3), each stratum half of the group: 2, not 1.5
    frame = pd.DataFrame({"year": 2020, "stratum": ["A", "A", "A", "B"], "weight": 0.5, "x": [1.0, 1.0, 1.0, 3.0]})
    assert stratified_means(frame, ["year"], ["x"]).loc[2020].to_dict() == {"x": 2.0, "x_n": 4}
    only_a = frame.assign(x=[1.0, 1.0, 1.0, np.nan])                                # a missing stratum
    assert stratified_means(only_a, ["year"], ["x"]).loc[2020, "x"] == 1.0
    mean, draws = stratified_bootstrap(frame["x"].to_numpy(), frame["stratum"].to_numpy(),
                                       frame["weight"].to_numpy(), 200, rng)
    assert mean == 2.0 and np.allclose(draws, 2.0)                                  # no spread within strata


def test_scaler_and_change_measures():
    run = pd.DataFrame({"mean": 1.0, "std": [2.0] * len(BIBER_FEATURES)},
                       index=pd.Index(BIBER_FEATURES, name="feature"))
    originals = pd.DataFrame(1.0, index=["a", "b", "c"], columns=BIBER_FEATURES)
    originals[MATTR] = [0.6, 0.7, np.nan]
    scaler = corpus_scaler(run, originals)
    assert scaler.loc[MATTR, ["mean", "std"]].tolist() == pytest.approx([0.65, 0.05])  # population sd
    assert (scaler.drop(MATTR)[["mean", "std"]] == [1.0, 2.0]).all().all()

    paraphrases = originals.loc[["a", "b"]].copy()
    paraphrases["f_01_past_tense"] += 4.0                                # 2 sds on the run's scaler
    paraphrases.loc["b", MATTR] = np.nan                                 # under 100 tokens
    measures, differences = change_measures(originals, paraphrases, scaler, BiberDimensionScorer(include_seem_appear=False))
    assert measures.loc["a", "biber"] == pytest.approx(2.0) and np.isnan(measures.loc["b", "biber"])
    assert measures["biber_no_f43"].tolist() == pytest.approx([2.0, 2.0])
    assert measures.loc["a", "dimensions"] == pytest.approx(2.0 * 0.90)   # past tense loads 0.90 on Dimension 2
    assert differences.loc["a", "Dim2_Narrative_vs_Nonnarrative"] == pytest.approx(1.8)


class MarkTagger:
    """Stands in for BiberTagger: fixed features per item, shifted by every MARK word."""

    def __init__(self, cache: Path) -> None:
        pass

    def tag(self, texts: pd.Series) -> pd.DataFrame:
        rows = {}
        for item_id, text in texts.items():
            base = np.random.default_rng(zlib.crc32(item_id.encode())).gamma(2.0, 5.0, len(BIBER_FEATURES))
            marks = text.split().count(MARK)
            values = dict(zip(BIBER_FEATURES, base + 0.3 * marks))
            values[MATTR] = np.nan if len(text.split()) < 30 else 0.6 + base[42] / 500 - 0.002 * marks
            values["f_62_split_infinitive"] = 0.0
            rows[item_id] = values
        return pd.DataFrame.from_dict(rows, orient="index")

    def clear_cache(self) -> None:
        pass


@pytest.fixture
def analysed_run(tmp_path):
    """ACL 2013-2024 by year, LotR and Hobbit 2019-2024 by quarter; after the break the models change
    about half as many words. One ACL output is cut to 10 words (short, no f_43)."""
    rng = np.random.default_rng(3)
    root = tmp_path / "outputs"
    for run, features in (("acl", BIBER_FEATURES), ("ao3", [f for f in BIBER_FEATURES if "f_62" not in f])):
        results = root / run / "results"
        results.mkdir(parents=True)
        pd.DataFrame({"feature": features, "mean": 10.0, "std": 5.0}).to_csv(results / "reference_scaler.csv", index=False)
        (results / "reference.json").write_text(json.dumps({"sigma": 8.0}))

    rows, dates = [], {}
    for year in range(2013, 2025):
        for stratum in ("A*", "Workshop"):
            for i in range(4):
                rows.append(("acl", f"P{year}-{stratum[0]}{i}", "acl", year, stratum))
    for group in ("lotr", "hobbit"):
        for year in range(2019, 2025):
            for quarter in range(4):
                for i, stratum in enumerate(("General Audiences", "Teen And Up Audiences") * 2):
                    doc_id = f"{group}-{year}-{quarter}-{i}"
                    dates[doc_id] = f"{year}-{3 * quarter + 1:02d}-{10 + i:02d}"
                    rows.append(("ao3", doc_id, group, year, stratum))
    sample = pd.DataFrame(rows, columns=["corpus", "doc_id", "group", "year", "stratum"])
    sample.insert(0, "item_id", sample["corpus"] + "/" + sample["doc_id"])
    sample["pilot"] = False
    sample["text"] = [prose(rng, 60) for _ in range(len(sample))]
    paths = ParaphrasePaths(root / "paraphrase")
    paths.root.mkdir(parents=True)
    sample.to_parquet(paths.sample, index=False)
    (root / "ao3" / "corpus").mkdir(parents=True)
    pd.DataFrame({"chapter_date": dates}).rename_axis("doc_id").to_parquet(root / "ao3" / "corpus" / "corpus.parquet")

    folder = paths.generations("main")
    folder.mkdir(parents=True)
    for model, factor in (("llama-3.1-8b", 1.0), ("qwen3-8b", 0.3), ("gemma-3-12b", 1.2)):
        with open(folder / f"{model}.jsonl", "w", encoding="utf-8") as handle:
            for item_id, year, text in zip(sample["item_id"], sample["year"], sample["text"]):
                words = text.split()
                changes = int(factor * (rng.integers(16, 24) if year < 2023 else rng.integers(6, 12)))
                for position in np.linspace(0, len(words) - 1, changes).astype(int):
                    words[position] = MARK
                words = words[:len(words) - int(rng.integers(0, 6))]       # lengths vary a little
                output = " ".join(words[:10] if item_id == "acl/P2015-A0" else words)
                handle.write(json.dumps({"item_id": item_id, "output": output, "finish_reason": "stop",
                                         "prompt_tokens": 500, "output_tokens": 80}) + "\n")

    overrides = [f"output_dir={root}", "analysis.bootstrap_resamples=50",
                 "analysis.series.acl.min_documents=4", "analysis.series.lotr.min_documents=2",
                 "analysis.series.hobbit.min_documents=2", "analysis.mmd.sizes.acl=6", "analysis.mmd.sizes.lotr=12",
                 "analysis.mmd.sizes.hobbit=20", "analysis.mmd.subsample_draws=20", "analysis.mmd.permutations=200"]
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", overrides)
    ParaphraseTagger(config, "main").run(make_tagger=MarkTagger)
    return config, sample


def test_analysis_measures_change_and_finds_the_drop(analysed_run):
    config, sample = analysed_run
    analysis = ParaphraseAnalysis(config, "main")
    assert analysis.keys == ["llama-3.1-8b", "qwen3-8b", "gemma-3-12b"]           # config order
    directory = analysis.run()

    texts = pd.read_parquet(analysis.texts_path("llama-3.1-8b")).set_index("item_id")
    assert len(texts) == len(sample) and set(MEASURES) <= set(texts.columns)
    short = texts.loc["acl/P2015-A0"]
    assert short["short"] and np.isnan(short["biber"]) and np.isnan(short["dimensions"])
    assert not np.isnan(short["biber_no_f43"]) and short["edit"] > 0.8
    marks = pd.read_parquet(analysis.differences_path("llama-3.1-8b"))
    assert marks.loc["acl/P2014-A0", "f_01_past_tense"] == pytest.approx(0.3 * 18 / 5, abs=0.3 * 6 / 5)

    scaler = pd.read_csv(directory / "scaler_ao3.csv", index_col="feature")
    assert len(scaler) == 66 and scaler.loc["f_01_past_tense", "std"] == 5.0
    originals = pd.read_parquet(config.run_dir / "features" / "main" / "original.parquet")
    ao3 = originals.loc[sample.loc[sample["corpus"] == "ao3", "item_id"], MATTR]
    assert scaler.loc[MATTR, "mean"] == pytest.approx(ao3.mean())

    yearly = pd.read_csv(directory / "yearly.csv")
    edit = yearly[(yearly["variant"] == "all") & (yearly["model"] == "llama-3.1-8b") & (yearly["measure"] == "edit")]
    acl = edit[edit["group"] == "acl"].set_index("year")
    assert acl.loc[2024, "mean"] < acl.loc[2019, "mean"] and (acl["ci_lower"] <= acl["mean"]).all()
    assert set(yearly["variant"]) == {"all", "without_short_long_copy"}

    series = pd.read_csv(directory / "trend" / "series.csv")
    assert {"2022Q4", "2023Q1"} <= set(series.loc[series["group"] == "lotr", "period"])
    its = pd.read_csv(directory / "trend" / "its.csv")
    drop = its[(its["variant"] == "all") & (its["measure"] == "biber_no_f43") & (its["role"] == "main")]
    assert len(drop) == 6 and (drop["level_change"] < 0).all() and (drop["level_p_adjusted"] < 0.05).all()

    regressions = pd.read_csv(directory / "regressions.csv")
    pooled = regressions[(regressions["estimator"] == "mixed") & (regressions["variant"] == "all")]
    assert set(pooled["corpus"]) == {"acl", "ao3"} and set(pooled["length"]) == {True, False}
    edit = pooled[pooled["measure"] == "edit"]
    assert (edit["estimate"] < 0).all() and (edit["p_adjusted"] < 0.001).all() and edit["converged"].all()
    assert (pooled.loc[pooled["measure"] == "overlap", "estimate"] > 0).all()      # less change: more kept
    assert set(regressions.loc[regressions["estimator"] == "ols", "model"]) == set(analysis.keys)
    assert (pooled["texts"] < pooled["observations"]).all()                        # two models per text

    without = regressions[regressions["variant"] == "without_short_long_copy"]
    assert without.loc[without["corpus"] == "acl", "texts"].max() == 95             # the short output left out

    cohorts = pd.read_csv(directory / "mmd" / "cohorts.csv")
    samples = pd.read_csv(directory / "mmd" / "samples.csv")
    assert set(cohorts["group"]) == {"acl", "lotr"}                  # Hobbit: 16 texts a year, fewer than 20
    assert (cohorts["texts"] == cohorts["group"].map({"acl": 6, "lotr": 12})).all()
    assert len(cohorts) == 3 * (12 + 6) and samples.groupby(["group", "year"]).size().isin([6, 12]).all()
    assert "acl/P2015-A0" not in set(samples["item_id"])               # an output without f_43: not eligible
    llama = cohorts[cohorts["model"] == "llama-3.1-8b"]
    assert llama.loc[llama["year"] >= 2023, "mmd2"].max() < llama.loc[llama["year"] < 2023, "mmd2"].min()
    assert cohorts["p"].between(0, 1).all() and (cohorts["null_mean"] < cohorts["mmd2"]).all()
    mmd_its = pd.read_csv(directory / "mmd" / "its.csv")
    acl = mmd_its[(mmd_its["group"] == "acl") & (mmd_its["measure"] == "mmd2") & (mmd_its["role"] == "main")]
    assert len(acl) == 2 and (acl["level_change"] < 0).all() and "level_p_adjusted" in mmd_its

    features = pd.read_csv(directory / "features_yearly.csv")
    assert {"f_01_past_tense", "Dim1_Involved_vs_Informational"} <= set(features["feature"])
    info = json.loads((directory / "analysis.json").read_text())
    assert info["series"]["lotr"]["hac_lags"] == 4 and info["texts"]["all"]["qwen3-8b"] == len(sample)

    for path in directory.rglob("*"):                                                # ids and numbers only
        if path.suffix == ".csv":
            content = path.read_text()
        elif path.suffix == ".parquet":
            content = pd.read_parquet(path).to_csv()
        else:
            continue
        assert MARK not in content and sample["text"].iloc[0][:40] not in content, path.name


def test_analysis_resumes_and_refuses_stale_features(analysed_run):
    config, _ = analysed_run
    analysis = ParaphraseAnalysis(config, "main", keys=["0", "gemma-3-12b"])
    analysis.run()
    written = analysis.texts_path("gemma-3-12b").stat().st_mtime_ns
    ParaphraseAnalysis(config, "main", keys=["gemma-3-12b", "llama-3.1-8b"]).run()
    assert analysis.texts_path("gemma-3-12b").stat().st_mtime_ns == written          # done: not measured again

    features = pd.read_parquet(analysis.tagger.features_path("qwen3-8b"))
    features.iloc[1:].to_parquet(analysis.tagger.features_path("qwen3-8b"))
    with pytest.raises(RuntimeError, match="tag again"):
        ParaphraseAnalysis(config, "main", keys=["qwen3-8b"]).run()
    with pytest.raises(FileNotFoundError):
        ParaphraseAnalysis(config, "main", keys=["olmo-2-13b"])


def test_analysis_config():
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml")
    assert config.series("lotr").time_unit == "quarter" and config.series("lotr").min_segment == 4
    assert config.series("acl").hac_lags == 0 and config.analysis.exclude_flags == ["short", "long", "copy"]
    with pytest.raises(ConfigError, match="date_column"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["analysis.series.acl.time_unit=quarter"])
    with pytest.raises(ConfigError, match="sample group"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["analysis.series.silmarillion={time_unit: year}"])
    with pytest.raises(ConfigError, match="Unknown key"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["analysis.mmd_size=250"])
    assert config.analysis.mmd.sizes == {"acl": 250, "lotr": 275, "hobbit": 275}
    with pytest.raises(ConfigError, match="sample group"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["analysis.mmd.sizes.silmarillion=50"])
    with pytest.raises(ConfigError, match="p_adjust"):
        ParaphraseAnalysis(load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["analysis.p_adjust=bonferroni"]))
