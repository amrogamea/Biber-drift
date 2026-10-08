import numpy as np
import pandas as pd
import pytest

from biber_drift.analysis import FeatureChangeAnalysis, minmax_scale, percent_change
from biber_drift.corpus import Corpus
from biber_drift.features import BIBER_1988_LOADINGS, BiberDimensionScorer, FeatureMatrix
from biber_drift.preprocessing import TextWindower, count_words

from .conftest import BIBER_FEATURES, fake_features


def test_window_ends_at_sentence_boundary_between_target_and_cap():
    text = " ".join(["word"] * 405) + ". " + " ".join(["more"] * 100) + "."
    window = TextWindower(400, 440).window(text)
    assert window.endswith("word.")
    assert count_words(window) == 405


def test_window_falls_back_to_hard_cap_and_keeps_short_texts():
    windower = TextWindower(400, 440)
    no_sentences = " ".join(["word"] * 600)
    assert count_words(windower.window(no_sentences)) == 440
    short = "A short text."
    assert windower.window(short) == short


def test_window_apply_drops_short_documents():
    frame = pd.DataFrame({"doc_id": ["a", "b"], "year": [2019, 2019],
                          "text": [" ".join(["w"] * 500) + ".", "short."]})
    corpus, step = TextWindower(400, 440).apply(Corpus.from_frame(frame), min_words=400)
    assert list(corpus.documents.index) == ["a"]
    assert step.documents_dropped == 1


def test_feature_matrix_selects_sorts_and_excludes():
    frame = fake_features(["a", "b"], [2000, 2001])
    frame["not_a_feature"] = 1
    frame.loc["b", "f_01_past_tense"] = np.inf
    matrix = FeatureMatrix.from_frame(frame, exclude=["f_62_split_infinitive"])
    assert matrix.feature_names == sorted(c for c in BIBER_FEATURES if c != "f_62_split_infinitive")
    assert list(matrix.values.index) == ["a"]


def test_dimension_scores_are_weighted_z_scores():
    frame = fake_features([str(i) for i in range(30)], list(range(30)))
    scores = BiberDimensionScorer().score(frame)
    weights = BIBER_1988_LOADINGS["Dim2_Narrative_vs_Nonnarrative"]
    z = (frame[list(weights)] - frame[list(weights)].mean()) / frame[list(weights)].std()
    expected = sum(z[name] * w for name, w in weights.items())
    np.testing.assert_allclose(scores["Dim2_Narrative_vs_Nonnarrative"], expected)
    assert scores.shape == (30, 7)


def test_loadings_equal_the_papers_table():
    from biber_drift.features import load_paper_loadings

    paper = load_paper_loadings()
    ours = list(BIBER_1988_LOADINGS.values())
    assert len(paper) == 6
    for i, (_, row) in enumerate(paper.iterrows()):
        assert row[row != 0].to_dict() == pytest.approx(ours[i])


def test_biber1988_standardization_uses_the_1988_norms():
    from biber_drift.features import load_biber1988_norms

    frame = fake_features([str(i) for i in range(20)], list(range(20)))
    scorer = BiberDimensionScorer("biber1988", include_seem_appear=False)
    scores = scorer.score(frame)
    assert list(scores.columns) == list(BIBER_1988_LOADINGS)[:6]

    norms = load_biber1988_norms()
    weights = BIBER_1988_LOADINGS["Dim4_Overt_Persuasion"]
    z = (frame[list(weights)] - norms.loc[list(weights), "Mean"]) / norms.loc[list(weights), "Std"]
    np.testing.assert_allclose(scores["Dim4_Overt_Persuasion"], sum(z[f] * w for f, w in weights.items()))
    # Independent of the other documents scored: the first row alone gives the same score.
    np.testing.assert_allclose(scorer.score(frame.iloc[:1]).to_numpy(), scores.iloc[:1].to_numpy())


def test_minmax_and_percent_change():
    assert list(minmax_scale(pd.Series([2.0, 4.0, 6.0]))) == [0.0, 0.5, 1.0]
    assert list(minmax_scale(pd.Series([3.0, 3.0]))) == [0.0, 0.0]
    pct = percent_change(pd.Series([0.0, -2.0]), pd.Series([1.0, 1.0]))
    assert np.isnan(pct.iloc[0]) and pct.iloc[1] == pytest.approx(50.0)


def test_feature_change_comparisons_and_scores():
    analysis = FeatureChangeAnalysis([2019, 2009, 2024])
    assert list(analysis.comparisons) == ["2009_to_2019", "2019_to_2024", "2009_to_2024"]

    years = pd.Series([2009] * 5 + [2019] * 5 + [2024] * 5, index=[str(i) for i in range(15)])
    features = pd.DataFrame({"f_a": [1.0] * 5 + [2.0] * 5 + [4.0] * 5,
                             "f_b": [10.0] * 5 + [10.5] * 5 + [11.0] * 5}, index=years.index)
    result = analysis.run(features, years)
    assert result.changes.loc["f_a", "change_2009_to_2024"] == pytest.approx(3.0)
    assert result.changes.loc["f_a", "pct_2009_to_2024"] == pytest.approx(300.0)
    assert result.top("2009_to_2024", k=1).index[0] == "f_a"
    assert result.combined_scores().loc["f_a"].max() == pytest.approx(1.0)


def test_feature_change_requires_key_years_present():
    years = pd.Series([2009, 2019], index=["a", "b"])
    with pytest.raises(ValueError):
        FeatureChangeAnalysis([2009, 2024]).run(pd.DataFrame({"f": [1.0, 2.0]}, index=years.index), years)
