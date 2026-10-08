"""Biber's (1988) dimensions of register variation.

Loadings are transcribed from Table 9 of Biber (1988) as reproduced in Nieth et al. (2026),
with feature names mapped to the pybiber ``f_##_name`` convention; they are identical to the
paper's ``biber_loadings_1988.csv`` (shipped in ``features/data/``, checked by the tests). A
dimension score is the loading-weighted sum of the features' z-scores.

Two ways to compute the z-scores:

* ``corpus`` -- against the mean / sd of the documents being compared. Scores are relative to
  the corpus (centred on 0) and each feature is weighted by its spread in this corpus.
* ``biber1988`` -- against the means / sds of Biber's original 1988 corpus, as the paper does
  (``biber_normalisations_1988.csv``). Scores sit on Biber's scale and can be compared with the
  paper and with Biber's register positions. Caveat: Biber's type/token norm counts types in
  the first 400 words (mean 51.1) while pybiber's ``f_43`` is a ratio (~0.4), so ``f_43`` then
  adds a near-constant to Dimension 1 and contributes almost no variation -- in the paper too.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

BIBER_1988_LOADINGS: dict[str, dict[str, float]] = {
    "Dim1_Involved_vs_Informational": {
        "f_56_verb_private": 0.96, "f_60_that_deletion": 0.91, "f_59_contractions": 0.90,
        "f_03_present_tense": 0.86, "f_07_second_person_pronouns": 0.86, "f_12_proverb_do": 0.82,
        "f_67_neg_analytic": 0.78, "f_10_demonstrative_pronoun": 0.76, "f_49_emphatics": 0.74,
        "f_06_first_person_pronouns": 0.74, "f_09_pronoun_it": 0.71, "f_19_be_main_verb": 0.71,
        "f_35_because": 0.66, "f_50_discourse_particles": 0.66, "f_11_indefinite_pronouns": 0.62,
        "f_47_hedges": 0.58, "f_48_amplifiers": 0.56, "f_34_sentence_relatives": 0.55,
        "f_13_wh_question": 0.52, "f_52_modal_possibility": 0.50, "f_65_clausal_coordination": 0.48,
        "f_23_wh_clause": 0.47, "f_61_stranded_preposition": 0.43,
        # negative loadings
        "f_16_other_nouns": -0.80, "f_44_mean_word_length": -0.58, "f_39_prepositions": -0.54,
        "f_43_type_token": -0.54, "f_40_adj_attr": -0.47,
    },
    "Dim2_Narrative_vs_Nonnarrative": {
        "f_01_past_tense": 0.90, "f_08_third_person_pronouns": 0.73, "f_02_perfect_aspect": 0.48,
        "f_55_verb_public": 0.43, "f_66_neg_synthetic": 0.40, "f_25_present_participle": 0.39,
    },
    "Dim3_Explicit_vs_SituationDependent": {
        "f_32_wh_obj": 0.63, "f_33_pied_piping": 0.61, "f_31_wh_subj": 0.45,
        "f_64_phrasal_coordination": 0.36, "f_14_nominalizations": 0.36,
        "f_05_time_adverbials": -0.60, "f_04_place_adverbials": -0.49, "f_42_adverbs": -0.46,
    },
    "Dim4_Overt_Persuasion": {
        "f_24_infinitives": 0.76, "f_54_modal_predictive": 0.54, "f_57_verb_suasive": 0.49,
        "f_37_if": 0.47, "f_53_modal_necessity": 0.46, "f_63_split_auxiliary": 0.44,
    },
    "Dim5_Abstract_vs_Nonabstract": {
        "f_45_conjuncts": 0.48, "f_17_agentless_passives": 0.43, "f_26_past_participle": 0.42,
        "f_18_by_passives": 0.41, "f_27_past_participle_whiz": 0.40, "f_38_other_adv_sub": 0.39,
    },
    "Dim6_Online_Informational_Elaboration": {
        "f_21_that_verb_comp": 0.56, "f_51_demonstratives": 0.55, "f_30_that_obj": 0.46,
        "f_22_that_adj_comp": 0.36,
    },
    "Dim7_Seem_Appear": {
        "f_58_verb_seem": 0.35,
    },
}

# (positive-pole label, negative-pole label) for plot annotations; dimensions 4, 6 and 7 only
# have one named pole.
DIMENSION_LABELS: dict[str, tuple[str, str | None]] = {
    "Dim1_Involved_vs_Informational": ("Involved Production", "Informational Production"),
    "Dim2_Narrative_vs_Nonnarrative": ("Narrative Discourse", "Non-Narrative Discourse"),
    "Dim3_Explicit_vs_SituationDependent": ("Explicit Reference", "Situation-Dependent Reference"),
    "Dim4_Overt_Persuasion": ("Overt Expression of Persuasion", None),
    "Dim5_Abstract_vs_Nonabstract": ("Abstract Information", "Non-Abstract Information"),
    "Dim6_Online_Informational_Elaboration": ("On-line Informational Elaboration", None),
    "Dim7_Seem_Appear": ("SEEM / APPEAR", None),
}


SEEM_APPEAR = "Dim7_Seem_Appear"
STANDARDIZATIONS = ("corpus", "biber1988")


def _data_file(name: str):
    return resources.files("biber_drift.features").joinpath("data", name)


def load_biber1988_norms() -> pd.DataFrame:
    """Means and standard deviations of Biber's (1988) corpus, indexed by feature."""
    with _data_file("biber_normalisations_1988.csv").open() as handle:
        return pd.read_csv(handle).set_index("Feature")[["Mean", "Std"]]


def load_paper_loadings() -> pd.DataFrame:
    """The paper's loading table (6 dimensions x 67 features, zeros where a feature does not load)."""
    with _data_file("biber_loadings_1988.csv").open() as handle:
        return pd.read_csv(handle).set_index("dimension")


class BiberDimensionScorer:
    """Scores documents on Biber's dimensions from their raw feature values.

    Args:
        standardization: ``corpus`` (z-scores against the documents scored) or ``biber1988``
            (z-scores against Biber's 1988 norms, as in Nieth et al.).
        include_seem_appear: keep Dimension 7 (``f_58_verb_seem`` only). The paper uses
            Dimensions 1-6.
    """

    def __init__(self, standardization: str = "corpus", include_seem_appear: bool = True,
                 loadings: dict[str, dict[str, float]] | None = None) -> None:
        if standardization not in STANDARDIZATIONS:
            raise ValueError(f"standardization must be one of {STANDARDIZATIONS}")
        self.standardization = standardization
        loadings = dict(loadings or BIBER_1988_LOADINGS)
        if not include_seem_appear:
            loadings.pop(SEEM_APPEAR, None)
        self.loadings = loadings
        self.norms = load_biber1988_norms() if standardization == "biber1988" else None

    @property
    def dimensions(self) -> list[str]:
        return list(self.loadings)

    @property
    def required_features(self) -> list[str]:
        return sorted({f for weights in self.loadings.values() for f in weights})

    def score(self, features: pd.DataFrame, reference: pd.Series | None = None) -> pd.DataFrame:
        """Dimension scores, one row per document.

        Args:
            features: raw feature values (rows = documents).
            reference: ``corpus`` standardization only -- boolean mask of the rows whose mean/sd
                standardise the features (default: all rows, the pooled documents compared).
        """
        missing = set(self.required_features) - set(features.columns)
        if missing:
            raise ValueError(f"Features needed for the dimension scores are missing: {sorted(missing)}")
        values = features[self.required_features]
        if self.norms is not None:
            norms = self.norms.loc[self.required_features]
            z = (values - norms["Mean"]) / norms["Std"].replace(0, 1.0)
        else:
            ref = values if reference is None else values.loc[reference]
            z = (values - ref.mean()) / ref.std().replace(0, 1.0)

        scores = {}
        for dimension, weights in self.loadings.items():
            names = list(weights)
            scores[dimension] = z[names].to_numpy() @ np.array([weights[n] for n in names])
        return pd.DataFrame(scores, index=features.index)
