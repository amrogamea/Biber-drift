"""Shared fixtures: synthetic corpora in the raw AO3 / ACL formats (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The 67 pybiber feature columns.
BIBER_FEATURES = [
    "f_01_past_tense", "f_02_perfect_aspect", "f_03_present_tense", "f_04_place_adverbials",
    "f_05_time_adverbials", "f_06_first_person_pronouns", "f_07_second_person_pronouns",
    "f_08_third_person_pronouns", "f_09_pronoun_it", "f_10_demonstrative_pronoun",
    "f_11_indefinite_pronouns", "f_12_proverb_do", "f_13_wh_question", "f_14_nominalizations",
    "f_15_gerunds", "f_16_other_nouns", "f_17_agentless_passives", "f_18_by_passives",
    "f_19_be_main_verb", "f_20_existential_there", "f_21_that_verb_comp", "f_22_that_adj_comp",
    "f_23_wh_clause", "f_24_infinitives", "f_25_present_participle", "f_26_past_participle",
    "f_27_past_participle_whiz", "f_28_present_participle_whiz", "f_29_that_subj", "f_30_that_obj",
    "f_31_wh_subj", "f_32_wh_obj", "f_33_pied_piping", "f_34_sentence_relatives", "f_35_because",
    "f_36_though", "f_37_if", "f_38_other_adv_sub", "f_39_prepositions", "f_40_adj_attr",
    "f_41_adj_pred", "f_42_adverbs", "f_43_type_token", "f_44_mean_word_length", "f_45_conjuncts",
    "f_46_downtoners", "f_47_hedges", "f_48_amplifiers", "f_49_emphatics", "f_50_discourse_particles",
    "f_51_demonstratives", "f_52_modal_possibility", "f_53_modal_necessity", "f_54_modal_predictive",
    "f_55_verb_public", "f_56_verb_private", "f_57_verb_suasive", "f_58_verb_seem", "f_59_contractions",
    "f_60_that_deletion", "f_61_stranded_preposition", "f_62_split_infinitive", "f_63_split_auxiliary",
    "f_64_phrasal_coordination", "f_65_clausal_coordination", "f_66_neg_synthetic", "f_67_neg_analytic",
]

_WORDS = ("the elf walked through quiet woods while the dwarf counted old coins and a hobbit "
          "wondered whether the long road would ever end before winter came again").split()


def prose(rng: np.random.Generator, n_words: int) -> str:
    """Sentences of 8-15 random words, about ``n_words`` words in total."""
    sentences, total = [], 0
    while total < n_words:
        length = int(rng.integers(8, 16))
        words = list(rng.choice(_WORDS, size=length))
        sentences.append(" ".join(words).capitalize() + ".")
        total += length
    return " ".join(sentences)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def ao3_dir(tmp_path, rng):
    """Two fandom files over three years; some chapters appear in both files."""
    directory = tmp_path / "ao3"
    directory.mkdir()
    rows = []
    chapter_id = 1000
    for year in (2015, 2019, 2024):
        for i in range(40):
            chapter_id += 1
            text = "Chapter Text\nAuthor's notes: thanks for reading!\n" + prose(rng, 480)
            rows.append({
                "work_id": chapter_id // 2, "chapter_id": chapter_id, "chapter_number": 1,
                "chapter_date": f"{year}-03-{(i % 27) + 1:02d}", "language": "English",
                "additional_tags": "Angst" if i else "Podfic; Audio Format: MP3",
                "freeform_tags": "Angst" if i else "Podfic", "word_count": 500,
                "story_text": text,
            })
    frame = pd.DataFrame(rows)
    frame.iloc[:90].to_csv(directory / "ao3_fandom_a.csv", index=False)
    frame.iloc[80:].to_csv(directory / "ao3_fandom_b.csv", index=False)  # rows 80-89 in both
    return directory


@pytest.fixture
def acl_dir(tmp_path, rng):
    directory = tmp_path / "acl"
    directory.mkdir()
    venues = ["Annual Meeting of the Association for Computational Linguistics",
              "BEA@NAACL-HLT", "International Conference on Computational Linguistics"]
    for year in (2000, 2019, 2024):
        rows = []
        for i in range(30):
            rows.append({
                "id": f"P{year % 100:02d}-{i}", "title": "t", "authors": "a", "year": year,
                "venue": venues[i % 3], "citation_count": 1,
                "abstract": prose(rng, 120), "introduction": prose(rng, 330) + " (Devlin et al., 2019).",
                "methods": prose(rng, 120), "results": "", "discussion": prose(rng, 60),
            })
        pd.DataFrame(rows).to_csv(directory / f"Dataset_{year}.csv", index=False)
    return directory


def fake_features(doc_ids, years, seed: int = 0) -> pd.DataFrame:
    """Plausible positive feature values whose distribution drifts with the year."""
    rng = np.random.default_rng(seed)
    years = np.asarray(years, dtype=float)
    values = rng.gamma(2.0, 5.0, size=(len(doc_ids), len(BIBER_FEATURES)))
    values += ((years - years.min()) / 5.0)[:, None] * np.linspace(0, 2, len(BIBER_FEATURES))
    frame = pd.DataFrame(values, index=pd.Index(doc_ids, name="doc_id"), columns=BIBER_FEATURES)
    frame["f_62_split_infinitive"] = 0.0
    return frame
