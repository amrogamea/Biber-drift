"""Cleaning, filtering and tagging of the paraphrases (no spaCy model, LLM or GPU needed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biber_drift.config import ConfigError
from biber_drift.features import moving_type_token_ratio
from biber_drift.paraphrase import (
    ORIGINAL, ParaphrasePaths, ParaphraseTagger, clean_output, load_paraphrase_config,
)

from .conftest import prose

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PLAIN = "The elf walked through the quiet woods. The dwarf counted his coins."
ALL = ["opener", "closer", "markdown", "quotes"]


def cleaned(output: str, original: str = PLAIN) -> str:
    return clean_output(output, original, ALL)[0]


def test_cleaners_strip_boilerplate_but_keep_the_paraphrase():
    assert cleaned("Sure! Here's the paraphrased text:\n\nThe elf walked on.") == "The elf walked on."
    assert cleaned("**Paraphrased Text:**\n\nThe elf walked on.") == "The elf walked on."
    assert cleaned("**Paraphrase:** The elf walked on.") == "The elf walked on."
    story = "Sure, here is the map: the road runs east."
    assert cleaned(story, original="Sure, here is a map: it runs east.") == story   # the original has it too

    assert cleaned("The elf walked on.\n\nI hope this helps!") == "The elf walked on."
    assert cleaned("The elf walked on.\n(Note: I kept the tone.)") == "The elf walked on."
    letter = "I hope this letter finds you well, she wrote."
    assert cleaned(letter) == letter                                                 # a single line stays
    assert cleaned("The elf walked on.\n\nI hope so.", original="Walk.\n\nI hope so.") == \
        "The elf walked on.\n\nI hope so."

    assert cleaned("# Title\n\n**Bold** words and\n- one item\n1. two items\n\n---\nEnd.") == \
        "Title\n\nBold words and\none item\ntwo items\n\nEnd."
    assert cleaned("A **bold** claim.", original="An **old** claim.") == "A **bold** claim."

    assert cleaned('"The elf walked through the woods."') == "The elf walked through the woods."
    assert cleaned('"Go," he said. "Now."', original='"Leave," he said. Then he left.') == \
        '"Go," he said. "Now."'                                                   # opens like the original

    text, applied = clean_output("Here is the paraphrased text:\n\n**The** elf.", PLAIN, ["quotes", "opener", "markdown"])
    assert (text, applied) == ("The elf.", ["opener", "markdown"])                # in registry order
    with pytest.raises(ConfigError, match="cleaner"):
        clean_output("x", "y", ["emoji"])


def test_moving_type_token_ratio():
    assert moving_type_token_ratio(list("abac"), 2) == 1.0
    assert moving_type_token_ratio(list("abac"), 3) == pytest.approx((2 / 3 + 3 / 3) / 2)
    assert np.isnan(moving_type_token_ratio(list("ab"), 3))                          # no full window
    rng = np.random.default_rng(0)
    types = list(rng.choice(list("abcdefghij"), size=60))
    brute = np.mean([len(set(types[i:i + 10])) / 10 for i in range(len(types) - 9)])
    assert moving_type_token_ratio(types, 10) == pytest.approx(brute)


def test_mattr_is_computed_per_document():
    pl = pytest.importorskip("polars")
    from biber_drift.features import mattr_by_document

    tokens = pl.DataFrame({
        "doc_id": ["long"] * 6 + ["short"] * 2 + ["none"],
        "token": ["The", "elf", ",", "the", "elf", "42", "An", "orc", "!"],
        "tag": ["DT", "NN", ",", "DT", "NN", "CD", "DT", "NN", "."],
    })
    result = dict(zip(*mattr_by_document(tokens, 3).to_dict(as_series=False).values()))
    # "long": the_dt elf_nn the_dt elf_nn (punctuation and digits out) -> windows of 2 types each
    assert result["long"] == pytest.approx(2 / 3)
    assert np.isnan(result["short"]) and np.isnan(result["none"])        # under one window: NaN, not 0


class FakeTagger:
    """Stands in for BiberTagger: two features, and a record of what it was given."""

    seen: dict[str, pd.Series] = {}

    def __init__(self, cache: Path) -> None:
        self.key = cache.name

    def tag(self, texts: pd.Series) -> pd.DataFrame:
        FakeTagger.seen[self.key] = texts
        words = texts.str.split().str.len().astype(float)
        return pd.DataFrame({"f_01_past_tense": words, "f_43_type_token": 0.5}, index=texts.index)

    def clear_cache(self) -> None:
        pass


@pytest.fixture
def generation_run(tmp_path):
    rng = np.random.default_rng(1)
    paths = ParaphrasePaths(tmp_path / "paraphrase")
    paths.root.mkdir(parents=True)
    ids = [f"acl/p{i}" for i in range(7)]
    originals = [prose(rng, 60) for _ in ids]
    pd.DataFrame({"item_id": ids, "corpus": "acl", "group": "acl", "year": [2019, 2019, 2019, 2024, 2024, 2024, 2024],
                  "stratum": "A*", "pilot": True, "text": originals}).to_parquet(paths.sample, index=False)
    long = " ".join(f"Sentence number {i} goes on and on here." for i in range(70))   # 560 words
    rows = [
        ("acl/p0", "Here is the paraphrased text:\n\n" + originals[1], "stop"),   # opener stripped
        ("acl/p1", "I'm sorry, but I can't help with this request.", "stop"),     # refusal: dropped
        ("acl/p2", originals[0][:200], "length"),                                  # cut off: dropped
        ("acl/p3", None, "stop"),                                                  # no answer: dropped
        ("acl/p4", long, "stop"),                                                  # cut to <= 440 words
        ("acl/p5", "**Paraphrased Text:**", "stop"),                               # empty once cleaned
        ("acl/p6", '"' + originals[2] + '"', "stop"),                              # quotes stripped
    ]
    folder = paths.generations("main")
    folder.mkdir(parents=True)
    with open(folder / "gemma-3-12b.jsonl", "w", encoding="utf-8") as handle:
        for item_id, output, finish in rows:
            handle.write(json.dumps({"item_id": item_id, "output": output, "finish_reason": finish,
                                     "prompt_tokens": 500, "output_tokens": 300}) + "\n")
    return tmp_path, originals


def test_tagger_cleans_filters_cuts_and_tags_without_writing_text(generation_run):
    root, originals = generation_run
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={root}"])
    tagger = ParaphraseTagger(config, "main")
    assert tagger.keys == [ORIGINAL, "gemma-3-12b"]
    written = tagger.run(make_tagger=FakeTagger)
    assert [path.name for path in written] == ["original.parquet", "gemma-3-12b.parquet"]

    record = pd.read_parquet(tagger.items_path("gemma-3-12b")).set_index("item_id")
    assert record.loc[~record["kept"], "reason"].to_dict() == {"acl/p1": "refusal", "acl/p2": "cut_off",
                                                               "acl/p3": "missing", "acl/p5": "empty"}
    assert record.index[record["kept"]].tolist() == ["acl/p0", "acl/p4", "acl/p6"]
    assert record.loc["acl/p0", "cleaned"] == "opener" and record.loc["acl/p6", "cleaned"] == "quotes"
    assert record.loc["acl/p4", "words_output"] == 560 and 400 <= record.loc["acl/p4", "words"] <= 440
    assert "text" not in record.columns and "output" not in record.columns          # ids and numbers only

    texts = FakeTagger.seen["gemma-3-12b"]
    assert list(texts.index) == ["acl/p0", "acl/p4", "acl/p6"]
    assert texts["acl/p0"] == originals[1] and texts["acl/p6"] == originals[2]
    features = pd.read_parquet(tagger.features_path("gemma-3-12b"))
    assert list(features.index) == ["acl/p0", "acl/p4", "acl/p6"] and features.index.name == "item_id"
    assert len(FakeTagger.seen[ORIGINAL]) == 7 and FakeTagger.seen[ORIGINAL]["acl/p1"] == originals[1]

    info = json.loads((tagger.directory / "gemma-3-12b.json").read_text())
    assert info["texts"] == 3 and info["tagging"]["mattr_window"] == 100 and "versions" in info
    assert ParaphraseTagger(config, "main", keys=["3"]).run(make_tagger=FakeTagger) == []   # done: skipped
    quick = ParaphraseTagger(config, "main", keys=["3"], limit=2)
    assert quick.directory.name == "main_limit2" and len(quick.run(make_tagger=FakeTagger)) == 1


def test_tagger_rejects_unknown_keys_and_rules(generation_run):
    root, _ = generation_run
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={root}"])
    with pytest.raises(FileNotFoundError):
        ParaphraseTagger(config, "main", keys=["olmo-2-13b"])
    bad = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={root}", "filters.drop=[shouting]"])
    with pytest.raises(ConfigError, match="shouting"):
        ParaphraseTagger(bad, "main")


def test_shipped_rules_and_roles():
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml")
    assert {key: model.role for key, model in config.models.items() if model.role != "main"} == \
        {"qwen3-8b": "separate", "llama-2-13b": "check"}
    assert config.filters.drop == ["missing", "cut_off", "refusal", "think", "template", "loop"]
    assert (config.filters.window_cap, config.filters.min_words, config.tagging.mattr_window) == (440, 0, 100)
    with pytest.raises(ConfigError, match="role"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["models.qwen3-8b.role=ignored"])
    with pytest.raises(ConfigError, match="window"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["filters.window_words=500"])
