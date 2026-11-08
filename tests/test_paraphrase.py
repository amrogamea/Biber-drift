"""The paraphrase test: config, sampling and the resumable generation loop (no LLM or GPU needed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biber_drift.config import ConfigError
from biber_drift.paraphrase import (
    DEFAULT_PROMPT, Completion, ParaphraseGenerator, ParaphrasePaths, allocate, ao3_canon, build_messages,
    channel_text, draw_sample, even_split, load_paraphrase_config, pin_date, spread_pick,
)

from .conftest import fake_features, prose

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

LOTR = "The Lord of the Rings - J. R. R. Tolkien"
HOBBIT = "The Hobbit (Jackson Movies)"


def write_run(root: Path, documents: pd.DataFrame, rng: np.random.Generator) -> None:
    """A prepared pipeline run: corpus (with analysis windows) and features."""
    documents = documents.copy()
    documents.index = pd.Index([f"d{i}" for i in range(len(documents))], name="doc_id")
    documents["analysis_text"] = [prose(rng, 400) for _ in range(len(documents))]
    documents["analysis_word_count"] = documents["analysis_text"].str.split().str.len()
    documents["text"] = documents["analysis_text"]
    (root / "corpus").mkdir(parents=True)
    (root / "features").mkdir(parents=True)
    documents.to_parquet(root / "corpus" / "corpus.parquet")
    features = fake_features(list(documents.index), documents["year"].to_numpy())
    features.iloc[0, 0] = np.nan  # incomplete features: never sampled
    features.to_parquet(root / "features" / "biber_features.parquet")


@pytest.fixture
def runs(tmp_path, rng):
    """An ACL-like and an AO3-like prepared run over 2019-2021."""
    rows = []
    for year in (2019, 2020, 2021):
        size = 15 if year == 2019 else 100
        rows += [{"year": year, "venue_tier": ["A*", "Conference", "Workshop", "A*", "Conference"][i % 5]}
                 for i in range(size)]
    write_run(tmp_path / "acl", pd.DataFrame(rows), rng)

    rows, work = [], 0
    for year in (2019, 2020, 2021):
        def add(fandoms, rating, works, chapters):
            nonlocal work
            for _ in range(works):
                work += 1
                rows.extend({"year": year, "fandoms": fandoms, "rating": rating, "work_id": work}
                            for _ in range(chapters))
        add(LOTR, "General Audiences", 10, 4)                        # 40 chapters, 4 per work
        add(f"{LOTR}; TOLKIEN J. R. R. - Works & Related Fandoms", "Teen And Up Audiences", 30, 1)
        add(HOBBIT, "Teen And Up Audiences", 20, 1)
        add(f"{LOTR}; Harry Potter - J. K. Rowling", "General Audiences", 10, 1)          # crossover
        add("The Lord of the Rings: The Rings of Power (TV 2022)", "General Audiences", 5, 1)  # spin-off
        add(LOTR, "Mature", 5, 1)
    write_run(tmp_path / "ao3", pd.DataFrame(rows), rng)
    return tmp_path


def small_config(output_dir, seed=0):
    return load_paraphrase_config(CONFIGS / "paraphrase.yaml", [
        f"output_dir={output_dir}", f"sample.seed={seed}",
        "sample.groups.acl.years=[2019, 2020, 2021]", "sample.groups.acl.per_year=20",
        "sample.groups.lotr.years=[2019, 2020, 2021]", "sample.groups.lotr.per_year=30",
        "sample.groups.lotr.max_per_spread=2",
        "sample.groups.hobbit.years=[2019, 2020, 2021]", "sample.groups.hobbit.per_year=10",
        "sample.pilot.per_year=4", "sample.pilot.years={acl: [2020], ao3: [2020]}",
    ])


def test_shipped_config_loads():
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml")
    assert list(config.models)[:6] == ["llama-3.1-8b", "qwen3-8b", "mistral-nemo-12b", "gemma-3-12b",
                                       "gpt-oss-20b", "olmo-2-13b"]
    assert config.model_key("4") == "gpt-oss-20b"
    assert config.models["qwen3-8b"].chat_template_kwargs == {"enable_thinking": False}
    assert config.generation.prompt == DEFAULT_PROMPT                 # the YAML and the default agree
    assert config.generation.temperature == 0.0
    assert config.models["gpt-oss-20b"].current_date == "2026-11-05"
    unquoted = load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["models.gpt-oss-20b.current_date=2026-11-06"])
    assert unquoted.models["gpt-oss-20b"].current_date == "2026-11-06"           # a YAML date becomes text
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["models.gpt-oss-20b.current_date=today"])
    with pytest.raises(ConfigError):
        config.model_key("9")
    with pytest.raises(ConfigError, match="max_tokens"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["generation.max_tokens=10"])
    with pytest.raises(ConfigError, match="spread_over"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["sample.groups.acl.max_per_spread=2"])


def test_ao3_canon():
    assert ao3_canon(f"{LOTR}; The Lord of the Rings (Movies)") == "lotr"
    assert ao3_canon(f"{LOTR}; TOLKIEN J. R. R. - Works & Related Fandoms") == "lotr"
    assert ao3_canon(HOBBIT) == "hobbit"
    assert ao3_canon(f"{LOTR}; {HOBBIT}") is None                                 # both canons
    assert ao3_canon(f"{LOTR}; Harry Potter - J. K. Rowling") is None             # crossover
    assert ao3_canon("The Lord of the Rings: The Rings of Power (TV 2022)") is None
    assert ao3_canon("TOLKIEN J. R. R. - Works") is None                          # no canon named
    assert ao3_canon(None) is None


def test_allocate_and_splits():
    shares = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    assert allocate(10, shares, pd.Series({"a": 99, "b": 99, "c": 99})).to_dict() == {"a": 5, "b": 3, "c": 2}
    short = allocate(10, shares, pd.Series({"a": 99, "b": 1, "c": 99}))
    assert short["b"] == 1 and short.sum() == 10 and short["a"] > short["c"]
    assert even_split(50, 2) == [25, 25] and even_split(50, 3) == [17, 17, 16]


def test_spread_pick_takes_one_document_per_work_first(rng):
    frame = pd.DataFrame({"work_id": [1, 1, 1, 1, 2, 3, 4]}, index=list("abcdefg"))
    picked = spread_pick(frame, 4, rng, "work_id")
    assert sorted(frame.loc[picked, "work_id"]) == [1, 2, 3, 4]


def test_sample_is_stratified_capped_and_reproducible(runs):
    sample = draw_sample(small_config(runs))
    paths = ParaphrasePaths(runs / "paraphrase")
    assert paths.sample.exists() and paths.sample_counts.exists() and paths.config.exists()
    assert sample["item_id"].is_unique and "d0" not in set(sample["doc_id"])   # d0 has no features

    acl = sample[sample["group"] == "acl"]
    assert acl.groupby("year").size().to_dict() == {2019: 14, 2020: 20, 2021: 20}  # 2019: all eligible
    assert acl[acl["year"] == 2020]["stratum"].value_counts().to_dict() == {"A*": 8, "Conference": 8, "Workshop": 4}

    ao3 = pd.read_parquet(runs / "ao3" / "corpus" / "corpus.parquet")
    lotr = sample[sample["group"] == "lotr"].join(ao3[["fandoms", "work_id"]], on="doc_id")
    assert lotr.groupby("year").size().eq(30).all()
    assert set(lotr["stratum"]) == {"General Audiences", "Teen And Up Audiences"}
    assert not lotr["fandoms"].str.contains("Harry Potter|Rings of Power").any()
    assert lotr.groupby(["year", "work_id"]).size().max() <= 2                   # max_per_spread
    assert (sample[sample["group"] == "hobbit"].groupby("year").size() == 10).all()

    assert sample.loc[sample["pilot"]].groupby(["corpus", "group", "year"]).size().to_dict() == {
        ("acl", "acl", 2020): 4, ("ao3", "hobbit", 2020): 2, ("ao3", "lotr", 2020): 2}
    windows = ao3["analysis_text"].reindex(lotr["doc_id"]).to_numpy()
    assert (lotr["text"].to_numpy() == windows).all()

    again = draw_sample(small_config(runs))
    assert again["item_id"].tolist() == sample["item_id"].tolist()
    other = draw_sample(small_config(runs, seed=1))
    assert other["item_id"].tolist() != sample["item_id"].tolist()


class FakeEngine:
    """Answers every prompt without a model; records the batch sizes it was given."""

    revision = "0123abcd"

    def __init__(self, answer: str = "A paraphrase.") -> None:
        self.answer = answer
        self.batches = []

    def encode(self, messages):
        return list(range(len(messages[0]["content"].split())))

    def generate(self, prompts):
        self.batches.append(len(prompts))
        return [Completion(self.answer, 3, "stop") for _ in prompts]

    def environment(self):
        return {"vllm": "fake", "gpus": []}


@pytest.fixture
def sample_dir(tmp_path):
    paths = ParaphrasePaths(tmp_path / "paraphrase")
    paths.root.mkdir(parents=True)
    pd.DataFrame({"item_id": [f"acl/p{i}" for i in range(10)], "corpus": "acl", "group": "acl", "year": 2019,
                  "pilot": [i < 7 for i in range(10)],
                  "text": [f"Text number {i} ." for i in range(10)]}).to_parquet(paths.sample, index=False)
    return tmp_path


def test_prompt_and_harmony_answer():
    messages = build_messages("Paraphrase this.", "Some text.")
    assert messages == [{"role": "user", "content": "Paraphrase this.\n\nSome text."}]
    raw = ("<|channel|>analysis<|message|>Keep it short.<|end|><|start|>assistant"
           "<|channel|>final<|message|>The answer.<|return|>")
    assert channel_text(raw) == "The answer."
    assert channel_text("<|channel|>analysis<|message|>still thinking") is None


def test_pin_date_replaces_only_the_templates_date():
    rendered = ("<|start|>system<|message|>Knowledge cutoff: 2024-06\nCurrent date: 2026-12-31\n\n<|end|>"
                "<|start|>user<|message|>Paraphrase.\n\nCurrent date: 2001-01-01<|end|>")
    pinned = pin_date(rendered, "2026-11-05")
    assert "Current date: 2026-11-05\n" in pinned and "2026-12-31" not in pinned
    assert pinned.endswith("Current date: 2001-01-01<|end|>")                    # the text is left alone
    with pytest.raises(ConfigError, match="Current date"):
        pin_date("<|start|>user<|message|>No date here.", "2026-11-05")


def test_generation_resumes_and_records_settings(sample_dir):
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={sample_dir}",
                                                                  "generation.chunk_size=3"])
    generator = ParaphraseGenerator(config, "1", stage="pilot")
    engine = FakeEngine()
    generator.run(engine, limit=4)
    assert engine.batches == [3, 1]

    with open(generator.output_path, "a", encoding="utf-8") as handle:
        handle.write('{"item_id": "acl/p4", "out')                     # a job killed mid-write
    generator.run(engine)
    rows = [json.loads(line) for line in generator.output_path.read_text().splitlines()]
    ids = [row["item_id"] for row in rows]
    assert sorted(ids) == [f"acl/p{i}" for i in range(7)]                        # the pilot texts only
    assert sorted(ids[:4]) == [f"acl/p{i}" for i in range(4)]                   # the limited first run
    assert ids[:4] != [f"acl/p{i}" for i in range(4)]                            # in shuffled order
    assert engine.batches == [3, 1, 3]
    words = len(config.generation.prompt.split()) + 4                 # the fake engine counts words
    assert rows[0]["output"] == "A paraphrase." and rows[0]["prompt_tokens"] == words

    meta = json.loads(generator.meta_path.read_text())
    assert meta["model"] == "qwen3-8b" and meta["revision_used"] == "0123abcd"
    assert [session["texts"] for session in meta["sessions"]] == [4, 3]

    changed = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={sample_dir}",
                                                                   "generation.temperature=0.7"])
    with pytest.raises(ConfigError, match="temperature"):
        ParaphraseGenerator(changed, "qwen3-8b", stage="pilot").run(FakeEngine())
    ParaphraseGenerator(changed, "qwen3-8b", stage="pilot", name="t07").run(FakeEngine(), limit=2)


def test_unicode_line_separators_in_outputs_do_not_split_lines(sample_dir):
    from biber_drift.paraphrase.check import read_outputs

    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={sample_dir}"])
    generator = ParaphraseGenerator(config, "gemma-3-12b", stage="pilot")
    answer = "One line still the same output\x85and more."                 # as Gemma wrote them
    generator.run(FakeEngine(answer), limit=3)
    generator.run(FakeEngine(answer))                                            # resumes: reads its lines
    outputs = read_outputs(generator.output_path)
    assert len(outputs) == 7 and outputs["item_id"].is_unique
    assert (outputs["output"] == answer).all()


def test_limit_takes_each_corpus_and_spreads_over_groups_and_years(tmp_path):
    from biber_drift.paraphrase import stage_items

    paths = ParaphrasePaths(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    rows = [("acl", "acl", year) for year in (2019, 2019, 2019, 2024, 2024)] + \
           [("ao3", group, year) for group, year in [("lotr", 2019), ("lotr", 2019), ("hobbit", 2019),
                                                      ("lotr", 2025), ("hobbit", 2025)]]
    pd.DataFrame({"item_id": [f"{c}/d{i}" for i, (c, _, _) in enumerate(rows)], "corpus": [r[0] for r in rows],
                  "group": [r[1] for r in rows], "year": [r[2] for r in rows], "pilot": True,
                  "text": "x"}).to_parquet(paths.sample, index=False)
    picked = stage_items(paths, "pilot", limit=3)
    assert picked["item_id"].tolist() == ["acl/d0", "acl/d3", "acl/d1",                # 2019, 2024, 2019
                                          "ao3/d5", "ao3/d7", "ao3/d8"]                # one per group x year
    assert list(picked.columns) == ["item_id", "pilot", "text"]
    assert len(stage_items(paths, "pilot")) == 10


def test_tensor_parallel_override_is_recorded_but_not_a_setting(sample_dir):
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={sample_dir}"])
    ParaphraseGenerator(config, "olmo-2-13b", stage="pilot").run(FakeEngine(), limit=2)
    split = ParaphraseGenerator(config, "olmo-2-13b", stage="pilot", tensor_parallel=4)
    assert split.model.tensor_parallel_size == 4
    assert config.models["olmo-2-13b"].tensor_parallel_size == 1              # the config is untouched
    split.run(FakeEngine(), limit=3)                                          # resumes the same file
    meta = json.loads(split.meta_path.read_text())
    assert [s["tensor_parallel_size"] for s in meta["sessions"]] == [1, 4]
    assert [s["texts"] for s in meta["sessions"]] == [2, 1]
    with pytest.raises(ConfigError, match="tensor-parallel"):
        ParaphraseGenerator(config, "olmo-2-13b", tensor_parallel=0)


def test_snapshot_lookup_skips_the_files_the_download_skips(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    from biber_drift.paraphrase.generate import DOWNLOAD_IGNORE, snapshot_path

    calls = []

    def fake_snapshot_download(repo, **kwargs):
        calls.append(kwargs)
        return "/cache/models--x/snapshots/abc123"

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml")
    assert snapshot_path(config.models["qwen3-8b"]).name == "abc123"
    assert calls == [{"revision": None, "local_files_only": True, "ignore_patterns": DOWNLOAD_IGNORE}]


def test_harmony_models_keep_only_the_final_channel(sample_dir):
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={sample_dir}"])
    raw = "<|channel|>analysis<|message|>Plan.<|end|><|start|>assistant<|channel|>final<|message|>Done."
    generator = ParaphraseGenerator(config, "gpt-oss-20b", stage="main")
    generator.run(FakeEngine(raw), limit=2)
    row = json.loads(generator.output_path.read_text().splitlines()[0])
    assert row["output"] == "Done." and row["raw_output"] == raw
