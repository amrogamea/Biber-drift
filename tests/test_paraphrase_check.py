"""The paraphrase check: flags against the original, summaries without texts (no LLM or GPU needed)."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biber_drift.config import ConfigError
from biber_drift.paraphrase import (
    CheckConfig, ParaphraseChecker, ParaphrasePaths, TEXT_CHECKS, flag_names, flag_outputs, kept_share,
    load_paraphrase_config, max_repeats,
)

from .conftest import prose

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PLAIN = "The elf walked through the quiet woods. The dwarf counted his coins."


def flagged(name: str, output: str, original: str = PLAIN, **settings) -> bool:
    return TEXT_CHECKS[name].test(output, original, CheckConfig(**settings))


def test_refusals_and_chatty_lines_count_only_what_the_original_lacks():
    assert flagged("refusal", "I'm sorry, but I can't help with this request.")
    assert flagged("refusal", "As an AI, I will not rewrite explicit content.")
    assert not flagged("refusal", '"I can\'t," she said. "I\'m sorry."')        # dialogue, not a refusal
    assert not flagged("refusal", "I'm sorry, but I can't stay.", original="I'm sorry, but I can't stay.")
    assert not flagged("refusal", PLAIN + " " + "x " * 200 + "I cannot help with that request.",
                       head_chars=100)                                          # too far in

    assert flagged("opener", "Here is the paraphrased text:\n\nThe elf walked on.")
    assert flagged("opener", "**Sure! Here's a rewritten version:**\n\nThe elf walked on.")
    assert flagged("opener", "Paraphrased text: The elf walked on.")
    assert not flagged("opener", "Sure enough, the elf walked on through the woods.")
    assert not flagged("opener", '"Sure," she said, "here is the map."')

    assert flagged("closer", "The elf walked on.\n\nNote: I kept the original tone.")
    assert flagged("closer", "The elf walked on.\n\nI hope this helps!")
    assert not flagged("closer", 'The elf walked on.\n\n"I hope so," he said.',
                       original='The elf walked.\n\n"I hope so," he said.')


def test_remnants_quotes_markdown_and_loops():
    assert flagged("think", "<think>\n\n</think>\n\nThe elf walked on.")
    assert flagged("template", "The elf walked on.<|eot_id|>")
    assert not flagged("template", PLAIN)

    assert flagged("quotes", '"The elf walked through the woods."')
    assert not flagged("quotes", '"Go," he said. "Now."', original='"Leave," he said. "Now."')

    assert flagged("markdown", "**Paraphrase**\n\n- The elf walked on.")
    assert flagged("markdown", "The elf walked on.\n\n---\n\nThe dwarf slept.")
    assert not flagged("markdown", "1. First point.", original="1. A numbered intro.")

    loop = "the road goes ever on and on down from the door " * 3
    assert max_repeats(loop, 8) == 3 and max_repeats("too short", 8) == 0
    assert flagged("loop", PLAIN + " " + loop)
    assert not flagged("loop", PLAIN + " " + loop, original=loop)                 # the original repeats too


@pytest.fixture
def pilot_run(tmp_path):
    """A sample with 6 pilot texts and one model's outputs: good, refusal, cut off, null, short, absent."""
    rng = np.random.default_rng(0)
    paths = ParaphrasePaths(tmp_path / "paraphrase")
    paths.root.mkdir(parents=True)
    texts = [prose(rng, 400) for _ in range(7)]
    pd.DataFrame({
        "item_id": [f"acl/p{i}" for i in range(4)] + ["ao3/c4", "ao3/c5", "ao3/c6"],
        "corpus": ["acl"] * 4 + ["ao3"] * 3, "doc_id": ["p0", "p1", "p2", "p3", "c4", "c5", "c6"],
        "group": ["acl"] * 4 + ["lotr", "hobbit", "lotr"], "year": [2019, 2019, 2024, 2024, 2019, 2025, 2025],
        "stratum": ["A*", "Workshop", "A*", "Conference", "General Audiences", "Teen And Up Audiences", "Mature"],
        "pilot": [True] * 6 + [False], "text": texts, "words": [len(t.split()) for t in texts],
    }).to_parquet(paths.sample, index=False)

    lines = [
        {"item_id": "acl/p0", "output": prose(rng, 400), "finish_reason": "stop"},
        {"item_id": "acl/p1", "output": "I'm sorry, but I can't help with that request.", "finish_reason": "stop"},
        {"item_id": "acl/p2", "output": prose(rng, 900), "finish_reason": "length"},
        {"item_id": "acl/p3", "output": None, "finish_reason": "length"},            # gpt-oss: no final channel
        {"item_id": "ao3/c4", "output": "Here is the paraphrased text:\n\n" + prose(rng, 100),
         "finish_reason": "stop"},
    ]                                                                                # ao3/c5: no line at all
    directory = paths.generations("pilot")
    directory.mkdir(parents=True)
    with open(directory / "qwen3-8b.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps({**line, "prompt_tokens": 600, "output_tokens": 500}) + "\n"
                          for line in lines)
        handle.write('{"item_id": "ao3/c5", "outp')                                 # a job still writing
    return tmp_path, texts


def checker_for(root, **kwargs):
    config = load_paraphrase_config(CONFIGS / "paraphrase.yaml", [f"output_dir={root}"])
    return ParaphraseChecker(config, stage="pilot", **kwargs)


def test_checker_flags_each_output_and_writes_no_text(pilot_run):
    root, texts = pilot_run
    checker = checker_for(root)
    flags = checker.run().set_index("item_id")

    assert list(flags.index) == ["acl/p0", "acl/p1", "acl/p2", "acl/p3", "ao3/c4", "ao3/c5"]  # the pilot only
    raised = {item: [name for name in flag_names() if flags.at[item, name]] for item in flags.index}
    assert raised == {"acl/p0": [], "acl/p1": ["refusal", "short"], "acl/p2": ["cut_off", "long"],
                      "acl/p3": ["missing", "cut_off"], "ao3/c4": ["opener", "short"], "ao3/c5": ["missing"]}
    assert flags.at["acl/p0", "family"] == "Qwen" and 0.9 < flags.at["acl/p0", "length_ratio"] < 1.1
    assert np.isnan(flags.at["ao3/c5", "length_ratio"])

    summary = pd.read_csv(checker.output_dir / "summary.csv")
    acl_2024 = summary[(summary["corpus"] == "acl") & (summary["year"] == 2024)].iloc[0]
    assert (acl_2024["texts"], acl_2024["outputs"], acl_2024["cut_off"], acl_2024["missing"]) == (2, 1, 2, 1)
    models = pd.read_csv(checker.output_dir / "models.csv")
    assert models.set_index("corpus")["texts"].to_dict() == {"acl": 4, "ao3": 2}
    assert models.set_index("corpus")["flagged"].to_dict() == {"acl": 3, "ao3": 2}

    written = b"".join(path.read_bytes() for path in checker.output_dir.iterdir())
    assert sorted(path.name for path in checker.output_dir.iterdir()) == ["flags.parquet", "models.csv",
                                                                         "summary.csv"]
    assert texts[0][:60].encode() not in written and b"sorry" not in written     # ids and numbers only


def test_show_prints_seeded_examples_to_the_terminal_only(pilot_run):
    root, texts = pilot_run
    checker = checker_for(root)
    with pytest.raises(RuntimeError):
        checker.show(2)
    checker.run()
    out = io.StringIO()
    checker.show(5, flag="opener", out=out)
    printed = out.getvalue()
    assert "1 of 1 outputs with flag opener" in printed and "ao3/c4" in printed and texts[4] in printed

    again = io.StringIO()
    checker.show(2, out=again)
    other = io.StringIO()
    checker.show(2, out=other)
    assert again.getvalue() == other.getvalue()                                    # seeded
    with pytest.raises(ConfigError, match="Unknown flag"):
        checker.show(2, flag="rude")
    assert sorted(path.name for path in checker.output_dir.iterdir()) == ["flags.parquet", "models.csv",
                                                                         "summary.csv"]


def test_overlap_tells_copies_from_paraphrases():
    rng = np.random.default_rng(1)
    original = prose(rng, 400)
    words = original.split()
    edited = " ".join(word if i % 20 else "changed" for i, word in enumerate(words))   # 1 word in 20
    assert kept_share(original, original, 8) == 1.0
    assert 0.5 < kept_share(original, edited, 8) < 1.0
    assert kept_share(original, prose(rng, 400), 8) < 0.05
    assert np.isnan(kept_share("too short", "too short", 8))

    items = pd.DataFrame({"item_id": ["a/1", "a/2", "a/3"], "corpus": "a", "group": "a", "year": 2020,
                          "stratum": "x", "text": original, "output": [original, prose(rng, 400), None],
                          "finish_reason": "stop", "prompt_tokens": 600, "output_tokens": 500})
    flags = flag_outputs(items, CheckConfig())
    assert flags["copy"].tolist() == [True, False, False]
    assert flags["overlap"].iloc[0] == 1.0 and np.isnan(flags["overlap"].iloc[2])
    assert not flags.loc[1, "flagged"] and flags.loc[0, "flagged"]


def test_checker_limit_models_and_config(pilot_run):
    root, _ = pilot_run
    limited = checker_for(root, limit=2, models=["1"]).run()                        # "1" = qwen3-8b
    assert limited["item_id"].tolist() == ["acl/p0", "acl/p2", "ao3/c4", "ao3/c5"]   # 2 per corpus, spread
    with pytest.raises(FileNotFoundError, match="gemma-3-12b"):
        checker_for(root, models=["gemma-3-12b"]).run()
    with pytest.raises(FileNotFoundError):
        checker_for(root / "nowhere").run()
    with pytest.raises(ConfigError, match="loop_repeats"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["check.loop_repeats=1"])
    with pytest.raises(ConfigError, match="copy_share"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["check.copy_share=1.5"])
    with pytest.raises(ConfigError, match="Unknown key"):
        load_paraphrase_config(CONFIGS / "paraphrase.yaml", ["check.threshold=1"])
