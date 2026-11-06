"""Diagnose the model outputs of one generation run before the cleaning and drop rules are frozen.

``biber-drift paraphrase check --stage pilot`` reads every ``generations/<name>/<model>.jsonl``,
joins it with the sample and flags each output: missing, cut off at the token limit, a refusal, a
chatty opener or closer, ``<think>`` or chat-template remnants, wrapping quotes, markdown, a
repetition loop, a length far from the original's, or a near copy of the original (``overlap``: the
share of the original's word n-grams that the output keeps). A text flag counts only what the original
does not show itself, so dialogue that starts with "I'm sorry" or a chapter in quotes is not
flagged. The files written hold ids, flags and numbers only (``check/flags.parquet``,
``check/summary.csv`` per model x corpus x year, ``check/models.csv`` per model x corpus);
``--show N`` prints N seeded examples per model to the terminal, never to a file.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO

import numpy as np
import pandas as pd

from ..config import ConfigError
from ..logging_utils import get_logger
from ..preprocessing.windowing import count_words
from .config import CheckConfig, ParaphraseConfig, ParaphrasePaths
from .generate import stage_items

logger = get_logger(__name__)

SAMPLE_COLUMNS = ("item_id", "pilot", "corpus", "group", "year", "stratum", "text")
OUTPUT_COLUMNS = ("item_id", "output", "finish_reason", "prompt_tokens", "output_tokens")
ROW_FLAGS = ("missing", "cut_off")      # from the output line itself
LENGTH_FLAGS = ("short", "long")        # from the length ratio (provisional bands)
OVERLAP_FLAGS = ("copy",)               # from the n-gram overlap with the original (provisional)


@dataclass(frozen=True)
class TextCheck:
    """One flag computed from an output and its original."""

    name: str
    description: str
    test: Callable[[str, str, CheckConfig], bool]


TEXT_CHECKS: dict[str, TextCheck] = {}


def register_check(name: str, description: str) -> Callable:
    """Decorator: add ``test(output, original, config) -> bool`` to the text checks as ``name``."""
    def decorator(test: Callable[[str, str, CheckConfig], bool]) -> Callable:
        TEXT_CHECKS[name] = TextCheck(name, description, test)
        return test
    return decorator


def flag_names() -> list[str]:
    return [*ROW_FLAGS, *TEXT_CHECKS, *LENGTH_FLAGS, *OVERLAP_FLAGS]


def _pattern(source: str) -> re.Pattern:
    """Case-insensitive, with every apostrophe matching both ' and ’ (models write either)."""
    return re.compile(source.replace("'", "['’]"), re.IGNORECASE)


# Assistant-voice refusals. Plain "I can't" or "I'm sorry" are left alone: they are common in AO3
# dialogue. Matched in the first `head_chars` of the output only.
_REFUSAL = _pattern(
    r"\b(?:I'm|I am) (?:sorry|afraid)\b[^\n]{0,100}?\b(?:can't|cannot|unable|won't|not able)\b"
    r"|\bI apologi[sz]e\b[^\n]{0,100}?\b(?:can't|cannot|unable|won't|not able)\b"
    r"|\b(?:can't|cannot|unable to|won't|will not|not able to)\s+(?:help|assist)(?: you)? with (?:this|that|your|the)\b"
    r"|\b(?:can't|cannot|unable to|won't|will not|not able to)\s+(?:comply|fulfil\w*|paraphrase|rewrite)\b"
    r"|\bas an AI\b|\bAI (?:language )?model\b"
    r"|\b(?:violates?|against) (?:my|our|the) (?:usage |content )?(?:guidelines|polic(?:y|ies))\b")

# First line: "Sure! Here is the paraphrase:", "Here's a rewritten version", "Paraphrased text:".
# Leading markdown is skipped, quotes are not (a story may open with "Sure," she said).
_OPENER = _pattern(
    r"^[\s*#>_]*(?:"
    r"(?:sure|certainly|of course|absolutely|okay|ok)\b[^\n]{0,100}?(?:paraphras|rewrit|reword|here's|here is|:\s*$)"
    r"|(?:here's|here is|here are|below is|the following is)\b[^\n]{0,100}?(?:paraphras|rewrit|reword|version|:\s*$)"
    r"|(?:paraphrased?|rewritten|revised|reworded)(?: text| version| passage)?\s*:)")

# Last line: "Note: ...", "I hope this helps", "I've kept the original tone", "This paraphrase ...".
_CLOSER = _pattern(
    r"^[\s*#>(_\[-]*(?:"
    r"(?:note|notes|explanation|changes(?: made)?|key changes)\s*:"
    r"|(?:i hope|let me know|feel free|(?:i've|i have) (?:kept|maintained|preserved|paraphrased|rewritten|retained)"
    r"|this (?:paraphrase|paraphrased|version|rewrite)|in this (?:paraphrase|version)"
    r"|the paraphrased (?:text|version))\b)")

_THINK = re.compile(r"</?think>", re.IGNORECASE)
_TEMPLATE = re.compile(r"<\|[A-Za-z_]+\|>|\[/?INST\]|<(?:start|end)_of_turn>|</?s>")
_MARKDOWN = (
    re.compile(r"^\s*#{1,6}\s", re.MULTILINE),                    # heading
    re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__"),                   # bold
    re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE),     # list item
    re.compile(r"```"),                                           # code fence
    re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE),  # rule
)
_OPENING_QUOTES, _CLOSING_QUOTES = "\"'“‘«`", "\"'”’»`"
_WORD = re.compile(r"\b\w+\b")


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _first_line(text: str) -> str:
    lines = _lines(text)
    return lines[0] if lines else ""


def _last_line(text: str) -> str:
    lines = _lines(text)
    return lines[-1] if lines else ""


def _wrapped(text: str) -> bool:
    text = text.strip()
    return len(text) > 1 and text[0] in _OPENING_QUOTES and text[-1] in _CLOSING_QUOTES


def max_repeats(text: str, n: int) -> int:
    """How often the most frequent word n-gram occurs (0 for texts shorter than ``n`` words)."""
    words = _WORD.findall(text.lower())
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values(), default=0)


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def kept_share(original: str, output: str, n: int) -> float:
    """Share of the original's distinct word n-grams that reappear in the output: about 0.1 for a
    paraphrase, near 1 for a copy (NaN for an original shorter than ``n`` words)."""
    grams = _ngrams(original, n)
    return len(grams & _ngrams(output, n)) / len(grams) if grams else float("nan")


@register_check("refusal", "an assistant-voice refusal at the start of the output")
def _refusal(output: str, original: str, config: CheckConfig) -> bool:
    return bool(_REFUSAL.search(output[:config.head_chars])) and not _REFUSAL.search(original[:config.head_chars])


@register_check("opener", "a chatty first line (\"Here is the paraphrased text:\")")
def _opener(output: str, original: str, config: CheckConfig) -> bool:
    return bool(_OPENER.search(_first_line(output))) and not _OPENER.search(_first_line(original))


@register_check("closer", "a chatty last line (\"Note: ...\", \"I hope this helps\")")
def _closer(output: str, original: str, config: CheckConfig) -> bool:
    return bool(_CLOSER.search(_last_line(output))) and not _CLOSER.search(_last_line(original))


@register_check("think", "<think> tags (Qwen3 remnants)")
def _think(output: str, original: str, config: CheckConfig) -> bool:
    return bool(_THINK.search(output)) and not _THINK.search(original)


@register_check("template", "chat-template tokens such as <|eot_id|> or [INST]")
def _template(output: str, original: str, config: CheckConfig) -> bool:
    return bool(_TEMPLATE.search(output)) and not _TEMPLATE.search(original)


@register_check("quotes", "the whole output wrapped in quotes")
def _quotes(output: str, original: str, config: CheckConfig) -> bool:
    return _wrapped(output) and not _wrapped(original)


@register_check("markdown", "markdown (headings, bold, lists, fences, rules) the original lacks")
def _markdown(output: str, original: str, config: CheckConfig) -> bool:
    return any(pattern.search(output) and not pattern.search(original) for pattern in _MARKDOWN)


@register_check("loop", "a repetition loop: one word n-gram repeated (more often than in the original)")
def _loop(output: str, original: str, config: CheckConfig) -> bool:
    repeats = max_repeats(output, config.loop_ngram)
    return repeats >= config.loop_repeats and repeats > max_repeats(original, config.loop_ngram)


def read_outputs(path: Path) -> pd.DataFrame:
    """One model's output lines. A half-written last line (a job still running) is skipped."""
    # Split at "\n" only: str.splitlines() also breaks at U+2028 or U+0085, which json.dumps(...,
    # ensure_ascii=False) leaves unescaped inside an output.
    rows, lines = [], path.read_text(encoding="utf-8").split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number < len(lines):
                raise ValueError(f"{path}, line {number}: not valid JSON") from None
            logger.warning("%s: skipping a partly written last line", path.name)
    outputs = pd.DataFrame(rows).reindex(columns=list(OUTPUT_COLUMNS))
    duplicated = outputs["item_id"].duplicated()
    if duplicated.any():
        logger.warning("%s: %d repeated item ids; keeping the first of each", path.name, duplicated.sum())
    return outputs[~duplicated]


def flag_outputs(items: pd.DataFrame, config: CheckConfig) -> pd.DataFrame:
    """Flags and numbers per item, for items joined with their output line (no text columns)."""
    # object dtype first: a model without any output gives a float NaN column, and str(nan) is "nan"
    output = items["output"].astype(object).where(items["output"].notna(), None)
    missing = output.map(lambda text: text is None or not str(text).strip())
    words_original = items["text"].map(count_words)
    words_output = output.map(lambda text: count_words(text) if text else 0).where(~missing)
    ratio = words_output / words_original

    flags = pd.DataFrame({
        "item_id": items["item_id"], "corpus": items["corpus"], "group": items["group"],
        "year": items["year"], "stratum": items["stratum"], "finish_reason": items["finish_reason"],
        "prompt_tokens": items["prompt_tokens"], "output_tokens": items["output_tokens"],
        "words_original": words_original, "words_output": words_output, "length_ratio": ratio,
        "missing": missing, "cut_off": items["finish_reason"].eq("length"),
    })
    pairs = list(zip(output, items["text"]))
    for name, check in TEXT_CHECKS.items():
        flags[name] = [not gone and check.test(text, original, config)
                       for gone, (text, original) in zip(missing, pairs)]
    flags["short"] = ratio.lt(config.short_ratio)
    flags["long"] = ratio.gt(config.long_ratio)
    overlap = pd.Series([np.nan if gone else kept_share(original, text, config.copy_ngram)
                         for gone, (text, original) in zip(missing, pairs)], index=flags.index)
    flags.insert(flags.columns.get_loc("length_ratio") + 1, "overlap", overlap)
    flags["copy"] = overlap.ge(config.copy_share)
    flags["flagged"] = flags[flag_names()].any(axis=1)
    return flags.reset_index(drop=True)


def summarise(flags: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Counts of every flag, and the length ratio's spread, per group of ``by``."""
    grouped = flags.groupby(by, sort=True, dropna=False)   # a model missing from the config has no family
    summary = grouped[[*flag_names(), "flagged"]].sum().astype(int)
    summary.insert(0, "texts", grouped.size())
    summary.insert(1, "outputs", summary["texts"] - summary["missing"])
    ratio = grouped["length_ratio"]
    summary["ratio_mean"] = ratio.mean()
    for share in (0.05, 0.5, 0.95):
        summary[f"ratio_p{int(share * 100):02d}"] = ratio.quantile(share)
    overlap = grouped["overlap"]
    summary["overlap_mean"] = overlap.mean()
    summary["overlap_p50"] = overlap.median()
    summary["overlap_p95"] = overlap.quantile(0.95)
    summary["prompt_tokens_mean"] = grouped["prompt_tokens"].mean()
    summary["output_tokens_mean"] = grouped["output_tokens"].mean()
    return summary.round(3).reset_index()


class ParaphraseChecker:
    """Flags the outputs of one generation run (``generations/<name>/``), model by model.

    Args:
        stage: ``pilot`` or ``main``: which sample texts each model should have answered.
        name: the run's folder (default: the stage), e.g. ``smoke``.
        models: model keys or config indexes (default: every ``<model>.jsonl`` in the folder).
        limit: only the first ``limit`` texts, as given to ``generate --limit``.
    """

    def __init__(self, config: ParaphraseConfig, stage: str = "main", name: str | None = None,
                 models: Iterable[str] | None = None, limit: int | None = None) -> None:
        self.config = config
        self.stage = stage
        self.name = name or stage
        self.paths = ParaphrasePaths(config.run_dir)
        self.directory = self.paths.generations(self.name)
        self.models = [key if key in config.models or not key.isdigit() else config.model_key(key)
                       for key in models] if models else None
        self.limit = limit
        self._texts: pd.DataFrame | None = None
        self._flags: pd.DataFrame | None = None

    @property
    def output_dir(self) -> Path:
        return self.paths.check(self.name)

    def model_files(self) -> dict[str, Path]:
        """``<model>.jsonl`` files to check, in config order (models not in the config last)."""
        files = {path.stem: path for path in sorted(self.directory.glob("*.jsonl"))}
        if self.models:
            absent = [key for key in self.models if key not in files]
            if absent:
                raise FileNotFoundError(f"No output file for {absent} in {self.directory}")
            files = {key: files[key] for key in self.models}
        order = list(self.config.models)
        return dict(sorted(files.items(), key=lambda item: (order.index(item[0]) if item[0] in order
                                                            else len(order), item[0])))

    def run(self) -> pd.DataFrame:
        """Flag every output and write ``flags.parquet``, ``summary.csv`` and ``models.csv``."""
        files = self.model_files() if self.directory.exists() else {}
        if not files:
            raise FileNotFoundError(f"No <model>.jsonl files in {self.directory}")
        items = stage_items(self.paths, self.stage, self.limit, SAMPLE_COLUMNS)
        frames, texts = [], []
        for key, path in files.items():
            outputs = read_outputs(path)
            extra = int((~outputs["item_id"].isin(items["item_id"])).sum())
            if extra:
                logger.warning("%s: %d outputs are not among the %d %s texts checked (another stage or "
                               "--limit?)", key, extra, len(items), self.stage)
            joined = items.merge(outputs, on="item_id", how="left")
            flags = flag_outputs(joined, self.config.check)
            model = self.config.models.get(key)
            flags.insert(0, "model", key)
            flags.insert(1, "family", model.family if model else None)
            frames.append(flags)
            texts.append(joined[["item_id", "text", "output"]].assign(model=key))
            self._log(key, flags)

        self._flags = pd.concat(frames, ignore_index=True)
        self._texts = pd.concat(texts, ignore_index=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._flags.to_parquet(self.output_dir / "flags.parquet", index=False)
        summarise(self._flags, ["model", "family", "corpus", "year"]).to_csv(self.output_dir / "summary.csv",
                                                                            index=False)
        summarise(self._flags, ["model", "family", "corpus"]).to_csv(self.output_dir / "models.csv", index=False)
        logger.info("Wrote flags.parquet, summary.csv and models.csv to %s", self.output_dir)
        return self._flags

    def show(self, count: int, flag: str | None = None, out: TextIO = sys.stdout) -> None:
        """Print ``count`` seeded examples per model, original and output, to the terminal.

        ``flag`` picks outputs with that flag (``any`` = any flag); by default every output that
        exists. Texts are printed only, never written to a file.
        """
        if self._flags is None or self._texts is None:
            raise RuntimeError("Call run() before show()")
        if flag is not None and flag != "any" and flag not in flag_names():
            raise ConfigError(f"Unknown flag {flag!r}; choose from {['any', *flag_names()]}")
        rng = np.random.default_rng(self.config.check.examples_seed)
        names = flag_names()
        for key, flags in self._flags.groupby("model", sort=False):
            chosen = flags["flagged"] if flag == "any" else flags[flag] if flag else ~flags["missing"]
            rows = flags[chosen]
            print(f"\n##### {key}: {min(count, len(rows))} of {len(rows)} outputs"
                  + (f" with flag {flag}" if flag else ""), file=out)
            if rows.empty:
                continue
            picked = rows.iloc[np.sort(rng.choice(len(rows), size=min(count, len(rows)), replace=False))]
            texts = self._texts[self._texts["model"] == key].set_index("item_id")
            for _, row in picked.iterrows():
                raised = [name for name in names if row[name]]
                print("=" * 100, file=out)
                print(f"{key} | {row['item_id']} | {row['corpus']} {row['year']} {row['stratum']} | "
                      f"ratio {row['length_ratio']:.2f} | overlap {row['overlap']:.2f} | "
                      f"finish {row['finish_reason']} | "
                      f"flags: {', '.join(raised) or 'none'}", file=out)
                print(f"--- original ({row['words_original']} words)", file=out)
                print(texts.at[row["item_id"], "text"], file=out)
                words = "none" if row["missing"] else f"{int(row['words_output'])} words"
                print(f"--- output ({words})", file=out)
                print(texts.at[row["item_id"], "output"], file=out)

    @staticmethod
    def _log(key: str, flags: pd.DataFrame) -> None:
        counts = {name: int(flags[name].sum()) for name in flag_names() if flags[name].any()}
        ratio = flags["length_ratio"]
        logger.info("%s: %d texts, %d outputs; %d flagged%s; length ratio median %.2f (5-95%%: %.2f-%.2f); "
                    "overlap mean %.2f", key, len(flags), int((~flags["missing"]).sum()),
                    int(flags["flagged"].sum()), "".join(f", {name} {n}" for name, n in counts.items()),
                    ratio.median(), ratio.quantile(0.05), ratio.quantile(0.95), flags["overlap"].mean())
