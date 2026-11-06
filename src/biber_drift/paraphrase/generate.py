"""Paraphrase the sample with one open-weight LLM (vLLM offline batch inference).

Runs on a GPU node of the HPC from a local Hugging Face snapshot, downloaded beforehand on the
login node with :func:`download_models` (compute nodes often have no internet). Every model gets
the same user message -- the instruction, a blank line, the text -- through its own chat template;
decoding is greedy by default. Texts go in a seeded random order, not by year, and outputs are
appended chunk by chunk to ``generations/<name>/<model>.jsonl``, so a stopped job resumes where it
ended; the settings, package versions and throughput of every session go to ``<model>.meta.json``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import ConfigError
from ..logging_utils import get_logger
from .config import GenerationConfig, ModelConfig, ParaphraseConfig, ParaphrasePaths

logger = get_logger(__name__)

STAGES = ("pilot", "main")

# Files a model repository carries besides the safetensors vLLM loads: the original checkpoints
# (Meta, Mistral and OpenAI formats), Apple Metal weights and PyTorch .bin copies.
DOWNLOAD_IGNORE = ["original/*", "metal/*", "consolidated*", "*.pth", "*.bin", "*.gguf"]


def build_messages(prompt: str, text: str) -> list[dict[str, str]]:
    """The single user turn every model gets: the instruction, a blank line, the text."""
    return [{"role": "user", "content": f"{prompt}\n\n{text}"}]


_TEMPLATE_DATE = re.compile(r"(Current date: )\d{4}-\d{2}-\d{2}")


def pin_date(rendered: str, current_date: str) -> str:
    """A rendered prompt with the day its chat template wrote in (gpt-oss) set to ``current_date``.

    Only the first match is replaced: the system block comes before the text, which may itself
    contain such a line.
    """
    pinned, count = _TEMPLATE_DATE.subn(lambda match: match.group(1) + current_date, rendered, count=1)
    if not count:
        raise ConfigError("current_date is set, but the chat template wrote no 'Current date: YYYY-MM-DD'")
    return pinned


def channel_text(raw: str, channel: str = "final") -> str | None:
    """The last ``channel`` message of harmony-format output (gpt-oss), or ``None`` when the model
    never reached it (e.g. cut off while reasoning)."""
    marker = f"<|channel|>{channel}<|message|>"
    start = raw.rfind(marker)
    if start < 0:
        return None
    answer = raw[start + len(marker):]
    for end in ("<|return|>", "<|end|>", "<|call|>"):
        answer = answer.split(end, 1)[0]
    return answer


def snapshot_path(model: ModelConfig) -> Path:
    """The model's snapshot in the local Hugging Face cache (``HF_HOME``); never downloads.

    The lookup skips the same files as the download: huggingface_hub 2.x checks that a cached
    snapshot is complete and would otherwise count the skipped ``original/`` checkpoints as missing.
    """
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model.repo, revision=model.revision, local_files_only=True,
                                      ignore_patterns=DOWNLOAD_IGNORE))
    except Exception as exc:  # LocalEntryNotFoundError, IncompleteSnapshotError; import paths differ by version
        cache = os.environ.get("HF_HOME", "~/.cache/huggingface")
        raise FileNotFoundError(f"{model.repo} is missing or incomplete in the Hugging Face cache ({cache}): "
                                f"{exc} -- run `biber-drift paraphrase download` on a node with internet") from exc


def download_models(config: ParaphraseConfig, keys: Iterable[str] | None = None) -> list[str]:
    """Download the models' weights into the Hugging Face cache (``HF_HOME``); returns the keys
    that failed.

    Gated repositories (Llama, Gemma) need their licence accepted on huggingface.co and an access
    token from ``hf auth login`` or the ``HF_TOKEN`` environment variable -- never in code or configs.
    """
    from huggingface_hub import snapshot_download

    failed = []
    for key in [config.model_key(k) for k in keys] if keys else list(config.models):
        model = config.models[key]
        logger.info("Downloading %s (%s, revision %s)", key, model.repo, model.revision or "main")
        try:
            path = Path(snapshot_download(model.repo, revision=model.revision, ignore_patterns=DOWNLOAD_IGNORE))
        except Exception as exc:  # gated, misspelt or no internet: report it and go on with the others
            logger.error("%s: %s", key, exc)
            failed.append(key)
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        logger.info("%s: commit %s, %.1f GB in %s", key, path.name, size, path)
    return failed


def stage_items(paths: ParaphrasePaths, stage: str, limit: int | None = None,
                columns: Iterable[str] = ("item_id", "pilot", "text")) -> pd.DataFrame:
    """The texts of a stage, in sample order: every text (``main``) or the pilot subset.

    ``limit`` keeps a quick test to ``limit`` texts per corpus, taken in turn from each of its
    groups x years, so that it sees both corpora and every year. Generation and the check use the
    same selection.
    """
    if not paths.sample.exists():
        raise FileNotFoundError(f"{paths.sample} not found -- draw it on the laptop with "
                                "`biber-drift paraphrase sample` and copy it here")
    columns = list(dict.fromkeys(["item_id", "pilot", *columns]))
    cells = ["corpus", "group", "year"]
    sample = pd.read_parquet(paths.sample, columns=list(dict.fromkeys([*columns, *cells])))
    if stage == "pilot":
        sample = sample[sample["pilot"]]
    if limit:
        turn = sample.groupby(cells, sort=False).cumcount()
        sample = (sample.assign(_turn=turn).sort_values(["corpus", "_turn"], kind="stable")
                  .groupby("corpus", sort=False).head(limit))
    return sample[columns]


@dataclass
class Completion:
    """One model answer: its text, length in tokens and why it stopped (``length`` = cut off)."""

    text: str
    tokens: int
    finish_reason: str | None


class VLLMEngine:
    """One model in vLLM, loaded from its local snapshot.

    Prompts are rendered with the model's own Hugging Face chat template and passed as token ids,
    so no second BOS token is added. Harmony-format models keep their special tokens in the
    output, so that the answer channel can be cut out of it.
    """

    def __init__(self, model: ModelConfig, generation: GenerationConfig) -> None:
        from vllm import LLM, SamplingParams

        path = snapshot_path(model)
        self.revision = path.name
        self.chat_template_kwargs = dict(model.chat_template_kwargs)
        self.current_date = model.current_date
        self.llm = LLM(model=str(path), seed=generation.seed, max_model_len=generation.max_model_len,
                       gpu_memory_utilization=generation.gpu_memory_utilization,
                       tensor_parallel_size=model.tensor_parallel_size, **model.engine)
        self.tokenizer = self.llm.get_tokenizer()
        stop_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in model.stop_tokens]
        if None in stop_ids:
            raise ConfigError(f"{model.repo}: unknown stop token(s) in {model.stop_tokens}")
        self.params = SamplingParams(
            temperature=generation.temperature, max_tokens=model.max_new_tokens or generation.max_new_tokens,
            seed=generation.seed, stop_token_ids=stop_ids or None,
            skip_special_tokens=model.answer_channel is None,
        )

    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.chat_template_kwargs)
        if self.current_date:
            text = pin_date(text, self.current_date)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts: list[list[int]]) -> list[Completion]:
        from vllm.inputs import TokensPrompt

        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], self.params)
        return [Completion(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason)
                for o in outputs]

    def environment(self) -> dict[str, Any]:
        import torch
        import transformers
        import vllm

        return {"vllm": vllm.__version__, "transformers": transformers.__version__,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


class ParaphraseGenerator:
    """Paraphrases the sample (or its pilot subset) with one model, resumably.

    Each output line holds ``item_id``, ``output`` (the answer; ``None`` when a harmony model gave
    none), ``finish_reason`` (``length`` = cut off at the token limit), ``prompt_tokens``,
    ``output_tokens`` and, for harmony models, ``raw_output`` with the reasoning.
    """

    def __init__(self, config: ParaphraseConfig, model_key: str, stage: str = "main",
                 name: str | None = None, tensor_parallel: int | None = None) -> None:
        if stage not in STAGES:
            raise ConfigError(f"Unknown stage {stage!r}; choose from {list(STAGES)}")
        self.config = config
        self.key = config.model_key(model_key)
        self.model = config.models[self.key]
        if tensor_parallel is not None:
            # Hardware only (e.g. 4 small GPUs instead of one A100), so it is no part of settings();
            # each session records it next to its GPUs.
            if tensor_parallel < 1:
                raise ConfigError("--tensor-parallel must be >= 1")
            self.model = replace(self.model, tensor_parallel_size=tensor_parallel)
        self.stage = stage
        self.paths = ParaphrasePaths(config.run_dir)
        self.directory = self.paths.generations(name or stage)

    @property
    def output_path(self) -> Path:
        return self.directory / f"{self.key}.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.directory / f"{self.key}.meta.json"

    def settings(self) -> dict[str, Any]:
        """Everything that changes the outputs: a file is only resumed with the same settings."""
        generation, model = self.config.generation, self.model
        return {"repo": model.repo, "revision": model.revision, "prompt": generation.prompt,
                "temperature": generation.temperature,
                "max_new_tokens": model.max_new_tokens or generation.max_new_tokens,
                "seed": generation.seed, "chat_template_kwargs": model.chat_template_kwargs,
                "current_date": model.current_date, "answer_channel": model.answer_channel, "stop_tokens": model.stop_tokens}

    def items(self, limit: int | None = None) -> pd.DataFrame:
        return stage_items(self.paths, self.stage, limit)

    def done(self) -> set[str]:
        """Item ids already in the output file (a line cut off by a killed job is removed first)."""
        if not self.output_path.exists():
            return set()
        data = self.output_path.read_bytes()
        if data and not data.endswith(b"\n"):
            logger.warning("Removing a partly written last line from %s", self.output_path)
            data = data[:data.rfind(b"\n") + 1]
            self.output_path.write_bytes(data)
        # "\n" only: splitlines() would also break at a U+2028 or U+0085 inside an output.
        return {json.loads(line)["item_id"] for line in data.decode("utf-8").split("\n") if line.strip()}

    def run(self, engine=None, limit: int | None = None) -> Path:
        """Paraphrase every text not yet in the output file.

        Args:
            engine: anything with ``revision``, ``encode(messages)``, ``generate(prompts)`` and
                ``environment()`` (default: a :class:`VLLMEngine` for this model).
            limit: only the first ``limit`` texts (a quick test of the setup).
        """
        meta = self._meta()
        items = self.items(limit)
        todo = items[~items["item_id"].isin(self.done())]
        # The sample is sorted by corpus and year, so in its order every chunk would hold two or
        # three years, and whatever differs between sessions or batches would line up with year.
        todo = todo.sample(frac=1, random_state=self.config.generation.seed)
        finished = len(items) - len(todo)
        logger.info("%s, %s: %d of %d texts done, %d to go -> %s", self.key, self.stage, finished,
                    len(items), len(todo), self.output_path)
        if todo.empty:
            return self.output_path

        engine = engine if engine is not None else VLLMEngine(self.model, self.config.generation)
        used = meta.setdefault("revision_used", engine.revision)
        if used != engine.revision:
            raise ConfigError(f"{self.output_path} was made with commit {used} of {self.model.repo}, but the "
                              f"cache now holds {engine.revision}; write to another folder with --name")
        session = {"started": _now(), "slurm_job": _slurm_job(), **engine.environment(),
                   "tensor_parallel_size": self.model.tensor_parallel_size,
                   "texts": 0, "output_tokens": 0, "seconds": 0.0}
        meta["sessions"].append(session)
        self._write_meta(meta)

        prompt, size = self.config.generation.prompt, self.config.generation.chunk_size
        for start in range(0, len(todo), size):
            chunk = todo.iloc[start:start + size]
            began = time.perf_counter()
            prompts = [engine.encode(build_messages(prompt, text)) for text in chunk["text"]]
            completions = engine.generate(prompts)
            self._append([self._row(item_id, ids, completion)
                          for item_id, ids, completion in zip(chunk["item_id"], prompts, completions)])
            seconds = time.perf_counter() - began
            tokens = sum(completion.tokens for completion in completions)
            session.update(texts=session["texts"] + len(chunk), output_tokens=session["output_tokens"] + tokens,
                           seconds=round(session["seconds"] + seconds, 1), finished=_now())
            self._write_meta(meta)
            finished += len(chunk)
            logger.info("%s: %d/%d texts, %.1f texts/s, %.0f output tokens/s", self.key, finished,
                        len(items), len(chunk) / seconds, tokens / seconds)
        return self.output_path

    # ------------------------------------------------------------------------------------

    def _row(self, item_id: str, prompt_ids: list[int], completion: Completion) -> dict[str, Any]:
        row = {"item_id": item_id, "output": completion.text, "finish_reason": completion.finish_reason,
               "prompt_tokens": len(prompt_ids), "output_tokens": completion.tokens}
        if self.model.answer_channel:
            row["output"] = channel_text(completion.text, self.model.answer_channel)
            row["raw_output"] = completion.text
        return row

    def _meta(self) -> dict[str, Any]:
        """This output file's record; refuses to resume a file made with other settings."""
        settings = self.settings()
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            changed = sorted(key for key in settings if meta["settings"].get(key) != settings[key])
            if changed:
                raise ConfigError(f"{self.output_path} was generated with other settings ({', '.join(changed)}); "
                                  "write to another folder with --name, or delete the file to start over")
            return meta
        if self.output_path.exists():
            raise ConfigError(f"{self.output_path} has no {self.meta_path.name}, so its settings are unknown; "
                              "move it away or delete it")
        return {"model": self.key, "family": self.model.family, "stage": self.stage,
                "settings": settings, "sessions": []}

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = self.meta_path.with_suffix(".json.tmp")
        partial.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(partial, self.meta_path)

    def _append(self, rows: list[dict[str, Any]]) -> None:
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            handle.flush()
            os.fsync(handle.fileno())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slurm_job() -> str | None:
    """``<array job>_<task>`` (or the plain job id) when running under Slurm."""
    array_job, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and task:
        return f"{array_job}_{task}"
    return os.environ.get("SLURM_JOB_ID")
