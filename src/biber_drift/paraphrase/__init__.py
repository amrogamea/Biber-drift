"""The paraphrase test: do open LLMs change post-2023 texts less than older ones?

Steps (``biber-drift paraphrase <step>``, settings in ``configs/paraphrase.yaml``): ``sample`` draws
the texts from the prepared runs on the laptop, ``download`` fetches the model weights on the HPC
login node, ``generate`` paraphrases them with one model per GPU job (vLLM), ``check`` flags
refusals, chatty openers, loops and odd lengths in the outputs, ``tag`` cleans and filters them
(``filters``) and Biber-tags them and their originals (``tagging``), and ``analyze`` measures how much
each model changed each text and follows that over the years (``analysis``). The analysis lives in
:mod:`biber_drift.paraphrase.analysis` and is not imported here: it needs statsmodels, which the
HPC's vLLM environment lacks.
"""

from .check import (
    TEXT_CHECKS, ParaphraseChecker, flag_names, flag_outputs, kept_share, max_repeats, register_check,
)
from .config import (
    DEFAULT_PROMPT, MODEL_ROLES, AnalysisConfig, CheckConfig, CohortConfig, FilterConfig, GenerationConfig,
    ModelConfig, ParaphraseConfig, ParaphrasePaths, PilotConfig, SampleConfig, SampleGroupConfig, SeriesConfig,
    TaggingConfig, load_paraphrase_config,
)
from .generate import (
    Completion, ParaphraseGenerator, VLLMEngine, build_messages, channel_text, download_models, pin_date,
    stage_items,
)
from .clean import CLEANERS, clean_output, register_cleaner
from .sample import ParaphraseSampler, allocate, ao3_canon, draw_sample, even_split, spread_pick
from .tag import ORIGINAL, ParaphraseTagger

__all__ = [
    "AnalysisConfig", "CohortConfig", "SeriesConfig", "CLEANERS", "CheckConfig", "FilterConfig", "MODEL_ROLES",
    "ORIGINAL", "ParaphraseTagger", "TaggingConfig", "clean_output",
    "register_cleaner", "Completion", "DEFAULT_PROMPT", "GenerationConfig", "ModelConfig", "ParaphraseChecker",
    "ParaphraseConfig", "ParaphraseGenerator", "ParaphrasePaths", "ParaphraseSampler", "PilotConfig",
    "SampleConfig", "SampleGroupConfig", "TEXT_CHECKS", "VLLMEngine", "allocate", "ao3_canon",
    "build_messages", "channel_text", "download_models", "draw_sample", "even_split", "flag_names",
    "flag_outputs", "kept_share", "load_paraphrase_config", "max_repeats", "pin_date", "register_check",
    "spread_pick", "stage_items",
]
