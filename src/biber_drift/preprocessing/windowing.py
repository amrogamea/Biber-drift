"""Fixed-length analysis windows.

Biber features are rates per 1,000 tokens, but their variance still depends on text length.
Cutting every document to the same window (by default the first 400 words, extended to the
next sentence end but never past 440 words) keeps documents comparable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..corpus import ANALYSIS_TEXT, ANALYSIS_WORD_COUNT, TEXT, Corpus
from ..logging_utils import get_logger
from .base import StepResult

logger = get_logger(__name__)

_WORD = re.compile(r"\b\w+\b")
_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)")


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


@dataclass
class TextWindower:
    """Cut each text to ``target_words`` words, extended to a sentence end within ``hard_cap``.

    A text of at most ``hard_cap`` words is returned unchanged. Otherwise the window ends at
    the first sentence boundary between word ``target_words`` and word ``hard_cap``; if there
    is none, it is cut right after word ``hard_cap``.
    """

    target_words: int = 400
    hard_cap: int = 440

    def window(self, text: str) -> str:
        if not text:
            return ""
        words = list(_WORD.finditer(text))
        if len(words) <= self.hard_cap:
            return text

        target_end = words[min(self.target_words, self.hard_cap) - 1].end()
        cap_end = words[self.hard_cap - 1].end()
        for sentence_end in _SENTENCE_END.finditer(text):
            if target_end <= sentence_end.end() <= cap_end:
                return text[:sentence_end.end()]
        return text[:cap_end]

    def apply(self, corpus: Corpus, min_words: int | None = None, enabled: bool = True
              ) -> tuple[Corpus, StepResult]:
        """Add ``analysis_text`` / ``analysis_word_count`` and drop windows below ``min_words``.

        With ``enabled=False`` the cleaned text is used as-is (e.g. for full papers).
        """
        documents = corpus.documents.copy()
        texts = documents[TEXT].astype(object)
        documents[ANALYSIS_TEXT] = texts.map(self.window) if enabled else texts
        documents[ANALYSIS_WORD_COUNT] = documents[ANALYSIS_TEXT].map(count_words)

        before = len(documents)
        if enabled and min_words:
            documents = documents.loc[documents[ANALYSIS_WORD_COUNT] >= min_words]
        result = StepResult(
            step="window",
            description=f"windows under {min_words} words" if enabled else "window disabled",
            documents_dropped=before - len(documents),
            documents_remaining=len(documents),
        )
        logger.info("Analysis windows: %d documents kept, %d dropped as too short",
                    len(documents), before - len(documents))
        return corpus.replace(documents), result
