"""Strip boilerplate from the model outputs before tagging (``filters.clean`` in the config).

Each cleaner undoes what the check flag of the same name detects, and only where the original does
not show it itself: a chatty first line ("Sure! Here's the paraphrased text:"), a model note as the
last line ("I hope this ...", "Note: ..."), markdown the original lacks (headings, bold, list
markers, code fences, rules) and quotes around the whole output. The words of the paraphrase stay;
a cleaner never rewrites them.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from ..config import ConfigError
from .check import _CLOSER, _CLOSING_QUOTES, _MARKDOWN, _OPENER, _OPENING_QUOTES, _first_line, _last_line, _wrapped

CLEANERS: dict[str, Callable[[str, str], str]] = {}


def register_cleaner(name: str) -> Callable:
    """Decorator: add ``clean(output, original) -> output`` as the cleaner of check flag ``name``.

    Cleaners run in the order they are registered, whatever the order in the config.
    """
    def decorator(clean: Callable[[str, str], str]) -> Callable:
        CLEANERS[name] = clean
        return clean
    return decorator


def clean_output(output: str, original: str, names: Iterable[str]) -> tuple[str, list[str]]:
    """``output`` with the boilerplate of the cleaners ``names`` removed, and the ones that changed it."""
    names = set(names)
    unknown = names - set(CLEANERS)
    if unknown:
        raise ConfigError(f"Unknown cleaner(s) {sorted(unknown)}; choose from {list(CLEANERS)}")
    applied = []
    for name, clean in CLEANERS.items():
        if name in names:
            cleaned = clean(output, original)
            if cleaned != output:
                applied.append(name)
            output = cleaned
    return output.strip(), applied


@register_cleaner("opener")
def strip_opener(output: str, original: str) -> str:
    """Remove "Sure! Here's the paraphrased text:" (in the main run always a line of its own, ending
    in a colon); text after the colon on the same line is kept."""
    lines = output.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or not _OPENER.search(lines[first]) or _OPENER.search(_first_line(original)):
        return output
    _, colon, rest = lines[first].partition(":")
    if not colon and len(lines[first].split()) > 25:     # no colon and long: may hold the paraphrase
        return output
    lines[first] = rest.lstrip(" *_") if colon else ""   # " **" closes "**Paraphrased Text:**"
    return "\n".join(lines[first:]).strip()


@register_cleaner("closer")
def strip_closer(output: str, original: str) -> str:
    """Remove a model note as the last line ("I hope this ...", "Note: ..."). Only the last line: further
    up, such lines are often story text (AO3 letters: "I hope this letter finds you well")."""
    lines = output.rstrip().split("\n")
    if (sum(bool(line.strip()) for line in lines) < 2 or not _CLOSER.search(lines[-1])
            or _CLOSER.search(_last_line(original))):
        return output
    return "\n".join(lines[:-1]).rstrip()


# How to undo each markdown pattern of the check, in the same order as check._MARKDOWN.
_UNMARK = (
    (re.compile(r"^([ \t]*)#{1,6}[ \t]+", re.MULTILINE), r"\1"),                          # heading
    (re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__"), lambda m: m.group(1) or m.group(2)),  # bold
    (re.compile(r"^([ \t]*)(?:[-*•]|\d+[.)])[ \t]+(?=\S)", re.MULTILINE), r"\1"),         # list marker
    (re.compile(r"^[ \t]*```[^\n]*\n?|```", re.MULTILINE), ""),                            # code fence
    (re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*\n?", re.MULTILINE), ""),            # rule
)
assert len(_UNMARK) == len(_MARKDOWN)


@register_cleaner("markdown")
def strip_markdown(output: str, original: str) -> str:
    """Remove the markdown marks the original lacks (pattern by pattern); the marked words stay."""
    for detect, (pattern, replacement) in zip(_MARKDOWN, _UNMARK):
        if detect.search(output) and not detect.search(original):
            output = pattern.sub(replacement, output)
    return output


@register_cleaner("quotes")
def strip_quotes(output: str, original: str) -> str:
    """Remove quotes around the whole output, unless the original itself opens or closes with one
    (a chapter that starts or ends with dialogue)."""
    text, source = output.strip(), original.strip()
    if (not _wrapped(text) or (source and (source[0] in _OPENING_QUOTES or source[-1] in _CLOSING_QUOTES))):
        return output
    return text[1:-1].strip()
