"""Classify ACL Anthology venue strings into tiers: A*, Conference, Workshop."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TIERS = ("A*", "Conference", "Workshop")

ASTAR_EXACT = {
    "annual meeting of the association for computational linguistics",
    "conference on empirical methods in natural language processing",
}

# Venues that are workshops/shared tasks in practice even though their name alone doesn't say
# "workshop" and doesn't contain "@".
WORKSHOP_NAME_OVERRIDES = {
    "findings", "starsem", "wnut", "law", "mwe", "socialnlp", "crac", "woah",
    "argmining", "fever", "discomt", "blackboxnlp", "codeswitch", "sigtyp",
    "sigmorphon", "lchange", "repl4nlp", "nlp4call", "wassa", "bea", "smm4h",
    "figlang", "trustnlp", "genbench", "nlp4dh", "gebnlp", "tsar", "clpsych",
    "nlp4pi", "computel", "privatenlp", "sustainlp", "eval4nlp", "sicon",
    "dash", "mrl", "americasnlp", "banglalp", "dravidianlangtech",
    "ltedi", "case", "wildre", "wanlp", "nlpcss", "nlp4prog",
}

# Main/flagship conferences whose name doesn't literally contain "conference".
CONFERENCE_NAME_OVERRIDES = {
    "coling", "naacl", "eacl", "aacl", "lrec", "conll", "sigdial conferences",
    "wmt", "inlg", "tacl", "cl", "emnlp-ijcnlp",
    "north american chapter of the association for computational linguistics",
    "conference of the european chapter of the association for computational linguistics",
    "international conference on computational linguistics",
    "international conference on language resources and evaluation",
    "conference on computational natural language learning",
    "conference on machine translation",
    "international conference on natural language generation",
    "transactions of the association for computational linguistics",
}


def normalize_venue(venue: str) -> str:
    return re.sub(r"\s+", " ", str(venue)).strip().lower()


class VenueTierClassifier:
    """Heuristic venue -> tier rules, optionally overridden by a reviewed CSV.

    Rules, checked in order: exact ACL/EMNLP main -> ``A*``; a known workshop name, any venue
    containing ``@`` or the word "workshop" -> ``Workshop``; everything else -> ``Conference``.
    This is a heuristic, not ground truth: review the CSV written by :meth:`table` and pass it
    back as ``overrides`` (columns ``venue`` and ``tier``) to fix individual venues.
    """

    def __init__(self, overrides: str | Path | None = None) -> None:
        self.overrides: dict[str, str] = {}
        if overrides:
            table = pd.read_csv(overrides)
            if not {"venue", "tier"} <= set(table.columns):
                raise ValueError(f"{overrides} needs 'venue' and 'tier' columns")
            bad = set(table["tier"].dropna()) - set(TIERS)
            if bad:
                raise ValueError(f"{overrides} has unknown tiers: {sorted(bad)}")
            self.overrides = dict(zip(table["venue"].astype(str), table["tier"]))

    @staticmethod
    def rule(venue: str) -> str:
        norm = normalize_venue(venue)
        if norm in ASTAR_EXACT:
            return "A*"
        if norm in WORKSHOP_NAME_OVERRIDES:
            return "Workshop"
        if "@" in str(venue) or "workshop" in norm:
            return "Workshop"
        if norm in CONFERENCE_NAME_OVERRIDES:
            return "Conference"
        return "Conference"  # safest default for unrecognised, conference-looking acronyms

    def classify(self, venue) -> str | None:
        if venue is None or (isinstance(venue, float) and pd.isna(venue)) or str(venue).strip() == "":
            return None
        return self.overrides.get(str(venue), self.rule(venue))

    def table(self, venues: pd.Series) -> pd.DataFrame:
        """Venue / paper count / tier table, for review."""
        counts = venues.dropna().value_counts().rename_axis("venue").reset_index(name="count")
        counts["tier"] = counts["venue"].map(self.classify)
        return counts
