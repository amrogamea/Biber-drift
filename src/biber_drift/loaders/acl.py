"""ACL Anthology papers from S2ORC, as written by ``biber-drift acquire-acl``."""

from __future__ import annotations

import pandas as pd

from ..corpus import DOC_ID, TEXT, YEAR, Corpus
from ..logging_utils import get_logger
from .base import CorpusLoader, register_loader
from .venues import VenueTierClassifier

logger = get_logger(__name__)

IMRAD_COLUMNS = ["abstract", "introduction", "methods", "results", "discussion"]

# Spelling variants of the same venue in Semantic Scholar metadata.
DEFAULT_VENUE_ALIASES = {
    "SIGDIAL Conference": "SIGDIAL Conferences",
    "WEBNLG": "WebNLG",
    "JEP/TALN/RECITAL": "JEPTALNRECITAL",
    "EVENTS@HLT-NAACL": "EVENTS@HLP-NAACL",
}


@register_loader
class ACLLoader(CorpusLoader):
    """One document per paper: the IMRaD sections joined into one text.

    ``path`` is either one CSV (e.g. a combined ``Dataset_all.csv``) or a directory of yearly
    ``Dataset_<year>.csv`` files. ``year`` is the publication year from Semantic Scholar; papers
    without one are dropped.

    Args:
        text_columns: sections joined (with a space) into ``text``.
        venue_aliases: venue spelling fixes applied before tiering.
        venue_tiers: optional reviewed CSV (``venue``, ``tier``) overriding the tier rules.
        tier_column: name of the added tier column (``None`` skips tiering).
        pattern: file glob when ``path`` is a directory.
    """

    name = "acl"

    def __init__(self, path, years=None, text_columns: list[str] | None = None,
                 id_column: str = "id", year_column: str = "year",
                 venue_aliases: dict[str, str] | None = None,
                 venue_tiers: str | None = None, tier_column: str | None = "venue_tier",
                 pattern: str = "Dataset_[0-9]*.csv") -> None:
        super().__init__(path, years)
        self.text_columns = text_columns or list(IMRAD_COLUMNS)
        self.id_column = id_column
        self.year_column = year_column
        self.venue_aliases = DEFAULT_VENUE_ALIASES if venue_aliases is None else venue_aliases
        self.tier_column = tier_column
        self.classifier = VenueTierClassifier(venue_tiers) if tier_column else None
        self.pattern = pattern

    def load(self) -> Corpus:
        frame = pd.concat(
            [pd.read_csv(path, low_memory=False) for path in self.csv_files(self.pattern)],
            ignore_index=True,
        )
        missing = ({self.id_column, self.year_column} | set(self.text_columns)) - set(frame.columns)
        if missing:
            raise ValueError(f"ACL data is missing column(s): {sorted(missing)}")

        years = pd.to_numeric(frame[self.year_column], errors="coerce")
        if years.isna().any():
            logger.info("Dropping %d papers without a publication year", int(years.isna().sum()))
        frame = frame.loc[years.notna()].copy()
        frame[YEAR] = years.loc[years.notna()].astype(int)
        if self.years is not None:
            frame = frame.loc[frame[YEAR].isin(self.years)]

        frame[TEXT] = frame[self.text_columns].fillna("").astype(str).agg(" ".join, axis=1)

        if "venue" in frame.columns:
            frame["venue"] = frame["venue"].replace(self.venue_aliases)
            if self.classifier is not None:
                frame[self.tier_column] = frame["venue"].map(self.classifier.classify)

        # The same ACL id can appear twice in S2ORC; keep both rows under distinct doc_ids
        # (``<id>``, ``<id>#2``) so the cleaner's duplicate-text step decides what to drop.
        ids = frame[self.id_column].astype(str)
        repeat = ids.groupby(ids).cumcount()
        frame[DOC_ID] = ids.where(repeat == 0, ids + "#" + (repeat + 1).astype(str))
        corpus = Corpus(frame.set_index(DOC_ID), name="acl")
        logger.info("Loaded %d ACL papers from %d year(s)", len(corpus), len(corpus.years))
        return corpus
