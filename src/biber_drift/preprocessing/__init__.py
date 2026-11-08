"""Text cleaning, quality diagnostics and windowing. Importing registers the built-in cleaners."""

from .acl import ACLCleaner
from .ao3 import AO3Cleaner
from .base import (
    CLEANERS, CleaningReport, CorpusCleaner, DocumentFilter, StepResult, TextStep,
    create_cleaner, register_cleaner,
)
from .quality import QualityDiagnostics, QualityReport
from .windowing import TextWindower, count_words

__all__ = [
    "ACLCleaner", "AO3Cleaner", "CLEANERS", "CleaningReport", "CorpusCleaner", "DocumentFilter",
    "QualityDiagnostics", "QualityReport", "StepResult", "TextStep", "TextWindower",
    "count_words", "create_cleaner", "register_cleaner",
]
