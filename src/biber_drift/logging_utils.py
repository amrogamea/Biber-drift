"""Logging setup shared by the CLI and the pipeline."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# Third-party loggers that are chatty at INFO (pybiber logs every batch, matplotlib every axis).
_QUIET = ("pybiber", "matplotlib", "PIL", "urllib3")


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once (safe to call repeatedly)."""
    logging.basicConfig(level=level.upper(), format=_FORMAT, datefmt="%H:%M:%S", force=True)
    for name in _QUIET:
        logging.getLogger(name).setLevel(max(logging.WARNING, logging.getLogger().level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
