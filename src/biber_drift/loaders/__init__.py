"""Dataset loaders. Importing this package registers the built-in loaders."""

from .acl import ACLLoader
from .ao3 import AO3Loader
from .base import LOADERS, CorpusLoader, create_loader, register_loader
from .venues import VenueTierClassifier

__all__ = [
    "ACLLoader", "AO3Loader", "CorpusLoader", "LOADERS", "VenueTierClassifier",
    "create_loader", "register_loader",
]
