"""
src/domain/interfaces/document_loader_resolver.py

Abstract-factory port: given a path, resolve the correct concrete
IDocumentLoader (or load an entire directory using whichever loaders
apply). The application layer depends only on this contract, never on
the concrete DocumentLoaderFactory implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces.document_loader import IDocumentLoader


class IDocumentLoaderResolver(ABC):
    @abstractmethod
    def resolve_for_file(self, path: Path) -> IDocumentLoader:
        """Return the loader capable of handling this single file."""
        ...

    @abstractmethod
    def load_all_from_directory(self, directory: Path) -> list[Document]:
        """Load every supported file found recursively under directory."""
        ...
