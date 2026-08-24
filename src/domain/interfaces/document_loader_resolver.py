"""
src/domain/interfaces/document_loader_resolver.py

Abstract-factory port: given a path, resolve the correct concrete
IDocumentLoader (or load an entire directory using whichever loaders
apply). The application layer depends only on this contract, never on
the concrete DocumentLoaderFactory implementation.

list_supported_files() is pure discovery (no loading) — it lets callers
(IngestionService, StreamingIngestionService, CorpusBuilderService) loop
file-by-file and run per-file validation/quarantine checks around each
load, instead of load_all_from_directory()'s all-at-once behavior which
has no hook for that.
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
    def list_supported_files(self, directory: Path) -> list[Path]:
        """Return every file under directory (recursively) some registered loader supports, sorted."""
        ...

    @abstractmethod
    def load_all_from_directory(self, directory: Path) -> list[Document]:
        """Load every supported file found recursively under directory."""
        ...