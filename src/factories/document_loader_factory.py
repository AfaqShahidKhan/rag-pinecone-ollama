"""
src/factories/document_loader_factory.py

Abstract factory: holds the set of registered IDocumentLoader adapters and
resolves the correct one per file extension. This is the ONLY place in the
codebase that knows which concrete loader classes exist; every other layer
only ever sees IDocumentLoader / IDocumentLoaderResolver.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, IDocumentLoaderResolver, ILogger

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class DocumentLoaderFactory(IDocumentLoaderResolver):
    def __init__(self, loaders: list[IDocumentLoader], logger: ILogger) -> None:
        """
        Args:
            loaders: every available IDocumentLoader adapter, injected by the
                     composition root (no loader is instantiated here).
            logger:  injected logger, never imported globally.
        """
        self._loaders = loaders
        self._logger = logger

    def resolve_for_file(self, path: Path) -> IDocumentLoader:
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise ValueError(
            f"Unsupported file type: '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    def load_all_from_directory(self, directory: Path) -> list[Document]:
        files = sorted(
            p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            raise FileNotFoundError(
                f"No supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) "
                f"found in '{directory}'."
            )

        self._logger.info(f"Found {len(files)} file(s) in '{directory}': {[f.name for f in files]}")

        all_docs: list[Document] = []
        for file_path in files:
            loader = self.resolve_for_file(file_path)
            all_docs.extend(loader.load(file_path))

        self._logger.info(f"Total sections loaded: {len(all_docs)}")
        return all_docs
