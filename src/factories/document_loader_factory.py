"""
src/factories/document_loader_factory.py

Abstract factory: holds the set of registered IDocumentLoader adapters and
resolves the correct one per file. Supported extensions are derived
dynamically from the registered loaders — no hardcoded extension list.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, IDocumentLoaderResolver, ILogger


class DocumentLoaderFactory(IDocumentLoaderResolver):
    def __init__(self, loaders: list[IDocumentLoader], logger: ILogger) -> None:
        self._loaders = loaders
        self._logger = logger

    def resolve_for_file(self, path: Path) -> IDocumentLoader:
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise ValueError(
            f"Unsupported file type: '{path.suffix}'. "
            f"No registered loader handles this extension."
        )

    def load_all_from_directory(self, directory: Path) -> list[Document]:
        # Dynamically collect every file any registered loader can handle
        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and any(loader.supports(p) for loader in self._loaders)
        )

        if not files:
            raise FileNotFoundError(
                f"No supported files found in '{directory}'. "
                f"Place PDF, DOCX, HTML, JSON, or image files there."
            )

        self._logger.info(
            f"Found {len(files)} file(s) in '{directory}': "
            f"{[f.name for f in files]}"
        )

        all_docs: list[Document] = []
        for file_path in files:
            loader = self.resolve_for_file(file_path)
            all_docs.extend(loader.load(file_path))

        self._logger.info(f"Total sections loaded: {len(all_docs)}")
        return all_docs