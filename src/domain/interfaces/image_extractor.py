"""
src/domain/interfaces/image_extractor.py

Ports for extracting embedded images from source documents (PDF, DOCX)
and saving them to disk. Separate from IDocumentLoader because loaders
produce text Documents; image extraction is a parallel concern that
attaches file references to that same Document's metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import Document


class IImageExtractor(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool:
        ...

    @abstractmethod
    def extract(self, path: Path, documents: list[Document]) -> list[Document]:
        """
        Extracts images embedded in the source file at `path`, saves them to
        disk, and attaches image references to the metadata of whichever
        Document(s) in `documents` came from that file (matched via
        metadata["source_path"]). Documents are mutated in place; the same
        list is returned for chaining.
        """
        ...


class IImageExtractorResolver(ABC):
    @abstractmethod
    def resolve_for_file(self, path: Path) -> IImageExtractor | None:
        """Returns None when no registered extractor supports this file type."""
        ...