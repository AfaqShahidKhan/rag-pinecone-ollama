"""
src/domain/interfaces/document_loader.py

Port for turning a file on disk into a list of page/section-level Documents.
One concrete adapter per file format (PDF, DOCX, ...) implements this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import Document


class IDocumentLoader(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this loader can handle the given file."""
        ...

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load a single file and return one Document per page/section."""
        ...
