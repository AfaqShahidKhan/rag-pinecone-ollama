"""
src/domain/interfaces/text_chunker.py

Port for splitting page-level Documents into overlapping, embedding-sized chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import Document


class ITextChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split page-level Documents into chunk-level Documents."""
        ...
