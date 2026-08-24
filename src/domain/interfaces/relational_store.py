"""
src/domain/interfaces/relational_store.py

Port for a relational/document store that persists landing_zone chunk text and
metadata alongside the vector store. Enables the Parent-Child Retrieval
pattern: the vector store holds dense embeddings for fast similarity
search, while the relational store holds the full original text for
context reconstruction without re-fetching from the source documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import Document, EmbeddedChunk


class IRelationalStore(ABC):
    @abstractmethod
    def ensure_schema(self) -> None:
        """Create tables/collections if they do not already exist."""
        ...

    @abstractmethod
    def save_chunks(self, chunks: list[EmbeddedChunk], vector_ids: list[str]) -> int:
        """
        Persist chunk text and metadata alongside their vector store IDs.
        Returns the number of rows written.
        """
        ...

    @abstractmethod
    def get_chunk_by_vector_id(self, vector_id: str) -> Document | None:
        """Retrieve the original chunk Document by its vector store ID."""
        ...

    @abstractmethod
    def search_by_source(self, source: str) -> list[Document]:
        """Return all chunks from a given source file."""
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source file. Returns rows deleted."""
        ...