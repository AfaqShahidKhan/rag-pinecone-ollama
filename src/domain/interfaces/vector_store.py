"""
src/domain/interfaces/vector_store.py

Port for a vector database. Implementations (Pinecone, Qdrant, Chroma, ...)
plug in behind this interface without the application layer ever knowing
which database is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import EmbeddedChunk, SearchResult


class IVectorIdStrategy(ABC):
    """Strategy for deriving a stable, idempotent vector ID from a chunk."""

    @abstractmethod
    def generate_id(self, chunk: EmbeddedChunk) -> str: ...


class IVectorStore(ABC):
    @abstractmethod
    def ensure_index_exists(self) -> None:
        """Create the backing index/collection if it does not already exist."""
        ...

    @abstractmethod
    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Upsert embedded chunks. Returns the number of vectors written."""
        ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the top_k most similar chunks to the given vector."""
        ...
