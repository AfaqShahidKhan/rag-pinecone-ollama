"""
src/domain/interfaces/embedding_provider.py

Port for turning text into embedding vectors. Implementations may call
Ollama, OpenAI, a local model, etc. - callers never know which.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IEmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input text."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of vectors this provider produces."""
        ...
