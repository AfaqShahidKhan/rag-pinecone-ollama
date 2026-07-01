"""
src/application/services/retrieval_service.py

Single responsibility: turn a natural-language query into ranked
SearchResults by embedding it and querying the vector store.
"""

from __future__ import annotations

from src.config.settings import RetrievalSettings
from src.domain.entities import SearchResult
from src.domain.interfaces import IEmbeddingProvider, ILogger, IVectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStore,
        logger: ILogger,
        retrieval_settings: RetrievalSettings,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._logger = logger
        self._settings = retrieval_settings

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        k = top_k or self._settings.top_k
        query_vector = self._embedding_provider.embed_query(query)
        results = self._vector_store.query(vector=query_vector, top_k=k)
        self._logger.debug(f"Retrieved {len(results)} results for query '{query}'.")
        return results
