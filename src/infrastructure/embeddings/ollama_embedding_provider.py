"""
src/infrastructure/embeddings/ollama_embedding_provider.py
"""

from __future__ import annotations

import time

from ollama import Client as OllamaClient

from src.config.settings import IngestionSettings, OllamaSettings
from src.domain.interfaces import IEmbeddingProvider, ILogger


class OllamaEmbeddingProvider(IEmbeddingProvider):
    def __init__(
        self,
        client: OllamaClient,
        logger: ILogger,
        ollama_settings: OllamaSettings,
        ingestion_settings: IngestionSettings,
    ) -> None:
        self._client = client
        self._logger = logger
        self._settings = ollama_settings
        self._retries = ingestion_settings.embed_retries
        self._batch_size = ingestion_settings.embed_batch_size

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dimension

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch_with_retry([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in configurable batches with per-batch retry."""
        total = len(texts)
        total_batches = (total + self._batch_size - 1) // self._batch_size
        all_vectors: list[list[float]] = []

        for batch_num, start in enumerate(range(0, total, self._batch_size), 1):
            batch = texts[start : start + self._batch_size]
            self._logger.debug(
                f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)..."
            )
            all_vectors.extend(self._embed_batch_with_retry(batch))

        self._logger.info(f"Embedded {total} chunks across {total_batches} batches.")
        return all_vectors

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._retries):
            try:
                response = self._client.embed(model=self._settings.embed_model, input=texts)
                return response.embeddings
            except Exception as exc:
                if attempt < self._retries - 1:
                    wait = 2 ** attempt
                    self._logger.warning(
                        f"Embedding attempt {attempt + 1} failed: {exc}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    self._logger.error(f"All {self._retries} embedding attempts failed for batch.")
                    raise
        raise RuntimeError("unreachable")