"""
src/infrastructure/vector_store/qdrant_vector_store.py

IVectorStore adapter backed by Qdrant.

Two modes — both require zero external server by default:
  - url=None  → local on-disk persistence at QdrantSettings.path
  - url=str   → connects to a running Qdrant server (cloud or self-hosted)

Install:  pip install qdrant-client
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config.settings import IngestionSettings, QdrantSettings
from src.domain.entities import EmbeddedChunk, SearchResult
from src.domain.interfaces import ILogger, IVectorIdStrategy, IVectorStore

DISTANCE_METRIC = qdrant_models.Distance.COSINE


class QdrantVectorStore(IVectorStore):
    def __init__(
        self,
        id_strategy: IVectorIdStrategy,
        logger: ILogger,
        qdrant_settings: QdrantSettings,
        ingestion_settings: IngestionSettings,
        embedding_dimension: int,
    ) -> None:
        self._id_strategy = id_strategy
        self._logger = logger
        self._settings = qdrant_settings
        self._batch_size = ingestion_settings.upsert_batch_size
        self._dimension = embedding_dimension

        if qdrant_settings.url:
            self._client = QdrantClient(url=qdrant_settings.url)
            self._logger.info(f"Qdrant connected to server at '{qdrant_settings.url}'.")
        else:
            self._client = QdrantClient(path=qdrant_settings.path)
            self._logger.info(f"Qdrant using local storage at '{qdrant_settings.path}'.")

    def ensure_index_exists(self) -> None:
        collection_name = self._settings.collection_name
        existing = [c.name for c in self._client.get_collections().collections]

        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self._dimension,
                    distance=DISTANCE_METRIC,
                ),
            )
            self._logger.info(f"Qdrant collection '{collection_name}' created.")
        else:
            self._logger.info(f"Qdrant collection '{collection_name}' already exists.")

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        collection_name = self._settings.collection_name
        total = len(chunks)
        self._logger.info(
            f"Upserting {total} vectors to Qdrant in batches of {self._batch_size}..."
        )

        for start in range(0, total, self._batch_size):
            batch = chunks[start : start + self._batch_size]
            points = [
                qdrant_models.PointStruct(
                    id=self._sha256_to_uint64(self._id_strategy.generate_id(chunk)),
                    vector=chunk.vector,
                    payload={
                        "text":        chunk.document.page_content,
                        "source":      chunk.document.metadata["source"],
                        "page":        int(chunk.document.metadata["page"]),
                        "total_pages": int(chunk.document.metadata["total_pages"]),
                        "chunk_index": int(chunk.document.metadata["chunk_index"]),
                        "chunk_total": int(chunk.document.metadata["chunk_total"]),
                    },
                )
                for chunk in batch
            ]
            self._client.upsert(collection_name=collection_name, points=points)

        self._logger.info(f"Qdrant upsert complete — {total} vectors.")
        return total

    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        collection_name = self._settings.collection_name
        hits = self._client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

        results = [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                source=hit.payload.get("source", "unknown"),
                page=int(hit.payload.get("page", 0)),
                chunk_index=int(hit.payload.get("chunk_index", 0)),
            )
            for hit in hits
        ]

        self._logger.info(f"Qdrant query returned {len(results)} results (top_k={top_k}).")
        return results

    @staticmethod
    def _sha256_to_uint64(hex_id: str) -> int:
        """Qdrant requires integer or UUID point IDs; convert our hex SHA-256 prefix."""
        return int(hex_id[:16], 16)