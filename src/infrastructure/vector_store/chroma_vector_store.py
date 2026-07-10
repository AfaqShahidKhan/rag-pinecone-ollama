"""
src/infrastructure/vector_store/chroma_vector_store.py

IVectorStore adapter backed by ChromaDB (local, no API key required).
Data is persisted on disk at ChromaSettings.persist_directory so it
survives restarts. Switching to in-memory mode is a one-line change:
    chromadb.Client()  instead of  chromadb.PersistentClient(path=...)
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaInternalSettings

from src.config.settings import ChromaSettings, IngestionSettings
from src.domain.entities import EmbeddedChunk, SearchResult
from src.domain.interfaces import ILogger, IVectorIdStrategy, IVectorStore


class ChromaVectorStore(IVectorStore):
    def __init__(
        self,
        id_strategy: IVectorIdStrategy,
        logger: ILogger,
        chroma_settings: ChromaSettings,
        ingestion_settings: IngestionSettings,
        embedding_dimension: int,
    ) -> None:
        self._id_strategy = id_strategy
        self._logger = logger
        self._settings = chroma_settings
        self._batch_size = ingestion_settings.upsert_batch_size
        self._dimension = embedding_dimension
        self._client = chromadb.PersistentClient(
            path=chroma_settings.persist_directory,
            settings=ChromaInternalSettings(anonymized_telemetry=False),
        )
        self._collection = None

    def ensure_index_exists(self) -> None:
        self._collection = self._client.get_or_create_collection(
            name=self._settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._logger.info(
            f"Chroma collection '{self._settings.collection_name}' ready "
            f"(persisted at '{self._settings.persist_directory}')."
        )

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        collection = self._get_collection()
        total = len(chunks)
        self._logger.info(f"Upserting {total} vectors to Chroma in batches of {self._batch_size}...")

        for start in range(0, total, self._batch_size):
            batch = chunks[start : start + self._batch_size]
            ids, embeddings, metadatas, documents = [], [], [], []

            for chunk in batch:
                ids.append(self._id_strategy.generate_id(chunk))
                embeddings.append(chunk.vector)
                documents.append(chunk.document.page_content)
                metadatas.append({
                    "source":       chunk.document.metadata["source"],
                    "page":         int(chunk.document.metadata["page"]),
                    "total_pages":  int(chunk.document.metadata["total_pages"]),
                    "chunk_index":  int(chunk.document.metadata["chunk_index"]),
                    "chunk_total":  int(chunk.document.metadata["chunk_total"]),
                })

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        self._logger.info(f"Chroma upsert complete — {total} vectors.")
        return total

    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        collection = self._get_collection()
        response = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        ids       = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        for doc_text, meta, distance in zip(documents, metadatas, distances):
            # Chroma returns L2 or cosine distance; with hnsw:space=cosine, distance ∈ [0,2]
            # Convert to similarity score ∈ [0,1]
            score = max(0.0, 1.0 - distance / 2.0)
            results.append(SearchResult(
                text=doc_text,
                score=score,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
            ))

        self._logger.info(f"Chroma query returned {len(results)} results (top_k={top_k}).")
        return results

    def _get_collection(self):
        if self._collection is None:
            self.ensure_index_exists()
        return self._collection