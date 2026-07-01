"""
src/infrastructure/vector_store/pinecone_vector_store.py

IVectorStore adapter backed by Pinecone. The pinecone.Pinecone client is
injected (built by PineconeClientFactory) rather than constructed here,
keeping this class focused on a single responsibility: translating
domain entities to/from Pinecone's wire format.
"""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config.settings import IngestionSettings, PineconeSettings
from src.domain.entities import EmbeddedChunk, SearchResult
from src.domain.interfaces import ILogger, IVectorIdStrategy, IVectorStore

METRIC = "cosine"


class PineconeVectorStore(IVectorStore):
    def __init__(
        self,
        client: Pinecone,
        id_strategy: IVectorIdStrategy,
        logger: ILogger,
        pinecone_settings: PineconeSettings,
        ingestion_settings: IngestionSettings,
        embedding_dimension: int,
    ) -> None:
        self._client = client
        self._id_strategy = id_strategy
        self._logger = logger
        self._settings = pinecone_settings
        self._batch_size = ingestion_settings.upsert_batch_size
        self._dimension = embedding_dimension
        self._index = None  # lazily resolved, see _get_index()

    def ensure_index_exists(self) -> None:
        index_name = self._settings.index_name
        existing = [i.name for i in self._client.list_indexes()]

        if index_name not in existing:
            self._logger.info(f"Index '{index_name}' not found — creating serverless index...")
            self._client.create_index(
                name=index_name,
                dimension=self._dimension,
                metric=METRIC,
                spec=ServerlessSpec(cloud=self._settings.cloud, region=self._settings.region),
            )
            self._logger.info(f"Index '{index_name}' created successfully.")
        else:
            self._logger.info(f"Index '{index_name}' already exists — skipping creation.")

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        vectors = [self._to_pinecone_vector(chunk) for chunk in chunks]
        total = len(vectors)
        index = self._get_index()

        self._logger.info(f"Upserting {total} vectors to Pinecone in batches of {self._batch_size}...")
        for i in range(0, total, self._batch_size):
            batch = vectors[i : i + self._batch_size]
            index.upsert(vectors=batch)

        self._logger.info(f"Upsert complete — {total} vectors in index.")
        return total

    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        index = self._get_index()
        response = index.query(vector=vector, top_k=top_k, include_metadata=True)

        results: list[SearchResult] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(SearchResult(
                text=meta.get("text", ""),
                score=match.score,
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
            ))

        self._logger.info(f"Query returned {len(results)} results (top_k={top_k}).")
        return results

    def _to_pinecone_vector(self, chunk: EmbeddedChunk) -> dict:
        doc = chunk.document
        return {
            "id": self._id_strategy.generate_id(chunk),
            "values": chunk.vector,
            "metadata": {
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
                "total_pages": doc.metadata["total_pages"],
                "chunk_index": doc.metadata["chunk_index"],
                "chunk_total": doc.metadata["chunk_total"],
                "text": doc.page_content,
            },
        }

    def _get_index(self):
        if self._index is None:
            self._index = self._client.Index(self._settings.index_name)
        return self._index
