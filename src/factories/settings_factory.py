"""
src/factories/settings_factory.py

The only module allowed to read os.environ. Builds a fully frozen Settings object.
Phase 5: reads PiiSettings and RelationalStoreSettings env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import (
    ChunkingSettings,
    ChromaSettings,
    IngestionSettings,
    OllamaSettings,
    PineconeSettings,
    PiiSettings,
    PromptSettings,
    QdrantSettings,
    RelationalStoreSettings,
    RetrievalSettings,
    SemanticChunkingSettings,
    Settings,
    VectorStoreType,
)


class SettingsFactory:
    def __init__(self, project_root: Path, env_file: str = ".env") -> None:
        self._project_root = project_root
        self._env_file = env_file

    def create(self, vector_store_type: VectorStoreType | None = None) -> Settings:
        load_dotenv(self._project_root / self._env_file)

        store_type = vector_store_type or VectorStoreType(
            self._optional("VECTOR_STORE_TYPE", VectorStoreType.PINECONE.value)
        )

        # PII enabled_types: comma-separated env var, e.g. "EMAIL,PHONE_PK,CNIC"
        pii_types_raw = self._optional("PII_ENABLED_TYPES", "")
        pii_types = tuple(t.strip() for t in pii_types_raw.split(",") if t.strip())

        return Settings(
            pinecone=PineconeSettings(
                api_key=self._optional("PINECONE_API_KEY", ""),
                index_name=self._optional("PINECONE_INDEX_NAME", "rag-index"),
                cloud=self._optional("PINECONE_CLOUD", "aws"),
                region=self._optional("PINECONE_REGION", "us-east-1"),
            ),
            ollama=OllamaSettings(
                base_url=self._optional("OLLAMA_BASE_URL", "http://localhost:11434"),
                embed_model=self._optional("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                generation_model=self._optional("OLLAMA_GENERATION_MODEL", "gemma3"),
                embedding_dimension=int(self._optional("OLLAMA_EMBED_DIMENSION", "768")),
            ),
            chunking=ChunkingSettings(
                chunk_size=int(self._optional("CHUNK_SIZE", "512")),
                chunk_overlap=int(self._optional("CHUNK_OVERLAP", "64")),
            ),
            semantic_chunking=SemanticChunkingSettings(
                similarity_threshold=float(
                    self._optional("SEMANTIC_SIMILARITY_THRESHOLD", "0.75")
                ),
                min_sentences_per_chunk=int(
                    self._optional("SEMANTIC_MIN_SENTENCES", "2")
                ),
                max_sentences_per_chunk=int(
                    self._optional("SEMANTIC_MAX_SENTENCES", "15")
                ),
            ),
            retrieval=RetrievalSettings(
                top_k=int(self._optional("RETRIEVAL_TOP_K", "5")),
            ),
            prompt=PromptSettings(
                max_context_chars=int(self._optional("MAX_CONTEXT_CHARS", "6000")),
            ),
            ingestion=IngestionSettings(
                upsert_batch_size=int(self._optional("UPSERT_BATCH_SIZE", "100")),
                embed_batch_size=int(self._optional("EMBED_BATCH_SIZE", "8")),
                embed_retries=int(self._optional("EMBED_RETRIES", "3")),
                docx_pseudo_page_chars=int(self._optional("DOCX_PSEUDO_PAGE_CHARS", "3000")),
            ),
            chroma=ChromaSettings(
                persist_directory=self._optional("CHROMA_PERSIST_DIR", "./data/chroma"),
                collection_name=self._optional("CHROMA_COLLECTION", "rag-collection"),
            ),
            qdrant=QdrantSettings(
                url=os.getenv("QDRANT_URL"),
                path=self._optional("QDRANT_PATH", "./data/qdrant"),
                collection_name=self._optional("QDRANT_COLLECTION", "rag-collection"),
            ),
            pii=PiiSettings(
                enabled=self._optional("PII_ENABLED", "true").lower() == "true",
                enabled_types=pii_types,
            ),
            relational_store=RelationalStoreSettings(
                enabled=self._optional("RELATIONAL_STORE_ENABLED", "true").lower() == "true",
                db_path=self._optional("RELATIONAL_STORE_DB_PATH", "./data/relational/rag_chunks.db"),
            ),
            vector_store_type=store_type,
            project_root=self._project_root,
        )

    @staticmethod
    def _require(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set."
            )
        return value

    @staticmethod
    def _optional(key: str, default: str) -> str:
        return os.getenv(key, default)