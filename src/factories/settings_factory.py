"""
src/factories/settings_factory.py

The only module allowed to read os.environ. Builds a fully frozen Settings object.

Precedence for every customizable value (db, chunk size, top_k, etc.):
    YAML config (config/default.yml, then an optional per-user override file)
        > environment variable (.env)
        > hardcoded dataclass default

Secrets (e.g. PINECONE_API_KEY) are only ever read from the environment —
never put a secret in a YAML file that might get committed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config.settings import (
    ChunkingSettings,
    ChromaSettings,
    CorpusSettings,
    ImageExtractionSettings,
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
    TableExtractionSettings,
    VectorStoreType,
)
from src.factories.yaml_config_loader import YamlConfigLoader


class SettingsFactory:
    def __init__(self, project_root: Path, env_file: str = ".env") -> None:
        self._project_root = project_root
        self._env_file = env_file
        self._yaml_loader = YamlConfigLoader(project_root=project_root)

    def create(
        self,
        vector_store_type: VectorStoreType | None = None,
        config_file: str | Path | None = None,
    ) -> Settings:
        load_dotenv(self._project_root / self._env_file)
        yaml_config = self._yaml_loader.load(config_file)

        store_type = vector_store_type or VectorStoreType(
            self._value(yaml_config, "vector_store_type", "VECTOR_STORE_TYPE", VectorStoreType.PINECONE.value)
        )

        pii_types_yaml = self._dig(yaml_config, "pii.enabled_types")
        if pii_types_yaml is not None:
            pii_types = tuple(str(t).strip() for t in pii_types_yaml if str(t).strip())
        else:
            pii_types_raw = self._optional("PII_ENABLED_TYPES", "")
            pii_types = tuple(t.strip() for t in pii_types_raw.split(",") if t.strip())

        return Settings(
            pinecone=PineconeSettings(
                api_key=self._optional("PINECONE_API_KEY", ""),  # secret — env only, never YAML
                index_name=self._value(yaml_config, "pinecone.index_name", "PINECONE_INDEX_NAME", "rag-index"),
                cloud=self._value(yaml_config, "pinecone.cloud", "PINECONE_CLOUD", "aws"),
                region=self._value(yaml_config, "pinecone.region", "PINECONE_REGION", "us-east-1"),
            ),
            ollama=OllamaSettings(
                base_url=self._value(yaml_config, "ollama.base_url", "OLLAMA_BASE_URL", "http://localhost:11434"),
                embed_model=self._value(yaml_config, "ollama.embed_model", "OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                generation_model=self._value(yaml_config, "ollama.generation_model", "OLLAMA_GENERATION_MODEL", "gemma3"),
                embedding_dimension=int(
                    self._value(yaml_config, "ollama.embedding_dimension", "OLLAMA_EMBED_DIMENSION", 768)
                ),
            ),
            chunking=ChunkingSettings(
                chunk_size=int(self._value(yaml_config, "chunking.chunk_size", "CHUNK_SIZE", 512)),
                chunk_overlap=int(self._value(yaml_config, "chunking.chunk_overlap", "CHUNK_OVERLAP", 64)),
            ),
            semantic_chunking=SemanticChunkingSettings(
                similarity_threshold=float(
                    self._value(
                        yaml_config, "semantic_chunking.similarity_threshold",
                        "SEMANTIC_SIMILARITY_THRESHOLD", 0.75,
                    )
                ),
                min_sentences_per_chunk=int(
                    self._value(yaml_config, "semantic_chunking.min_sentences_per_chunk", "SEMANTIC_MIN_SENTENCES", 2)
                ),
                max_sentences_per_chunk=int(
                    self._value(
                        yaml_config, "semantic_chunking.max_sentences_per_chunk", "SEMANTIC_MAX_SENTENCES", 15
                    )
                ),
            ),
            retrieval=RetrievalSettings(
                top_k=int(self._value(yaml_config, "retrieval.top_k", "RETRIEVAL_TOP_K", 5)),
            ),
            prompt=PromptSettings(
                max_context_chars=int(
                    self._value(yaml_config, "prompt.max_context_chars", "MAX_CONTEXT_CHARS", 6000)
                ),
            ),
            ingestion=IngestionSettings(
                upsert_batch_size=int(
                    self._value(yaml_config, "ingestion.upsert_batch_size", "UPSERT_BATCH_SIZE", 100)
                ),
                embed_batch_size=int(
                    self._value(yaml_config, "ingestion.embed_batch_size", "EMBED_BATCH_SIZE", 8)
                ),
                embed_retries=int(
                    self._value(yaml_config, "ingestion.embed_retries", "EMBED_RETRIES", 3)
                ),
                docx_pseudo_page_chars=int(
                    self._value(yaml_config, "ingestion.docx_pseudo_page_chars", "DOCX_PSEUDO_PAGE_CHARS", 3000)
                ),
            ),
            chroma=ChromaSettings(
                persist_directory=self._value(
                    yaml_config, "chroma.persist_directory", "CHROMA_PERSIST_DIR", "./data/chroma"
                ),
                collection_name=self._value(
                    yaml_config, "chroma.collection_name", "CHROMA_COLLECTION", "rag-collection"
                ),
            ),
            qdrant=QdrantSettings(
                url=self._value(yaml_config, "qdrant.url", "QDRANT_URL", None) or None,
                path=self._value(yaml_config, "qdrant.path", "QDRANT_PATH", "./data/qdrant"),
                collection_name=self._value(yaml_config, "qdrant.collection_name", "QDRANT_COLLECTION", "rag-collection"),
            ),
            pii=PiiSettings(
                enabled=self._bool(yaml_config, "pii.enabled", "PII_ENABLED", True),
                enabled_types=pii_types,
            ),
            relational_store=RelationalStoreSettings(
                enabled=self._bool(yaml_config, "relational_store.enabled", "RELATIONAL_STORE_ENABLED", True),
                db_path=self._value(
                    yaml_config, "relational_store.db_path", "RELATIONAL_STORE_DB_PATH",
                    "./data/relational/rag_chunks.db",
                ),
            ),
            corpus=CorpusSettings(
                enabled=self._bool(yaml_config, "corpus.enabled", "CORPUS_WRITER_ENABLED", True),
                output_dir=self._value(yaml_config, "corpus.output_dir", "CORPUS_OUTPUT_DIR", "./data/corpus"),
            ),
            table_extraction=TableExtractionSettings(
                enabled=self._bool(yaml_config, "table_extraction.enabled", "TABLE_EXTRACTION_ENABLED", True),
            ),
            image_extraction=ImageExtractionSettings(
                enabled=self._bool(yaml_config, "image_extraction.enabled", "IMAGE_EXTRACTION_ENABLED", True),
                output_dir=self._value(
                    yaml_config, "image_extraction.output_dir", "IMAGE_OUTPUT_DIR", "./data/images"
                ),
            ),
            vector_store_type=store_type,
            project_root=self._project_root,
        )

    # ── Layered value resolution: YAML > env > default ─────────────────────────

    @staticmethod
    def _dig(config: dict[str, Any], dotted_key: str) -> Any:
        """Walk a dotted path (e.g. 'ollama.embed_model') through nested dicts."""
        node: Any = config
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def _value(self, yaml_config: dict[str, Any], yaml_key: str, env_key: str, default: Any) -> Any:
        yaml_value = self._dig(yaml_config, yaml_key)
        if yaml_value is not None:
            return yaml_value
        return os.getenv(env_key, default)

    def _bool(self, yaml_config: dict[str, Any], yaml_key: str, env_key: str, default: bool) -> bool:
        yaml_value = self._dig(yaml_config, yaml_key)
        if isinstance(yaml_value, bool):
            return yaml_value
        if yaml_value is not None:
            return str(yaml_value).strip().lower() == "true"
        return self._optional(env_key, str(default)).lower() == "true"

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