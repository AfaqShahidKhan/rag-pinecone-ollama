"""
src/factories/settings_factory.py

Factory responsible for the one thing nothing else in the codebase should
do directly: reading environment variables and turning them into a frozen
Settings object. Everything downstream receives Settings by injection.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import (
    ChunkingSettings,
    IngestionSettings,
    OllamaSettings,
    PineconeSettings,
    PromptSettings,
    RetrievalSettings,
    Settings,
)


class SettingsFactory:
    """Abstract-factory-style builder: one method, one fully-formed Settings object."""

    def __init__(self, project_root: Path, env_file: str = ".env") -> None:
        self._project_root = project_root
        self._env_file = env_file

    def create(self) -> Settings:
        load_dotenv(self._project_root / self._env_file)

        return Settings(
            pinecone=PineconeSettings(
                api_key=self._require("PINECONE_API_KEY"),
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
            project_root=self._project_root,
        )

    @staticmethod
    def _require(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{key}' is not set. "
                f"Copy .env.example -> .env and fill in the values."
            )
        return value

    @staticmethod
    def _optional(key: str, default: str) -> str:
        return os.getenv(key, default)
