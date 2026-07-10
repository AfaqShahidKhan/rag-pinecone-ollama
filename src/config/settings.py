"""
src/config/settings.py

Pure configuration value objects. No module-level singleton instance.
Built once in the composition root via SettingsFactory, injected everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── Vector store selector ────────────────────────────────────────────────────

class VectorStoreType(str, Enum):
    PINECONE = "pinecone"
    CHROMA   = "chroma"
    QDRANT   = "qdrant"


# ── Per-database settings ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str = "rag-index"
    cloud: str = "aws"
    region: str = "us-east-1"


@dataclass(frozen=True)
class ChromaSettings:
    persist_directory: str = "./data/chroma"
    collection_name: str = "rag-collection"


@dataclass(frozen=True)
class QdrantSettings:
    # url=None → local on-disk mode (no server required)
    url: str | None = None
    path: str = "./data/qdrant"
    collection_name: str = "rag-collection"


# ── Other settings ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    generation_model: str = "gemma3"
    embedding_dimension: int = 768


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5


@dataclass(frozen=True)
class PromptSettings:
    max_context_chars: int = 6000
    system_prompt: str = (
        "You are a helpful question-answering assistant.\n\n"
        "Answer the user's question using ONLY the information in the context "
        "provided below.\n"
        "Follow these rules strictly:\n"
        "- Write a clear, complete answer in your own words — do NOT copy raw "
        "text from the context\n"
        "- Use 2-3 sentences unless the answer is a simple fact\n"
        "- If the answer requires explaining a reason or motivation, explain it fully\n"
        "- Cite which source number(s) support your answer at the end, like: [1] or [1, 3]\n"
        '- If the answer is not in the context, say: "I don\'t have enough information to answer that."\n'
        "- Never make up facts not present in the context"
    )
    prompt_template: str = "{system}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


@dataclass(frozen=True)
class IngestionSettings:
    upsert_batch_size: int = 100
    embed_batch_size: int = 8
    embed_retries: int = 3
    docx_pseudo_page_chars: int = 3000


# ── Root settings object ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    pinecone: PineconeSettings
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    chroma: ChromaSettings = field(default_factory=ChromaSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    vector_store_type: VectorStoreType = VectorStoreType.PINECONE
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"