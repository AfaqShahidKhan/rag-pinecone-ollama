"""
src/domain/entities.py

Pure domain data structures. This module must never import an external
SDK (pinecone, ollama, pypdf, docx, rich, etc.). It is the one layer every
other layer is allowed to depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """
    A unit of text with provenance metadata.
    Flows through the whole ingestion pipeline: load -> chunk -> embed -> upsert.
    """
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A Document paired with its embedding vector, ready for upsert."""
    document: Document
    vector: list[float]


@dataclass
class SearchResult:
    """A single retrieved chunk, ranked by similarity score."""
    text: str
    score: float
    source: str
    page: int
    chunk_index: int


@dataclass
class PromptPackage:
    """The fully assembled prompt plus the metadata needed to trace it back."""
    prompt: str
    question: str
    sources: list[SearchResult]
    context_chunks_used: int
    context_chars: int


@dataclass
class RAGResponse:
    """The final answer returned to the caller, with full provenance."""
    answer: str
    question: str
    sources: list[SearchResult]
    model: str
    latency_seconds: float
    context_chunks_used: int


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    notes: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    passed: bool
    matched_keywords: list[str]
    latency_seconds: float
    top_score: float
    sources: list[SearchResult] = field(default_factory=list)
