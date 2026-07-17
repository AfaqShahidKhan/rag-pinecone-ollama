"""
src/infrastructure/chunking/semantic_chunker.py

ITextChunker adapter that splits documents by measuring cosine similarity
between adjacent sentence embeddings. A split is inserted wherever the
similarity drops below a configurable threshold — meaning the topic has
shifted enough to warrant a new chunk.

Advantages over RecursiveCharacterTextSplitter:
  - Chunks respect semantic boundaries (topic shifts), not arbitrary char counts.
  - Context continuity within a chunk is maximised.

Trade-offs:
  - Slower: requires one embedding pass per document before chunking.
  - Recommended for shorter, structured content (HTML, JSON).
    Use RecursiveTextChunker for long, dense PDFs via DocumentRouter.

Dependencies:
  - nltk (sentence tokenisation) — pip install nltk
  - IEmbeddingProvider (injected — no direct SDK import here)

NLTK data:
  punkt tokeniser is downloaded on first use (quiet, cached afterwards).
"""

from __future__ import annotations

import math

import nltk

from src.config.settings import SemanticChunkingSettings
from src.domain.entities import Document
from src.domain.interfaces import IEmbeddingProvider, ILogger, ITextChunker

# Download punkt data silently if not already cached
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class SemanticChunker(ITextChunker):
    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        logger: ILogger,
        semantic_settings: SemanticChunkingSettings,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._logger = logger
        self._settings = semantic_settings

    def chunk(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []

        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)

        self._logger.info(
            f"SemanticChunker: {len(documents)} documents → {len(all_chunks)} chunks "
            f"(threshold={self._settings.similarity_threshold})"
        )
        return all_chunks

    def _chunk_document(self, doc: Document) -> list[Document]:
        sentences = nltk.sent_tokenize(doc.page_content)

        # Too few sentences to split — return as single chunk
        if len(sentences) <= self._settings.min_sentences_per_chunk:
            return [Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "chunk_index": 0, "chunk_total": 1},
            )]

        embeddings = self._embedding_provider.embed_texts(sentences)
        breakpoints = self._find_breakpoints(sentences, embeddings)
        groups = self._group_sentences(sentences, breakpoints)

        return [
            Document(
                page_content=group,
                metadata={**doc.metadata, "chunk_index": i, "chunk_total": len(groups)},
            )
            for i, group in enumerate(groups)
        ]

    def _find_breakpoints(
        self, sentences: list[str], embeddings: list[list[float]]
    ) -> list[int]:
        """Return indices where a new chunk should start (1-based sentence index)."""
        breakpoints: list[int] = []
        current_chunk_size = 0

        for i in range(len(sentences) - 1):
            current_chunk_size += 1
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            should_split = (
                sim < self._settings.similarity_threshold
                and current_chunk_size >= self._settings.min_sentences_per_chunk
            ) or current_chunk_size >= self._settings.max_sentences_per_chunk

            if should_split:
                breakpoints.append(i + 1)
                current_chunk_size = 0

        return breakpoints

    def _group_sentences(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[str]:
        """Assemble sentence groups from split indices. Merges tiny tail groups."""
        groups: list[str] = []
        start = 0

        for bp in breakpoints:
            groups.append(" ".join(sentences[start:bp]))
            start = bp

        tail = sentences[start:]
        if tail:
            if groups and len(tail) < self._settings.min_sentences_per_chunk:
                # Merge tiny tail into the previous group
                groups[-1] += " " + " ".join(tail)
            else:
                groups.append(" ".join(tail))

        return groups if groups else [" ".join(sentences)]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Pure-Python cosine similarity — avoids numpy dependency."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)