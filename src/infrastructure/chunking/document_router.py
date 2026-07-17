"""
src/infrastructure/chunking/document_router.py

ITextChunker composite that implements the Strategy + Router patterns.

Instead of a single chunking algorithm for all document types, DocumentRouter
holds a registry of (file_types → ITextChunker) mappings and routes each
Document to the appropriate strategy based on its metadata["file_type"].

Default routing:
  html, json  → SemanticChunker  (structured, shorter content benefits from
                                  semantic boundary detection)
  pdf, docx,
  image, *    → RecursiveTextChunker (fast, reliable for dense / long docs)

Why this matters:
  - A 300-page annual report PDF should NOT use SemanticChunker (too slow,
    too many embedding API calls during chunking).
  - An HTML knowledge-base article SHOULD use SemanticChunker (topic shifts
    are meaningful; semantic boundaries improve retrieval precision).

DocumentRouter IS an ITextChunker so IngestionService needs zero changes.
The factory swaps in DocumentRouter wherever RecursiveTextChunker was used.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.entities import Document
from src.domain.interfaces import ILogger, ITextChunker


@dataclass
class ChunkingRoute:
    """Maps a set of file_type values to a named chunking strategy."""
    name: str
    file_types: frozenset[str]
    chunker: ITextChunker


class DocumentRouter(ITextChunker):
    def __init__(
        self,
        routes: list[ChunkingRoute],
        default_chunker: ITextChunker,
        logger: ILogger,
    ) -> None:
        """
        Args:
            routes:          Ordered list of ChunkingRoute rules. First match wins.
            default_chunker: Fallback chunker for file types not covered by any route.
            logger:          Injected logger.
        """
        self._routes = routes
        self._default_chunker = default_chunker
        self._logger = logger

    def chunk(self, documents: list[Document]) -> list[Document]:
        # Group documents by the chunker they map to
        groups: dict[str, tuple[ITextChunker, list[Document]]] = {}

        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown").lower()
            route_name, chunker = self._resolve(file_type)

            if route_name not in groups:
                groups[route_name] = (chunker, [])
            groups[route_name][1].append(doc)

        # Apply each chunker to its group and collect results
        all_chunks: list[Document] = []
        for route_name, (chunker, group_docs) in groups.items():
            self._logger.info(
                f"DocumentRouter: routing {len(group_docs)} document(s) "
                f"[file_type={self._get_types(group_docs)}] → '{route_name}'"
            )
            all_chunks.extend(chunker.chunk(group_docs))

        return all_chunks

    def _resolve(self, file_type: str) -> tuple[str, ITextChunker]:
        for route in self._routes:
            if file_type in route.file_types:
                return route.name, route.chunker
        return "default", self._default_chunker

    @staticmethod
    def _get_types(docs: list[Document]) -> str:
        types = sorted({d.metadata.get("file_type", "?") for d in docs})
        return ", ".join(types)