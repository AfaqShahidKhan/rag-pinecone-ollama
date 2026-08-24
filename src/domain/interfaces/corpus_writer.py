"""
src/domain/interfaces/corpus_writer.py

Port for persisting pre-processed Documents as a human-readable, inspectable
corpus on disk (e.g. Markdown with YAML front-matter). This lets a user open
a plain-text folder and see exactly what will be chunked and embedded —
independent of chunk size, chunking strategy, or vector store choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import Document


class ICorpusWriter(ABC):
    @abstractmethod
    def write(self, documents: list[Document]) -> int:
        """
        Persist each Document to the corpus.
        Returns the number of files written.
        """
        ...