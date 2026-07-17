"""
src/domain/interfaces/pre_processor.py

Two ports for the pre-processing layer:

  IPreProcessor      — transforms a single Document (one responsibility per class).
  IDocumentProcessor — processes a full batch of Documents (what IngestionService
                       depends on; implemented by PreProcessingPipeline).

Keeping them separate follows ISP: individual processors never need to know
they are part of a pipeline, and the pipeline never needs to implement
single-document logic itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import Document


class IPreProcessor(ABC):
    """Transforms a single Document — clean, normalize, validate, or enrich."""

    @abstractmethod
    def process(self, document: Document) -> Document:
        """
        Return a transformed copy of the document.
        Raise ValueError if the document is so malformed it must be dropped.
        """
        ...


class IDocumentProcessor(ABC):
    """Processes a full batch of Documents, returning a cleaned/filtered batch."""

    @abstractmethod
    def process_all(self, documents: list[Document]) -> list[Document]:
        """
        Apply all registered processors to every document in the list.
        Documents that fail validation are logged and excluded from the result.
        """
        ...