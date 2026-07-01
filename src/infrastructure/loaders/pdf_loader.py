"""
src/infrastructure/loaders/pdf_loader.py

IDocumentLoader adapter backed by pypdf. One page on disk -> one Document.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSION = ".pdf"


class PdfDocumentLoader(IDocumentLoader):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        total = len(reader.pages)
        docs: list[Document] = []

        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()

            if not text:
                self._logger.warning(
                    f"Page {i + 1}/{total} of '{path.name}' yielded no text — skipping."
                )
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "page": i + 1,
                    "total_pages": total,
                    "file_type": "pdf",
                },
            ))

        self._logger.info(f"Loaded '{path.name}': {len(docs)}/{total} pages with text.")
        return docs
