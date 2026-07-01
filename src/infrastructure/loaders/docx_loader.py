"""
src/infrastructure/loaders/docx_loader.py

IDocumentLoader adapter backed by python-docx. Paragraphs are grouped into
pseudo-pages (DOCX has no native page breaks) to approximate page-level
provenance, sized via IngestionSettings.docx_pseudo_page_chars.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxFile

from src.config.settings import IngestionSettings
from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSION = ".docx"


class DocxDocumentLoader(IDocumentLoader):
    def __init__(self, logger: ILogger, ingestion_settings: IngestionSettings) -> None:
        self._logger = logger
        self._settings = ingestion_settings

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        docx_file = DocxFile(str(path))
        paragraphs = [p.text.strip() for p in docx_file.paragraphs if p.text.strip()]

        if not paragraphs:
            self._logger.warning(f"'{path.name}' yielded no text — skipping.")
            return []

        pages = self._group_into_pseudo_pages(paragraphs)

        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "page": i + 1,
                    "total_pages": len(pages),
                    "file_type": "docx",
                },
            )
            for i, text in enumerate(pages)
        ]

        self._logger.info(
            f"Loaded '{path.name}': {len(docs)} sections from {len(paragraphs)} paragraphs."
        )
        return docs

    def _group_into_pseudo_pages(self, paragraphs: list[str]) -> list[str]:
        page_size = self._settings.docx_pseudo_page_chars
        pages: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > page_size and current:
                pages.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            pages.append("\n\n".join(current))

        return pages
