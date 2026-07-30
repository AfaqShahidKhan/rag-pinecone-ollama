"""
src/infrastructure/loaders/pdf_loader.py

IDocumentLoader adapter backed by pypdf (text) + pdfplumber (tables).

Table extraction: pdfplumber detects and extracts tables per page far more
reliably than raw text parsing would. Detected tables are converted to
Markdown (same convention DocxDocumentLoader already uses) and appended
after the page's regular text — MetadataEnricher's existing has_tables
regex detects Markdown pipe syntax automatically, so no change was needed
there.

Table extraction is best-effort: a page that fails table detection still
returns its plain text via pypdf, and a failure never aborts the whole
document (logged as a warning instead).
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from src.config.settings import TableExtractionSettings
from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSION = ".pdf"


class PdfDocumentLoader(IDocumentLoader):
    def __init__(self, logger: ILogger, table_extraction_settings: TableExtractionSettings) -> None:
        self._logger = logger
        self._table_settings = table_extraction_settings

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        total = len(reader.pages)
        tables_by_page = self._extract_tables(path) if self._table_settings.enabled else {}
        docs: list[Document] = []

        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            page_tables = tables_by_page.get(i + 1, [])

            if not text and not page_tables:
                self._logger.warning(
                    f"Page {i + 1}/{total} of '{path.name}' yielded no text or tables — skipping."
                )
                continue

            combined = "\n\n".join(part for part in [text, *page_tables] if part)

            docs.append(Document(
                page_content=combined,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "page": i + 1,
                    "total_pages": total,
                    "file_type": "pdf",
                    "table_count": len(page_tables),
                },
            ))

        table_pages = sum(1 for v in tables_by_page.values() if v)
        self._logger.info(
            f"Loaded '{path.name}': {len(docs)}/{total} pages with content "
            f"({table_pages} page(s) contained tables)."
        )
        return docs

    def _extract_tables(self, path: Path) -> dict[int, list[str]]:
        """
        Returns {page_number: [markdown_table, ...]} for every page with at
        least one detectable table. Best-effort — pdfplumber's detection can
        miss borderless tables; that content still comes through as plain
        text via pypdf either way, so nothing is lost, just not structured.
        """
        tables_by_page: dict[int, list[str]] = {}
        try:
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    raw_tables = page.extract_tables()
                    if not raw_tables:
                        continue
                    markdown_tables = [
                        md for md in (self._table_to_markdown(t) for t in raw_tables) if md
                    ]
                    if markdown_tables:
                        tables_by_page[i + 1] = markdown_tables
        except Exception as exc:
            self._logger.warning(
                f"Table extraction failed for '{path.name}': {exc}. "
                f"Continuing with text-only content."
            )
        return tables_by_page

    @staticmethod
    def _table_to_markdown(rows: list[list[str | None]]) -> str:
        """Convert a pdfplumber extracted table (list of rows of cells) to Markdown."""
        cleaned_rows = [
            [
                str(cell).strip().replace("|", "\\|").replace("\n", " ") if cell else ""
                for cell in row
            ]
            for row in rows
        ]
        cleaned_rows = [row for row in cleaned_rows if any(cell for cell in row)]
        if not cleaned_rows:
            return ""

        lines: list[str] = []
        for i, row in enumerate(cleaned_rows):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        return "\n".join(lines)