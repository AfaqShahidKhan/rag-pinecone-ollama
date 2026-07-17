"""
src/infrastructure/loaders/docx_loader.py

IDocumentLoader adapter backed by python-docx.

Phase 3 update: DOCX tables are now extracted and converted to Markdown
tables. Previously only paragraphs were read; tables (which appear
frequently in financial reports, policy docs, and structured content)
were silently skipped.

Approach:
  - Iterate document body elements in order (paragraphs AND tables)
  - Convert each <w:tbl> element to a Markdown table string
  - Merge tables and paragraphs into the same pseudo-page grouping flow
    so tabular content stays co-located with its surrounding prose

Paragraphs are grouped into pseudo-pages sized by IngestionSettings.
docx_pseudo_page_chars to approximate page-level provenance.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxFile
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

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
        blocks = self._extract_blocks(docx_file)

        if not blocks:
            self._logger.warning(f"'{path.name}' yielded no text — skipping.")
            return []

        pages = self._group_into_pseudo_pages(blocks)
        docs = [
            Document(
                page_content=text,
                metadata={
                    "source":      path.name,
                    "source_path": str(path),
                    "page":        i + 1,
                    "total_pages": len(pages),
                    "file_type":   "docx",
                },
            )
            for i, text in enumerate(pages)
        ]

        self._logger.info(
            f"Loaded '{path.name}': {len(docs)} section(s) "
            f"from {len(blocks)} blocks (paragraphs + tables)."
        )
        return docs

    def _extract_blocks(self, docx_file: DocxFile) -> list[str]:
        """
        Extract all text blocks (paragraphs and tables) in document order.
        Uses the raw XML body to preserve ordering between paragraphs and tables.
        """
        blocks: list[str] = []
        body = docx_file.element.body

        for child in body:
            tag = child.tag

            if tag == qn("w:p"):
                # Paragraph
                para = Paragraph(child, docx_file)
                text = para.text.strip()
                if text:
                    blocks.append(text)

            elif tag == qn("w:tbl"):
                # Table
                table = Table(child, docx_file)
                md = self._table_to_markdown(table)
                if md:
                    blocks.append(md)

        return blocks

    @staticmethod
    def _table_to_markdown(table: Table) -> str:
        """Convert a python-docx Table to a Markdown table string."""
        rows: list[str] = []
        for i, row in enumerate(table.rows):
            cells = [
                cell.text.strip().replace("|", "\\|").replace("\n", " ")
                for cell in row.cells
            ]
            if not any(cells):
                continue
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n".join(rows)

    def _group_into_pseudo_pages(self, blocks: list[str]) -> list[str]:
        page_size = self._settings.docx_pseudo_page_chars
        pages: list[str] = []
        current: list[str] = []
        current_len = 0

        for block in blocks:
            if current_len + len(block) > page_size and current:
                pages.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(block)
            current_len += len(block)

        if current:
            pages.append("\n\n".join(current))

        return pages