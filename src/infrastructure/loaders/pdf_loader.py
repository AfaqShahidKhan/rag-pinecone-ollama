"""
src/infrastructure/loaders/pdf_loader.py

IDocumentLoader adapter with configurable fallback chains for both text
and table extraction (see config/default.yml's document_loading.pdf
section). Each page runs the primary strategy first; if its confidence is
below the configured threshold (or it raises), the next strategy in the
chain is tried, down to a guaranteed-to-succeed last resort:
  text:  pypdf -> PyMuPDF -> OCR (rasterize + tesseract/easyocr/paddleocr)
  table: pdfplumber -> PyMuPDF tables -> give up (rely on plain text)

Every page where a fallback was actually needed is logged with LOAD-003
(see src.domain.errors) so it's easy to find in logs/exceptions.log which
pages needed extra help and which strategy ultimately won.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.config.settings import TableExtractionSettings
from src.domain.entities import Document
from src.domain.errors import ErrorCode, format_error
from src.domain.fallback_chain import run_fallback_chain
from src.domain.interfaces import (
    IDocumentLoader,
    ILogger,
    ITableExtractionStrategy,
    ITextExtractionStrategy,
)

SUPPORTED_EXTENSION = ".pdf"


class PdfDocumentLoader(IDocumentLoader):
    def __init__(
        self,
        logger: ILogger,
        table_extraction_settings: TableExtractionSettings,
        text_strategies: list[tuple[str, ITextExtractionStrategy]],
        table_strategies: list[tuple[str, ITableExtractionStrategy]],
        text_confidence_threshold: float,
        table_min_confidence: float,
    ) -> None:
        self._logger = logger
        self._table_settings = table_extraction_settings
        self._text_strategies = text_strategies
        self._table_strategies = table_strategies
        self._text_confidence_threshold = text_confidence_threshold
        self._table_min_confidence = table_min_confidence

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        total = self._page_count(path)
        docs: list[Document] = []
        fallback_pages = 0
        table_pages = 0

        for i in range(total):
            text_result = run_fallback_chain(
                [
                    (name, lambda s=strategy: s.extract_page(path, i, total))
                    for name, strategy in self._text_strategies
                ],
                confidence_threshold=self._text_confidence_threshold,
            )
            if text_result.fell_back:
                fallback_pages += 1
                self._logger.warning(format_error(
                    ErrorCode.LOAD_FALLBACK_USED,
                    f"'{path.name}' page {i + 1}: text extraction fell back to "
                    f"'{text_result.attempt.strategy_name}' (tried {text_result.attempts_tried}).",
                ))

            table_md = ""
            if self._table_settings.enabled:
                table_result = run_fallback_chain(
                    [
                        (name, lambda s=strategy: s.extract_page(path, i, total))
                        for name, strategy in self._table_strategies
                    ],
                    confidence_threshold=self._table_min_confidence,
                )
                table_md = table_result.attempt.content
                if table_md:
                    table_pages += 1

            text = text_result.attempt.content.strip()
            if not text and not table_md:
                self._logger.warning(
                    f"Page {i + 1}/{total} of '{path.name}' yielded no text or tables — skipping."
                )
                continue

            combined = "\n\n".join(part for part in [text, table_md] if part)

            docs.append(Document(
                page_content=combined,
                metadata={
                    "source": path.name,
                    "source_path": str(path),
                    "page": i + 1,
                    "total_pages": total,
                    "file_type": "pdf",
                    "table_count": len(table_md.split("\n\n")) if table_md else 0,
                    "text_extraction_method": text_result.attempt.strategy_name,
                    "text_extraction_confidence": text_result.attempt.confidence,
                },
            ))

        self._logger.info(
            f"Loaded '{path.name}': {len(docs)}/{total} pages with content "
            f"({table_pages} page(s) with tables, {fallback_pages} page(s) needed a fallback extractor)."
        )
        return docs

    @staticmethod
    def _page_count(path: Path) -> int:
        return len(PdfReader(str(path)).pages)