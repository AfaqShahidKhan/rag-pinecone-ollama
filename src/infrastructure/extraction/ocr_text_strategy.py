"""
src/infrastructure/extraction/ocr_text_strategy.py

Last-resort PDF text extraction: rasterizes the page to an image, then
runs it through the configured OCR engine chain (tesseract -> easyocr ->
paddleocr, or whatever is configured). Only invoked when both pypdf and
PyMuPDF failed to meet the confidence threshold — this is by far the
slowest strategy in the chain, so it should rarely run in practice.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz  # PyMuPDF — used here only for rasterization, not text extraction

from src.domain.entities import ExtractionAttempt
from src.domain.errors import ErrorCode, format_error
from src.domain.fallback_chain import run_fallback_chain
from src.domain.interfaces import ILogger, IOcrEngine, ITextExtractionStrategy


class OcrTextExtractionStrategy(ITextExtractionStrategy):
    name = "tesseract_ocr"  # matches the config key; the actual winning engine is recorded separately

    def __init__(
        self,
        ocr_engines: list[tuple[str, IOcrEngine]],
        ocr_confidence_threshold: float,
        logger: ILogger,
    ) -> None:
        self._ocr_engines = ocr_engines
        self._threshold = ocr_confidence_threshold
        self._logger = logger
        self._cached_path: Path | None = None
        self._cached_doc: fitz.Document | None = None

    def _doc_for(self, path: Path) -> fitz.Document:
        if self._cached_path != path:
            if self._cached_doc is not None:
                self._cached_doc.close()
            self._cached_doc = fitz.open(str(path))
            self._cached_path = path
        return self._cached_doc

    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        doc = self._doc_for(path)
        page = doc[page_index]
        pix = page.get_pixmap(dpi=200)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "page.png"
            pix.save(str(image_path))

            result = run_fallback_chain(
                [(name, lambda e=engine: e.recognize(image_path)) for name, engine in self._ocr_engines],
                confidence_threshold=self._threshold,
            )

        if result.fell_back:
            self._logger.warning(format_error(
                ErrorCode.LOAD_FALLBACK_USED,
                f"OCR fallback on '{path.name}' page {page_index + 1}: used "
                f"'{result.attempt.strategy_name}' (tried {result.attempts_tried}).",
            ))

        return ExtractionAttempt(
            content=result.attempt.content,
            confidence=result.attempt.confidence,
            strategy_name=f"ocr:{result.attempt.strategy_name}",
        )