"""
src/infrastructure/ocr/easyocr_engine.py

First OCR fallback. The Reader (which loads detection/recognition models)
is created lazily on first use, not in __init__ — this strategy is rarely
invoked (only when Tesseract's confidence is too low), so paying the
model-load cost only when actually needed matters here.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.confidence import score_text_confidence
from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import IOcrEngine


class EasyOcrEngine(IOcrEngine):
    name = "easyocr"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        self._languages = list(languages) if languages else ["en"]
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=False)
        return self._reader

    def recognize(self, image_path: Path) -> ExtractionAttempt:
        reader = self._get_reader()
        results = reader.readtext(str(image_path), detail=0)
        text = "\n".join(results).strip()
        return ExtractionAttempt(
            content=text,
            confidence=score_text_confidence(text),
            strategy_name=self.name,
        )