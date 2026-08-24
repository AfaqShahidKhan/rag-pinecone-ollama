"""
src/infrastructure/ocr/tesseract_ocr_engine.py

Primary OCR engine (moved from OcrLoader's inline logic — same behavior,
now reusable as a strategy in a fallback chain).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytesseract
from PIL import Image

from src.domain.confidence import score_text_confidence
from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import IOcrEngine

_TESSERACT_CONFIG = "--psm 3"


class TesseractOcrEngine(IOcrEngine):
    name = "tesseract"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        self._lang = "+".join(_map_lang(l) for l in languages) if languages else "eng"
        cmd = os.getenv("TESSERACT_CMD", "")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def recognize(self, image_path: Path) -> ExtractionAttempt:
        image = Image.open(str(image_path))
        text = pytesseract.image_to_string(image, lang=self._lang, config=_TESSERACT_CONFIG).strip()
        return ExtractionAttempt(
            content=text,
            confidence=score_text_confidence(text),
            strategy_name=self.name,
        )


def _map_lang(code: str) -> str:
    """Map short ISO codes (from config.yaml's [en, ar, zh, es]) to Tesseract's language codes."""
    mapping = {"en": "eng", "ar": "ara", "zh": "chi_sim", "es": "spa"}
    return mapping.get(code, code)