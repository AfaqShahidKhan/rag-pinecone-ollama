"""
src/infrastructure/ocr/paddleocr_engine.py

Second OCR fallback. Written against PaddleOCR 3.x's API (a
non-backward-compatible rewrite from 2.x — use_angle_cls/show_log and
.ocr() are gone; .predict() + .save_to_json() is the current, documented
interface). Uses save_to_json() rather than reading result attributes
directly, since that's PaddleOCR's officially stable output format.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.domain.confidence import score_text_confidence
from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import IOcrEngine

_LANG_MAP = {"en": "en", "ar": "ar", "zh": "ch", "es": "es"}


class PaddleOcrEngine(IOcrEngine):
    name = "paddleocr"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        first = languages[0] if languages else "en"
        self._lang = _LANG_MAP.get(first, "en")
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(lang=self._lang)
        return self._engine

    def recognize(self, image_path: Path) -> ExtractionAttempt:
        engine = self._get_engine()
        results = engine.predict(str(image_path))

        lines: list[str] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for res in results:
                res.save_to_json(save_path=tmp_dir)
            for json_file in Path(tmp_dir).glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lines.extend(data.get("rec_texts", []))

        text = "\n".join(lines).strip()
        return ExtractionAttempt(
            content=text,
            confidence=score_text_confidence(text),
            strategy_name=self.name,
        )