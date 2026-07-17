"""
src/infrastructure/loaders/ocr_loader.py

IDocumentLoader adapter for image files backed by pytesseract + Pillow.
Converts scanned images into text Documents via OCR.

Supported formats: .png, .jpg, .jpeg, .tiff, .tif, .bmp, .gif

System requirement:
  Tesseract OCR engine must be installed on the machine (not pip):
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
           Default install path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe

  After installing, either:
    a) Add Tesseract to your system PATH, OR
    b) Set TESSERACT_CMD in your .env file, e.g.:
       TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

Python packages:
  pip install pytesseract Pillow

OCR language defaults to English ('eng'). Set TESSERACT_LANG in .env
to use another language (e.g. 'urd' for Urdu, 'ara' for Arabic).
Multiple languages: TESSERACT_LANG=eng+urd
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
import pytesseract

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

# Tesseract config: page segmentation mode 3 = fully automatic
_TESSERACT_CONFIG = "--psm 3"


class OcrLoader(IDocumentLoader):
    def __init__(self, logger: ILogger, lang: str = "eng") -> None:
        """
        Args:
            lang: Tesseract language code(s). Default 'eng'.
                  Use '+' to combine: 'eng+urd'.
        """
        self._logger = logger
        self._lang = lang
        self._configure_tesseract()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> list[Document]:
        try:
            image = Image.open(str(path))
        except Exception as exc:
            self._logger.error(f"Failed to open image '{path.name}': {exc}")
            return []

        pages = self._extract_pages(image, path)

        if not pages:
            self._logger.warning(f"OCR on '{path.name}' yielded no text — skipping.")
            return []

        self._logger.info(f"OCR loaded '{path.name}': {len(pages)} frame(s).")
        return pages

    def _extract_pages(self, image: Image.Image, path: Path) -> list[Document]:
        """Handle both single-frame images and multi-frame GIF/TIFF."""
        docs: list[Document] = []

        try:
            total_frames = getattr(image, "n_frames", 1)
        except Exception:
            total_frames = 1

        for frame_index in range(total_frames):
            try:
                image.seek(frame_index)
            except EOFError:
                break

            frame = image.copy().convert("RGB")
            text = self._run_ocr(frame, path, frame_index)

            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":      path.name,
                        "source_path": str(path),
                        "page":        frame_index + 1,
                        "total_pages": total_frames,
                        "file_type":   "image",
                        "ocr_lang":    self._lang,
                    },
                ))

        return docs

    def _run_ocr(self, image: Image.Image, path: Path, frame_index: int) -> str:
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self._lang,
                config=_TESSERACT_CONFIG,
            )
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            self._logger.error(
                "Tesseract not found. Install from: "
                "https://github.com/UB-Mannheim/tesseract/wiki "
                "and set TESSERACT_CMD in your .env"
            )
            raise
        except Exception as exc:
            self._logger.warning(
                f"OCR failed on '{path.name}' frame {frame_index + 1}: {exc}"
            )
            return ""

    @staticmethod
    def _configure_tesseract() -> None:
        """Point pytesseract at the Tesseract binary if TESSERACT_CMD is set in env."""
        cmd = os.getenv("TESSERACT_CMD", "")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd