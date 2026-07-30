"""
src/infrastructure/images/pdf_image_extractor.py

IImageExtractor adapter for PDF files, backed by pypdf's built-in
page.images accessor — no extra dependency beyond pypdf, already a
project dependency.

Saves each image to:
    {output_dir}/{sanitized_source_stem}/img_{page:03d}_{index:03d}.{ext}

and attaches an "images" list to the metadata of the Document whose "page"
matches — PDF page numbers are exact, so this mapping is reliable.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from src.config.settings import ImageExtractionSettings
from src.domain.entities import Document
from src.domain.interfaces import IImageExtractor, ILogger

SUPPORTED_EXTENSION = ".pdf"
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_\-]+")


class PdfImageExtractor(IImageExtractor):
    def __init__(self, logger: ILogger, settings: ImageExtractionSettings) -> None:
        self._logger = logger
        self._settings = settings

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def extract(self, path: Path, documents: list[Document]) -> list[Document]:
        reader = PdfReader(str(path))
        stem = self._sanitize(path.stem)
        out_dir = Path(self._settings.output_dir) / stem
        by_page: dict[int, list[dict]] = {}
        saved = 0

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            try:
                page_images = list(page.images)
            except Exception as exc:
                self._logger.warning(
                    f"PdfImageExtractor: failed reading images on page {page_number} "
                    f"of '{path.name}': {exc}"
                )
                continue

            if not page_images:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            for img_index, image in enumerate(page_images):
                ext = Path(image.name).suffix or ".png"
                filename = f"img_{page_number:03d}_{img_index:03d}{ext}"
                dest = out_dir / filename
                dest.write_bytes(image.data)
                by_page.setdefault(page_number, []).append({
                    "path": str(dest),
                    "page": page_number,
                    "index": img_index,
                })
                saved += 1

        if saved:
            for doc in documents:
                if doc.metadata.get("source_path") != str(path):
                    continue
                images = by_page.get(doc.metadata.get("page"))
                if images:
                    doc.metadata["images"] = images

            self._logger.info(
                f"PdfImageExtractor: saved {saved} image(s) from '{path.name}' to '{out_dir}'."
            )

        return documents

    @staticmethod
    def _sanitize(name: str) -> str:
        cleaned = _UNSAFE_CHARS_RE.sub("_", name).strip("_")
        return cleaned or "unknown"