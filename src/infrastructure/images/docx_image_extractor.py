"""
src/infrastructure/images/docx_image_extractor.py

IImageExtractor adapter for DOCX files, backed by python-docx's package
relationships (document.part.rels), which is how embedded images are
addressable regardless of where they're anchored in the document body.

DOCX has no real page concept (DocxDocumentLoader only builds pseudo-pages
by character count), so images can't be reliably mapped to one specific
page the way PDF pages can. Every extracted image is instead attached to
metadata["source_images"] on every Document from that file — document-wide,
not page-specific.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document as DocxFile

from src.config.settings import ImageExtractionSettings
from src.domain.entities import Document
from src.domain.interfaces import IImageExtractor, ILogger

SUPPORTED_EXTENSION = ".docx"
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_\-]+")


class DocxImageExtractor(IImageExtractor):
    def __init__(self, logger: ILogger, settings: ImageExtractionSettings) -> None:
        self._logger = logger
        self._settings = settings

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def extract(self, path: Path, documents: list[Document]) -> list[Document]:
        docx_file = DocxFile(str(path))
        stem = self._sanitize(path.stem)
        out_dir = Path(self._settings.output_dir) / stem

        image_parts = [
            rel.target_part
            for rel in docx_file.part.rels.values()
            if "image" in rel.reltype
        ]
        if not image_parts:
            return documents

        out_dir.mkdir(parents=True, exist_ok=True)
        saved_images: list[dict] = []
        for index, part in enumerate(image_parts):
            ext = Path(part.partname).suffix or ".png"
            filename = f"img_{index:03d}{ext}"
            dest = out_dir / filename
            dest.write_bytes(part.blob)
            saved_images.append({"path": str(dest), "page": None, "index": index})

        for doc in documents:
            if doc.metadata.get("source_path") == str(path):
                doc.metadata["source_images"] = saved_images

        self._logger.info(
            f"DocxImageExtractor: saved {len(saved_images)} image(s) from "
            f"'{path.name}' to '{out_dir}'."
        )
        return documents

    @staticmethod
    def _sanitize(name: str) -> str:
        cleaned = _UNSAFE_CHARS_RE.sub("_", name).strip("_")
        return cleaned or "unknown"