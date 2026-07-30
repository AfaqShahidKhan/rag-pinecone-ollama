"""
src/infrastructure/loaders/ppt_loader.py

IDocumentLoader adapter for legacy .ppt (PowerPoint 97-2003) files.

python-pptx cannot read the old binary .ppt format at all — it's a
completely different file structure (OLE2/CFB), not just an older XML
schema. This loader converts .ppt -> .pptx via LibreOffice headless
(LibreOfficeConverter), then delegates entirely to PptxDocumentLoader —
so slide/table/notes extraction logic lives in exactly one place.

Requires LibreOffice installed on the machine running ingestion (see
LibreOfficeSettings) — a system dependency, not a pip package. If
conversion fails (LibreOffice missing, corrupt file, timeout), the file
is skipped with a warning rather than crashing the whole ingestion run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger
from src.infrastructure.conversion import LibreOfficeConversionError, LibreOfficeConverter
from src.infrastructure.loaders.pptx_loader import PptxDocumentLoader

SUPPORTED_EXTENSION = ".ppt"


class PptDocumentLoader(IDocumentLoader):
    def __init__(
        self,
        logger: ILogger,
        converter: LibreOfficeConverter,
        pptx_loader: PptxDocumentLoader,
    ) -> None:
        self._logger = logger
        self._converter = converter
        self._pptx_loader = pptx_loader

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        converted_dir: Path | None = None
        try:
            converted_path = self._converter.convert(path, target_extension="pptx")
            converted_dir = converted_path.parent
            docs = self._pptx_loader.load(converted_path)
        except LibreOfficeConversionError as exc:
            self._logger.warning(f"Skipping '{path.name}': {exc}")
            return []
        finally:
            if converted_dir and converted_dir.exists():
                shutil.rmtree(converted_dir, ignore_errors=True)

        # Re-point metadata at the original .ppt file so downstream steps
        # (corpus writer, relational store) reference what the user
        # actually has on disk, not the throwaway converted copy.
        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["source_path"] = str(path)
            doc.metadata["file_type"] = "pptx"
            doc.metadata["converted_from"] = "ppt"

        self._logger.info(
            f"Loaded '{path.name}' via LibreOffice conversion: {len(docs)} slide(s)."
        )
        return docs