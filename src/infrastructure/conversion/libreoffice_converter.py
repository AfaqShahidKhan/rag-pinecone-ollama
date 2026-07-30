"""
src/infrastructure/conversion/libreoffice_converter.py

Converts legacy Office formats (.ppt, and in principle .doc/.xls too) to
their modern OOXML equivalents via LibreOffice headless mode, so existing
loaders (PptxDocumentLoader, DocxDocumentLoader, ...) can read them
completely unmodified.

Requires LibreOffice installed on the machine running ingestion — this is
a system dependency, not a Python package. Nothing here imports it as a
library; it's invoked as a subprocess (`soffice --headless --convert-to`).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from src.config.settings import LibreOfficeSettings
from src.domain.interfaces import ILogger


class LibreOfficeConversionError(RuntimeError):
    pass


class LibreOfficeConverter:
    def __init__(self, logger: ILogger, settings: LibreOfficeSettings) -> None:
        self._logger = logger
        self._settings = settings

    def convert(self, source: Path, target_extension: str) -> Path:
        """
        Converts `source` to `target_extension` (e.g. "pptx") using
        LibreOffice headless, writing into a fresh temp directory. Returns
        the path to the converted file. The caller owns cleanup of that
        temp directory (see PptDocumentLoader).
        """
        out_dir = Path(tempfile.mkdtemp(prefix="rag_libreoffice_"))
        command = [
            self._settings.executable_path,
            "--headless",
            "--convert-to", target_extension,
            "--outdir", str(out_dir),
            str(source),
        ]

        self._logger.info(f"LibreOfficeConverter: converting '{source.name}' -> .{target_extension} ...")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._settings.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise LibreOfficeConversionError(
                f"LibreOffice executable not found at '{self._settings.executable_path}'. "
                f"Install LibreOffice (https://www.libreoffice.org/download/) or set "
                f"LIBREOFFICE_PATH in .env to the full path of soffice/soffice.exe."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LibreOfficeConversionError(
                f"LibreOffice conversion of '{source.name}' timed out after "
                f"{self._settings.timeout_seconds}s."
            ) from exc

        if result.returncode != 0:
            raise LibreOfficeConversionError(
                f"LibreOffice failed to convert '{source.name}' (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        converted_path = out_dir / f"{source.stem}.{target_extension}"
        if not converted_path.exists():
            raise LibreOfficeConversionError(
                f"LibreOffice reported success but expected output "
                f"'{converted_path}' was not found."
            )

        self._logger.info(f"LibreOfficeConverter: '{source.name}' converted successfully.")
        return converted_path