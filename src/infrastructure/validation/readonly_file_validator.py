"""
src/infrastructure/validation/readonly_file_validator.py

Rejects files that are still writable — a writable file may still be
mid-copy or mid-write by whatever produced it, so processing it risks
reading a half-finished file. Files are expected to be marked read-only
once they're finished and ready for ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.domain.errors import ErrorCode, PipelineError
from src.domain.interfaces import IFileValidator


class ReadOnlyFileValidator(IFileValidator):
    def validate(self, path: Path) -> None:
        if os.access(path, os.W_OK):
            raise PipelineError(
                ErrorCode.FILE_NOT_READONLY,
                "File is writable, not read-only — it may still be mid-copy/mid-write.",
                file_path=str(path),
            )