"""
src/infrastructure/validation/file_not_empty_validator.py

Rejects zero-byte files before they ever reach a loader.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.errors import ErrorCode, PipelineError
from src.domain.interfaces import IFileValidator


class FileNotEmptyValidator(IFileValidator):
    def validate(self, path: Path) -> None:
        if path.stat().st_size == 0:
            raise PipelineError(
                ErrorCode.FILE_EMPTY,
                "File is 0 bytes.",
                file_path=str(path),
            )