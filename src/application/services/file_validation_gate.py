"""
src/application/services/file_validation_gate.py

Orchestrates file-level and content-level validation before a document is
processed. Any failure is logged with its error code (via
src.domain.errors.format_error — lands in logs/exceptions.log) and the
offending file is moved to data/unprocessed/<reason>/.

Used identically by IngestionService, StreamingIngestionService, and
CorpusBuilderService, so "read-only files are skipped" and "empty/corrupt
files are quarantined" behave the same everywhere: batch ingest,
corpus-only builds, and the landing-zone watcher.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.errors import ErrorCode, PipelineError, format_error
from src.domain.interfaces import IContentValidator, IFileValidator, ILogger, IUnprocessedFileMover

_REASON_BY_CODE: dict[ErrorCode, str] = {
    ErrorCode.FILE_NOT_READONLY: "writable",
    ErrorCode.FILE_EMPTY: "empty",
    ErrorCode.LOAD_FAILED: "load_failed",
    ErrorCode.VALID_EMPTY_CONTENT: "invalid_content",
    ErrorCode.VALID_ENCODING_ERROR: "invalid_content",
    ErrorCode.VALID_CORRUPT_FILE: "load_failed",
}


class FileValidationGate:
    def __init__(
        self,
        file_validators: list[IFileValidator],
        content_validators: list[IContentValidator],
        mover: IUnprocessedFileMover,
        logger: ILogger,
    ) -> None:
        self._file_validators = file_validators
        self._content_validators = content_validators
        self._mover = mover
        self._logger = logger

    def check_file(self, path: Path) -> bool:
        """Runs file-level checks (read-only, not-empty) BEFORE loading. False = rejected."""
        for validator in self._file_validators:
            try:
                validator.validate(path)
            except PipelineError as exc:
                self._reject(path, exc.code, exc.message)
                return False
        return True

    def check_content(self, path: Path, documents: list[Document]) -> bool:
        """Runs content-level checks AFTER loading. False = rejected."""
        for validator in self._content_validators:
            try:
                validator.validate(path, documents)
            except PipelineError as exc:
                self._reject(path, exc.code, exc.message)
                return False
        return True

    def reject_load_exception(self, path: Path, exc: Exception) -> None:
        """Call when loader.load() itself raises — treated as a corrupt/unreadable file."""
        self._reject(path, ErrorCode.LOAD_FAILED, str(exc))

    def _reject(self, path: Path, code: ErrorCode, message: str) -> None:
        self._logger.error(format_error(code, f"{message} (file: '{path.name}')"))
        reason = _REASON_BY_CODE.get(code, "other")
        self._mover.move(path, reason)