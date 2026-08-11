"""
src/domain/errors.py

Central error-code registry for the whole pipeline. Every rejection or
failure that should be traceable in logs carries one of these codes,
category-prefixed so log files are greppable by subsystem:

    FILE-xxx    file-system level checks, before a file is even opened
    LOAD-xxx    document loader failures
    VALID-xxx   post-load content validation
    EXT-xxx     table / image extraction
    CONV-xxx    format conversion (LibreOffice .ppt -> .pptx)
    PII-xxx     PII redaction
    EMBED-xxx   embedding generation
    STORE-xxx   vector store / relational store
    SYS-xxx     generic / unexpected

Add new codes here as new failure modes are identified — never invent an
ad-hoc code inline at the call site, so this file stays the single source
of truth for what codes exist and what they mean.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    # FILE-xxx — file-system level, checked before a file is loaded
    FILE_NOT_READONLY = "FILE-001"
    FILE_NOT_FOUND = "FILE-002"
    FILE_EMPTY = "FILE-003"

    # LOAD-xxx — document loader failures
    LOAD_UNSUPPORTED_EXTENSION = "LOAD-001"
    LOAD_FAILED = "LOAD-002"

    # VALID-xxx — post-load content validation
    VALID_EMPTY_CONTENT = "VALID-001"
    VALID_ENCODING_ERROR = "VALID-002"
    VALID_CORRUPT_FILE = "VALID-003"

    # EXT-xxx — table / image extraction
    EXT_TABLE_FAILED = "EXT-001"
    EXT_IMAGE_FAILED = "EXT-002"

    # CONV-xxx — format conversion (LibreOffice)
    CONV_LIBREOFFICE_FAILED = "CONV-001"

    # PII-xxx — PII redaction
    PII_REDACTION_FAILED = "PII-001"

    # EMBED-xxx — embedding generation
    EMBED_FAILED = "EMBED-001"

    # STORE-xxx — vector store / relational store
    STORE_UPSERT_FAILED = "STORE-001"
    STORE_CONNECTION_FAILED = "STORE-002"

    # SYS-xxx — generic / unexpected
    SYS_UNEXPECTED = "SYS-001"


def format_error(code: ErrorCode, message: str) -> str:
    """Consistent 'CODE + message' formatting for every logged error."""
    return f"[{code.value}] {message}"


class PipelineError(Exception):
    """
    Raised by validators/gates when a file or its content fails a check.
    Callers catch this, log it via format_error(), and (in Step 2) route
    the file to unprocessed/<reason>/.
    """

    def __init__(self, code: ErrorCode, message: str, *, file_path: str | None = None) -> None:
        self.code = code
        self.message = message
        self.file_path = file_path
        super().__init__(format_error(code, message))