"""
src/domain/interfaces/validation.py

Ports for pre-ingestion file/content validation and quarantine of rejected
files. New checks are added by implementing IFileValidator or
IContentValidator — never by editing an existing validator to do two things.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import Document


class IFileValidator(ABC):
    @abstractmethod
    def validate(self, path: Path) -> None:
        """Raises PipelineError (src.domain.errors) if this file fails the check."""
        ...


class IContentValidator(ABC):
    @abstractmethod
    def validate(self, path: Path, documents: list[Document]) -> None:
        """Raises PipelineError (src.domain.errors) if the loaded content fails the check."""
        ...


class IUnprocessedFileMover(ABC):
    @abstractmethod
    def move(self, path: Path, reason: str) -> Path:
        """Moves path into <unprocessed_dir>/<reason>/, returns the new path."""
        ...