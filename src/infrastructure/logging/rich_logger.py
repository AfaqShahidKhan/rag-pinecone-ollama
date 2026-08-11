"""
src/infrastructure/logging/rich_logger.py

Concrete ILogger implementation backed by the `rich` + stdlib `logging`
libraries. This is the only file in the project allowed to configure
`logging.basicConfig` / instantiate `RichHandler` / `RotatingFileHandler`.

Three handlers sit on the root logger:
  - RichHandler:        terminal output.
  - RotatingFileHandler: full detail (down to file_level) -> logs/rag.log.
  - RotatingFileHandler: ERROR-and-above ONLY -> logs/exceptions.log — a
    focused, short file for "what broke", separate from the full-detail
    main log. Callers should format error messages via
    src.domain.errors.format_error(code, message) so both the code and the
    message land in this file together.

Every RichLogger(name) is a child of the root logger, so all three handlers
apply automatically to every existing call site — no other file needed to
change.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config.settings import LoggingSettings
from src.domain.interfaces import ILogger


class RichLogger(ILogger):
    """Adapter around Python's stdlib logging, rendered via rich + persisted to disk."""

    _configured: bool = False

    def __init__(self, name: str, logging_settings: LoggingSettings | None = None) -> None:
        self._ensure_configured(logging_settings or LoggingSettings())
        self._logger = logging.getLogger(name)

    @classmethod
    def _ensure_configured(cls, settings: LoggingSettings) -> None:
        if cls._configured:
            return

        console_level = cls._resolve_level(settings.console_level, logging.INFO)
        file_level = cls._resolve_level(settings.file_level, logging.DEBUG)

        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=True,
        )
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        main_file_handler = RotatingFileHandler(
            filename=log_dir / settings.filename,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
        main_file_handler.setLevel(file_level)
        main_file_handler.setFormatter(file_formatter)

        exceptions_handler = RotatingFileHandler(
            filename=log_dir / settings.exceptions_filename,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
        exceptions_handler.setLevel(logging.ERROR)  # ONLY errors/exceptions land here
        exceptions_handler.setFormatter(file_formatter)

        root = logging.getLogger()
        root.setLevel(min(console_level, file_level))
        root.handlers.clear()
        root.addHandler(console_handler)
        root.addHandler(main_file_handler)
        root.addHandler(exceptions_handler)

        cls._configured = True

    @staticmethod
    def _resolve_level(level_name: str, default: int) -> int:
        return getattr(logging, level_name.upper(), default)

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)