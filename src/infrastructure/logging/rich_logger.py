"""
src/infrastructure/logging/rich_logger.py

Concrete ILogger implementation backed by the `rich` + stdlib `logging`
libraries. This is the only file in the project allowed to configure
`logging.basicConfig` / instantiate `RichHandler`.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from src.domain.interfaces import ILogger


class RichLogger(ILogger):
    """Adapter around Python's stdlib logging, rendered via rich."""

    _configured: bool = False

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._ensure_configured(level)
        self._logger = logging.getLogger(name)

    @classmethod
    def _ensure_configured(cls, level: int) -> None:
        if cls._configured:
            return
        console = Console(stderr=True)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    tracebacks_show_locals=False,
                    show_path=True,
                )
            ],
        )
        cls._configured = True

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)
