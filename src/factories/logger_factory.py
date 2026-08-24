"""
src/factories/logger_factory.py

Factory for ILogger instances. Swapping logging backends (rich -> structlog,
json logging, etc.) means changing only this file.

Instance-based (not a bare @staticmethod anymore) so LoggingSettings can be
injected once at composition time, then every ILogger created through this
factory shares the same file/console configuration.
"""

from __future__ import annotations

from src.config.settings import LoggingSettings
from src.domain.interfaces import ILogger
from src.infrastructure.logging import RichLogger


class LoggerFactory:
    def __init__(self, logging_settings: LoggingSettings) -> None:
        self._settings = logging_settings

    def create(self, name: str) -> ILogger:
        return RichLogger(name, logging_settings=self._settings)