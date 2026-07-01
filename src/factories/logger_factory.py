"""
src/factories/logger_factory.py

Factory for ILogger instances. Swapping logging backends (rich -> structlog,
json logging, etc.) means changing only this file.
"""

from __future__ import annotations

from src.domain.interfaces import ILogger
from src.infrastructure.logging import RichLogger


class LoggerFactory:
    @staticmethod
    def create(name: str) -> ILogger:
        return RichLogger(name)
