"""
src/domain/interfaces/logger.py

Port for logging. Concrete implementations live in infrastructure/logging/.
Nothing outside that folder may import a logging SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ILogger(ABC):
    @abstractmethod
    def debug(self, message: str) -> None: ...

    @abstractmethod
    def info(self, message: str) -> None: ...

    @abstractmethod
    def warning(self, message: str) -> None: ...

    @abstractmethod
    def error(self, message: str) -> None: ...
