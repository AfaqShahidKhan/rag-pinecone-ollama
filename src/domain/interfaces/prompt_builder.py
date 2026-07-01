"""
src/domain/interfaces/prompt_builder.py

Port for assembling retrieved search results + a question into a PromptPackage
ready to send to an answer generator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import PromptPackage, SearchResult


class IPromptBuilder(ABC):
    @abstractmethod
    def build(self, question: str, results: list[SearchResult]) -> PromptPackage:
        ...
