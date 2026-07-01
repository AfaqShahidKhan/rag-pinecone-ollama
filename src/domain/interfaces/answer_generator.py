"""
src/domain/interfaces/answer_generator.py

Port for turning a PromptPackage into a final RAGResponse via an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import PromptPackage, RAGResponse


class IAnswerGenerator(ABC):
    @abstractmethod
    def generate(self, package: PromptPackage, stream: bool = False) -> RAGResponse:
        ...
