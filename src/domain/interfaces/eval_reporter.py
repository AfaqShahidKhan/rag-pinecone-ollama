"""
src/domain/interfaces/eval_reporter.py

Port for rendering evaluation diagnostics and results. Keeps
EvaluationService free of any presentation-library dependency (e.g. rich).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import EvalResult, PromptPackage, SearchResult


class IEvalReporter(ABC):
    @abstractmethod
    def report_debug_query(
        self, question: str, results: list[SearchResult], package: PromptPackage
    ) -> None:
        """Render a single-query diagnostic trace (retrieved chunks + prompt)."""
        ...

    @abstractmethod
    def report_case_result(self, index: int, total: int, result: EvalResult) -> None:
        """Render the outcome of a single evaluation case as it completes."""
        ...

    @abstractmethod
    def report_summary(self, results: list[EvalResult]) -> None:
        """Render the final pass/fail summary table for a full eval run."""
        ...
