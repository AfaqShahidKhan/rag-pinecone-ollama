"""
src/application/services/evaluation_service.py

Orchestrates retrieval debugging and batch evaluation. Delegates all
rendering to an injected IEvalReporter, so this class has a single
responsibility: computing results, not printing them.
"""

from __future__ import annotations

import time

from src.application.services.rag_query_service import RagQueryService
from src.application.services.retrieval_service import RetrievalService
from src.domain.entities import EvalCase, EvalResult
from src.domain.interfaces import IEvalReporter, ILogger, IPromptBuilder

DEFAULT_EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        question="How much money did Della have at the start?",
        expected_keywords=["1.87", "one dollar", "eighty-seven"],
        notes="Opening fact — should be high confidence",
    ),
    EvalCase(
        question="What did Della sell to get money?",
        expected_keywords=["hair", "brown hair"],
        notes="Core plot point",
    ),
    EvalCase(
        question="What gift did Jim buy for Della?",
        expected_keywords=["comb", "combs", "hair comb", "watch chain", "chain"],
        notes="Ironic gift reveal — context may surface watch chain instead of combs",
    ),
    EvalCase(
        question="What did Jim sell to buy Della's gift?",
        expected_keywords=["watch", "gold watch"],
        notes="Jim's sacrifice — tests retrieval of Jim-focused chunks",
    ),
    EvalCase(
        question="Who cut Della's hair?",
        expected_keywords=["sofronie", "mrs. sofronie"],
        notes="Specific named character",
    ),
    EvalCase(
        question="What is the theme of the story?",
        expected_keywords=["love", "sacrifice", "gift", "wise", "magi"],
        notes="Abstract question — tests generalization",
    ),
]


class EvaluationService:
    def __init__(
        self,
        rag_query_service: RagQueryService,
        retrieval_service: RetrievalService,
        prompt_builder: IPromptBuilder,
        reporter: IEvalReporter,
        logger: ILogger,
    ) -> None:
        self._rag_query_service = rag_query_service
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder
        self._reporter = reporter
        self._logger = logger

    def debug_query(self, question: str, top_k: int = 5) -> None:
        """Print a full diagnostic trace for a single query."""
        results = self._retrieval_service.search(question, top_k=top_k)
        package = self._prompt_builder.build(question, results)
        self._reporter.report_debug_query(question, results, package)

    def run_eval(self, suite: list[EvalCase] | None = None, stream: bool = False) -> list[EvalResult]:
        """Run a batch evaluation suite and report pass/fail per case + summary."""
        cases = suite or DEFAULT_EVAL_SUITE
        results: list[EvalResult] = []

        self._logger.info(f"Running {len(cases)} test cases...")

        for i, case in enumerate(cases):
            result = self._run_single_case(case, stream)
            results.append(result)
            self._reporter.report_case_result(i + 1, len(cases), result)

        self._reporter.report_summary(results)
        return results

    def _run_single_case(self, case: EvalCase, stream: bool) -> EvalResult:
        start = time.perf_counter()
        response = self._rag_query_service.ask(case.question, stream=stream)
        latency = time.perf_counter() - start

        answer_lower = response.answer.lower()
        matched = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
        passed = len(matched) > 0
        top_score = response.sources[0].score if response.sources else 0.0

        return EvalResult(
            case=case,
            answer=response.answer,
            passed=passed,
            matched_keywords=matched,
            latency_seconds=latency,
            top_score=top_score,
            sources=response.sources,
        )
