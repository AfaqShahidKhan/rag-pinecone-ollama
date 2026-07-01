"""
src/application/services/rag_query_service.py

Top-level RAG orchestration: retrieve -> build prompt -> generate answer.
Equivalent to the original generation/rag.py `ask()`, now a class composed
entirely of injected interfaces.
"""

from __future__ import annotations

from src.application.services.retrieval_service import RetrievalService
from src.domain.entities import RAGResponse
from src.domain.interfaces import IAnswerGenerator, ILogger, IPromptBuilder


class RagQueryService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: IPromptBuilder,
        answer_generator: IAnswerGenerator,
        logger: ILogger,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator
        self._logger = logger

    def ask(self, question: str, top_k: int | None = None, stream: bool = False) -> RAGResponse:
        self._logger.info(f"RAG query: '{question}'")

        results = self._retrieval_service.search(question, top_k=top_k)
        package = self._prompt_builder.build(question, results)
        response = self._answer_generator.generate(package, stream=stream)

        return response
