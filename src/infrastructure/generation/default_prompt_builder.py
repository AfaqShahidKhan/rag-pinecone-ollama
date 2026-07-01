"""
src/infrastructure/generation/default_prompt_builder.py

IPromptBuilder adapter. Pure string assembly — no external SDK — but lives
in infrastructure because it is a swappable strategy (e.g. you may later
add a DefaultPromptBuilder variant per LLM/prompt style).
"""

from __future__ import annotations

from src.config.settings import PromptSettings
from src.domain.entities import PromptPackage, SearchResult
from src.domain.interfaces import ILogger, IPromptBuilder


class DefaultPromptBuilder(IPromptBuilder):
    def __init__(self, logger: ILogger, prompt_settings: PromptSettings) -> None:
        self._logger = logger
        self._settings = prompt_settings

    def build(self, question: str, results: list[SearchResult]) -> PromptPackage:
        if not results:
            self._logger.warning("No search results provided — prompt will have empty context.")

        context_parts, total_chars, used = self._assemble_context(results)
        context = "\n\n".join(context_parts) if context_parts else "No context available."

        prompt = self._settings.prompt_template.format(
            system=self._settings.system_prompt,
            context=context,
            question=question,
        )

        self._logger.info(
            f"Prompt built — {used}/{len(results)} chunks used, {total_chars} context chars."
        )

        return PromptPackage(
            prompt=prompt,
            question=question,
            sources=results[:used],
            context_chunks_used=used,
            context_chars=total_chars,
        )

    def _assemble_context(self, results: list[SearchResult]) -> tuple[list[str], int, int]:
        context_parts: list[str] = []
        total_chars = 0
        used = 0
        budget = self._settings.max_context_chars

        for i, result in enumerate(results):
            chunk_text = f"[{i + 1}] (Source: {result.source}, Page {result.page})\n{result.text}"
            chunk_chars = len(chunk_text)

            if total_chars + chunk_chars > budget:
                self._logger.warning(
                    f"Context budget reached at chunk {i + 1}/{len(results)} "
                    f"({total_chars} chars) — truncating."
                )
                break

            context_parts.append(chunk_text)
            total_chars += chunk_chars
            used += 1

        return context_parts, total_chars, used
