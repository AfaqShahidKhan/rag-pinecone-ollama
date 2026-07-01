"""
src/infrastructure/generation/ollama_answer_generator.py

IAnswerGenerator adapter backed by the `ollama` SDK. Streaming output is
written through the injected ILogger/console abstraction rather than a
bare print(), keeping this class free of hidden I/O dependencies.
"""

from __future__ import annotations

import time
from typing import Callable

from ollama import Client as OllamaClient

from src.config.settings import OllamaSettings
from src.domain.entities import PromptPackage, RAGResponse
from src.domain.interfaces import IAnswerGenerator, ILogger

# Receives each streamed token as it arrives. Defaults to a no-op so the
# generator never has a hard dependency on stdout.
TokenSink = Callable[[str], None]


class OllamaAnswerGenerator(IAnswerGenerator):
    def __init__(
        self,
        client: OllamaClient,
        logger: ILogger,
        ollama_settings: OllamaSettings,
        token_sink: TokenSink | None = None,
    ) -> None:
        self._client = client
        self._logger = logger
        self._settings = ollama_settings
        self._token_sink = token_sink or (lambda _token: None)

    def generate(self, package: PromptPackage, stream: bool = False) -> RAGResponse:
        model = self._settings.generation_model
        self._logger.info(f"Generating answer with '{model}' (stream={stream})...")

        start = time.perf_counter()
        answer = self._stream_answer(model, package.prompt) if stream else self._single_shot_answer(model, package.prompt)
        latency = time.perf_counter() - start

        self._logger.info(f"Generation complete in {latency:.2f}s — {len(answer)} chars.")

        return RAGResponse(
            answer=answer,
            question=package.question,
            sources=package.sources,
            model=model,
            latency_seconds=latency,
            context_chunks_used=package.context_chunks_used,
        )

    def _stream_answer(self, model: str, prompt: str) -> str:
        parts: list[str] = []
        for chunk in self._client.generate(model=model, prompt=prompt, stream=True):
            token = chunk.get("response", "")
            self._token_sink(token)
            parts.append(token)
        return "".join(parts).strip()

    def _single_shot_answer(self, model: str, prompt: str) -> str:
        response = self._client.generate(model=model, prompt=prompt, stream=False)
        return response.get("response", "").strip()
