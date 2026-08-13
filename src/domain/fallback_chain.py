"""
src/domain/fallback_chain.py

Generic Chain-of-Responsibility executor. Given an ordered list of named
attempts and a confidence threshold, tries each in order and accepts the
first result meeting the threshold. If none meet it, returns the single
best attempt seen (so a marginal result still beats nothing), flagged so
the caller can log a warning.

Reused for all three fallback chains in this project: PDF text extraction,
PDF table extraction, and OCR engines — same mechanism, different strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.domain.entities import ExtractionAttempt


@dataclass(frozen=True)
class ChainResult:
    attempt: ExtractionAttempt
    fell_back: bool             # True if the primary (first) strategy wasn't the winner
    met_threshold: bool         # False if even the best attempt was below threshold
    attempts_tried: list[str]   # names of every strategy actually tried, in order


def run_fallback_chain(
    strategies: list[tuple[str, Callable[[], ExtractionAttempt]]],
    confidence_threshold: float,
) -> ChainResult:
    if not strategies:
        return ChainResult(
            attempt=ExtractionAttempt(content="", confidence=0.0, strategy_name="none"),
            fell_back=False,
            met_threshold=False,
            attempts_tried=[],
        )

    best: ExtractionAttempt | None = None
    tried: list[str] = []

    for name, run in strategies:
        tried.append(name)
        try:
            attempt = run()
        except Exception:
            continue  # this strategy failed outright — try the next one

        if best is None or attempt.confidence > best.confidence:
            best = attempt

        if attempt.confidence >= confidence_threshold:
            return ChainResult(
                attempt=attempt,
                fell_back=(name != strategies[0][0]),
                met_threshold=True,
                attempts_tried=tried,
            )

    if best is None:
        best = ExtractionAttempt(content="", confidence=0.0, strategy_name="none")

    return ChainResult(
        attempt=best,
        fell_back=(best.strategy_name != strategies[0][0]),
        met_threshold=False,
        attempts_tried=tried,
    )