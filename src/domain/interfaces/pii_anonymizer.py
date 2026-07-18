"""
src/domain/interfaces/pii_anonymizer.py

Port for PII detection and redaction. Implementations may use regex,
spaCy NER, Microsoft Presidio, or any other strategy. The application
layer only depends on this contract, never on the implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RedactionResult:
    """
    The outcome of a single anonymization pass.

    redacted_text:    The text with PII replaced by placeholder tokens.
    entities_found:   List of PII entity type labels detected (e.g. "EMAIL", "PHONE").
    redaction_count:  Total number of replacements made.
    """
    redacted_text: str
    entities_found: list[str]
    redaction_count: int


class IPiiAnonymizer(ABC):
    @abstractmethod
    def anonymize(self, text: str) -> RedactionResult:
        """
        Detect and redact PII in the given text.
        Returns a RedactionResult with the cleaned text and detection metadata.
        """
        ...

    @property
    @abstractmethod
    def entity_types(self) -> list[str]:
        """Return the list of PII entity types this anonymizer can detect."""
        ...