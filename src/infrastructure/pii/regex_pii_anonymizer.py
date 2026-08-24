"""
src/infrastructure/pii/regex_pii_anonymizer.py

IPiiAnonymizer adapter backed by pure Python regex. No external ML
dependencies required — works offline, fast, and deterministic.

Detects and redacts:
  EMAIL          user@example.com          → [EMAIL]
  PHONE_PK       Pakistani mobile numbers  → [PHONE]
  PHONE_INTL     International numbers     → [PHONE]
  CNIC           Pakistani CNIC XXXXX-XXXXXXX-X → [CNIC]
  IBAN           PK-format IBAN            → [IBAN]
  CREDIT_CARD    16-digit card numbers     → [CREDIT_CARD]
  URL            http/https URLs           → [URL]
  IP_ADDRESS     IPv4 addresses            → [IP_ADDRESS]
  DATE_OF_BIRTH  "Date of Birth: DD/MM/YYYY" patterns → [DOB]

Patterns are applied in order — more specific patterns first to avoid
partial matches shadowing more precise ones.

To add a new entity type: add an entry to _PII_PATTERNS with a label
and compiled regex. The rest of the pipeline picks it up automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.domain.interfaces import IPiiAnonymizer, RedactionResult
from src.domain.interfaces import ILogger


@dataclass
class _PiiPattern:
    label: str
    pattern: re.Pattern
    placeholder: str


_PII_PATTERNS: list[_PiiPattern] = [
    # Email — must come before URL to avoid partial matches
    _PiiPattern(
        label="EMAIL",
        pattern=re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        placeholder="[EMAIL]",
    ),
    # URLs
    _PiiPattern(
        label="URL",
        pattern=re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE),
        placeholder="[URL]",
    ),
    # Pakistani CNIC: XXXXX-XXXXXXX-X
    _PiiPattern(
        label="CNIC",
        pattern=re.compile(r"\b\d{5}-\d{7}-\d{1}\b"),
        placeholder="[CNIC]",
    ),
    # Pakistani IBAN: PK + 2 digits + 4 alpha + 16 digits
    _PiiPattern(
        label="IBAN",
        pattern=re.compile(r"\bPK\d{2}[A-Z]{4}\d{16}\b", re.IGNORECASE),
        placeholder="[IBAN]",
    ),
    # Credit card: 16 digits with optional separators
    _PiiPattern(
        label="CREDIT_CARD",
        pattern=re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
        placeholder="[CREDIT_CARD]",
    ),
    # Pakistani mobile: 03XX-XXXXXXX or +923XXXXXXXXX
    _PiiPattern(
        label="PHONE_PK",
        pattern=re.compile(r"(?:\+92|0092|0)3\d{2}[\s\-]?\d{7}\b"),
        placeholder="[PHONE]",
    ),
    # International phone: +XX (X)XXX XXX XXXX variations
    _PiiPattern(
        label="PHONE_INTL",
        pattern=re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}\b"),
        placeholder="[PHONE]",
    ),
    # IPv4 address
    _PiiPattern(
        label="IP_ADDRESS",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        placeholder="[IP_ADDRESS]",
    ),
    # Date of birth contextual pattern
    _PiiPattern(
        label="DATE_OF_BIRTH",
        pattern=re.compile(
            r"(?:date\s+of\s+birth|dob|d\.o\.b\.?)\s*:?\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}",
            re.IGNORECASE,
        ),
        placeholder="[DOB]",
    ),
]


class RegexPiiAnonymizer(IPiiAnonymizer):
    def __init__(self, logger: ILogger, enabled_types: list[str] | None = None) -> None:
        """
        Args:
            logger:        Injected logger.
            enabled_types: Optional whitelist of entity type labels to enforce.
                           If None, all patterns are active.
        """
        self._logger = logger
        self._active_patterns = [
            p for p in _PII_PATTERNS
            if enabled_types is None or p.label in enabled_types
        ]

    @property
    def entity_types(self) -> list[str]:
        return [p.label for p in self._active_patterns]

    def anonymize(self, text: str) -> RedactionResult:
        redacted = text
        entities_found: list[str] = []
        total_count = 0

        for pii_pattern in self._active_patterns:
            matches = pii_pattern.pattern.findall(redacted)
            if matches:
                redacted = pii_pattern.pattern.sub(pii_pattern.placeholder, redacted)
                entities_found.append(pii_pattern.label)
                total_count += len(matches)
                self._logger.debug(
                    f"PiiAnonymizer: redacted {len(matches)} {pii_pattern.label} instance(s)."
                )

        return RedactionResult(
            redacted_text=redacted,
            entities_found=entities_found,
            redaction_count=total_count,
        )