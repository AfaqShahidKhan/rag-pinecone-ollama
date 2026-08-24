"""
src/infrastructure/pre_processing/pii_anonymizing_pre_processor.py

IPreProcessor adapter that wraps an IPiiAnonymizer and integrates it
into the pre-processing pipeline. Sits between MetadataEnricher and
SchemaMapper — after content is clean but before final validation.

Adds pii_redacted (bool) and pii_entities (list[str]) to the document
metadata so downstream components and the relational store can record
what was found and redacted.
"""

from __future__ import annotations

from src.domain.entities import Document
from src.domain.interfaces import IPiiAnonymizer, ILogger, IPreProcessor


class PiiAnonymizingPreProcessor(IPreProcessor):
    def __init__(self, anonymizer: IPiiAnonymizer, logger: ILogger) -> None:
        self._anonymizer = anonymizer
        self._logger = logger

    def process(self, document: Document) -> Document:
        result = self._anonymizer.anonymize(document.page_content)

        if result.redaction_count > 0:
            self._logger.info(
                f"PiiAnonymizer: redacted {result.redaction_count} item(s) "
                f"({result.entities_found}) from "
                f"'{document.metadata.get('source', '?')}' "
                f"p{document.metadata.get('page', '?')}"
            )

        meta = {
            **document.metadata,
            "pii_redacted": result.redaction_count > 0,
            "pii_entities": result.entities_found,
        }

        return Document(page_content=result.redacted_text, metadata=meta)