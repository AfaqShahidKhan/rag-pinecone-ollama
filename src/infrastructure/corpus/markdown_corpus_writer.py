"""
src/infrastructure/corpus/markdown_corpus_writer.py

Implements ICorpusWriter using Markdown files with YAML front-matter.

Writes one .md file per pre-processed Document into:

    {output_dir}/{sanitized_source_stem}/{page|section}_{NNN}.md

Uses "section_" instead of "page_" for file types that don't have a natural
page concept (html, json) — everything else uses "page_".

Runs after pre-processing, so it only ever sees clean, PII-redacted text —
never raw loader output. table_count comes from PdfDocumentLoader;
image_count comes from PdfImageExtractor ("images") or DocxImageExtractor
("source_images") — whichever is present.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.config.settings import CorpusSettings
from src.domain.entities import Document
from src.domain.interfaces import ICorpusWriter, ILogger

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_\-]+")
_SECTION_FILE_TYPES = frozenset({"html", "json"})


class MarkdownCorpusWriter(ICorpusWriter):
    def __init__(self, logger: ILogger, corpus_settings: CorpusSettings) -> None:
        self._logger = logger
        self._settings = corpus_settings

    def write(self, documents: list[Document]) -> int:
        written = 0
        for document in documents:
            path = self._resolve_path(document)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._render(document), encoding="utf-8")
            written += 1

        if written:
            self._logger.info(
                f"MarkdownCorpusWriter: wrote {written} file(s) to "
                f"'{self._settings.output_dir}'."
            )
        return written

    def _resolve_path(self, document: Document) -> Path:
        meta = document.metadata
        source = str(meta.get("source", "unknown"))
        stem = self._sanitize(Path(source).stem or "unknown")
        file_type = str(meta.get("file_type", "unknown")).lower()
        prefix = "section" if file_type in _SECTION_FILE_TYPES else "page"
        page = int(meta.get("page", 0) or 0)
        filename = f"{prefix}_{page:03d}.md"
        return Path(self._settings.output_dir) / stem / filename

    def _render(self, document: Document) -> str:
        meta = document.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        front_matter = self._build_front_matter(meta)
        heading = f"# {source} — Page {page}"
        return f"{front_matter}\n\n{heading}\n\n{document.page_content}\n"

    @staticmethod
    def _build_front_matter(meta: dict) -> str:
        image_count = len(meta.get("images", [])) + len(meta.get("source_images", []))
        lines = [
            "---",
            f"source: {meta.get('source', 'unknown')}",
            f"page: {meta.get('page', 0)}",
            f"total_pages: {meta.get('total_pages', 0)}",
            f"file_type: {meta.get('file_type', 'unknown')}",
            f"word_count: {meta.get('word_count', 0)}",
            f"has_tables: {str(meta.get('has_tables', False)).lower()}",
            f"table_count: {meta.get('table_count', 0)}",
            f"image_count: {image_count}",
            f"pii_redacted: {str(meta.get('pii_redacted', False)).lower()}",
            f"ingested_at: {meta.get('ingested_at', '')}",
            "---",
        ]
        return "\n".join(lines)

    @staticmethod
    def _sanitize(name: str) -> str:
        cleaned = _UNSAFE_CHARS_RE.sub("_", name).strip("_")
        return cleaned or "unknown"