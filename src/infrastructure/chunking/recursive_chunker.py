"""
src/infrastructure/chunking/recursive_chunker.py

ITextChunker adapter backed by langchain_text_splitters. Also responsible
for normalizing common PDF extraction artifacts before splitting.

Table-aware chunking: Markdown tables (produced by PdfDocumentLoader and
DocxDocumentLoader) are detected and pulled out of the normal recursive
splitter entirely, because splitting a table mid-row silently separates
data cells from their column headers — the LLM then has no way to tell
which column a value came from.

Default behavior per table:
  - Small/medium tables (<= chunking.max_table_chunk_chars): become ONE
    atomic chunk, however large that is relative to chunk_size. A whole
    table with full header context beats a "correctly sized" fragment
    that's missing its header row.
  - Oversized tables (> chunking.max_table_chunk_chars): split by row
    groups, with the header + separator row repeated at the top of every
    resulting chunk, so column context survives the split.

Prose surrounding the tables is chunked exactly as before, unaffected.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import ChunkingSettings
from src.domain.entities import Document
from src.domain.interfaces import ILogger, ITextChunker

# Matches runs of single characters separated by single spaces, e.g.
# "T h e  G i f t" -> "The Gift", a common PDF text-extraction artifact.
_SPACED_CHARS_PATTERN = re.compile(r"(?<!\w)((\w) )+(\w)(?!\w)")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")

# Detects a full Markdown table block: header row, separator row
# (e.g. "| --- | --- |"), then zero or more data rows — the exact format
# produced by PdfDocumentLoader / DocxDocumentLoader. Kept as a capturing
# group so re.split() returns tables interleaved with surrounding prose.
_TABLE_SEPARATOR = r"\|(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*"
_TABLE_BLOCK_RE = re.compile(
    rf"(^\|.*\|[ \t]*\n{_TABLE_SEPARATOR}\n(?:^\|.*\|[ \t]*\n?)*)",
    re.MULTILINE,
)


class RecursiveTextChunker(ITextChunker):
    def __init__(self, logger: ILogger, chunking_settings: ChunkingSettings) -> None:
        self._logger = logger
        self._settings = chunking_settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_settings.chunk_size,
            chunk_overlap=chunking_settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        table_chunk_count = 0

        for doc in documents:
            pieces = self._split_into_pieces(doc.page_content)
            if not pieces:
                continue

            for i, (text, is_table) in enumerate(pieces):
                extra_meta = {"is_table_chunk": True} if is_table else {}
                all_chunks.append(Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        **extra_meta,
                        "chunk_index": i,
                        "chunk_total": len(pieces),
                    },
                ))
                if is_table:
                    table_chunk_count += 1

        self._logger.info(
            f"Chunked {len(documents)} pages -> {len(all_chunks)} chunks "
            f"(size={self._settings.chunk_size}, overlap={self._settings.chunk_overlap}, "
            f"{table_chunk_count} table chunk(s) kept atomic/row-grouped)."
        )
        return all_chunks

    def _split_into_pieces(self, text: str) -> list[tuple[str, bool]]:
        """
        Returns an ordered list of (chunk_text, is_table_chunk) for one
        document's page_content, preserving the original order prose and
        table segments appeared in.
        """
        segments = _TABLE_BLOCK_RE.split(text)
        pieces: list[tuple[str, bool]] = []

        for index, segment in enumerate(segments):
            if not segment or not segment.strip():
                continue

            is_table_segment = bool(index % 2)  # odd indices are regex captures (tables)
            if is_table_segment:
                pieces.extend(
                    (table_piece, True) for table_piece in self._chunk_table(segment.strip())
                )
            else:
                cleaned = self._clean_text(segment)
                if cleaned:
                    pieces.extend((p, False) for p in self._splitter.split_text(cleaned))

        return pieces

    def _chunk_table(self, table_text: str) -> list[str]:
        """
        Small/medium tables are kept as ONE atomic chunk regardless of
        chunk_size. Tables larger than max_table_chunk_chars are split by
        data rows, with the header + separator row repeated at the top of
        each part so column context is never lost.
        """
        if len(table_text) <= self._settings.max_table_chunk_chars:
            return [table_text]

        lines = table_text.split("\n")
        if len(lines) < 3:
            # Not enough structure to safely split (header + separator only) — keep atomic.
            return [table_text]

        header, separator, *data_rows = lines
        prefix = f"{header}\n{separator}\n"
        max_body_chars = max(self._settings.max_table_chunk_chars - len(prefix), 1)

        parts: list[str] = []
        current_rows: list[str] = []
        current_len = 0

        for row in data_rows:
            row_len = len(row) + 1  # +1 for the joining newline
            if current_rows and current_len + row_len > max_body_chars:
                parts.append(prefix + "\n".join(current_rows))
                current_rows = []
                current_len = 0
            current_rows.append(row)
            current_len += row_len

        if current_rows:
            parts.append(prefix + "\n".join(current_rows))

        self._logger.debug(
            f"RecursiveTextChunker: table ({len(table_text)} chars) split into "
            f"{len(parts)} row-grouped chunk(s), header repeated in each."
        )
        return parts

    @staticmethod
    def _clean_text(text: str) -> str:
        text = _SPACED_CHARS_PATTERN.sub(lambda m: m.group(0).replace(" ", ""), text)
        text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
        text = _MULTI_SPACE_PATTERN.sub(" ", text)
        return text.strip()