"""
src/infrastructure/chunking/recursive_chunker.py

ITextChunker adapter backed by langchain_text_splitters. Also responsible
for normalizing common PDF extraction artifacts before splitting.
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

        for doc in documents:
            cleaned = self._clean_text(doc.page_content)
            if not cleaned:
                continue

            splits = self._splitter.split_text(cleaned)

            for i, chunk_text in enumerate(splits):
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_total": len(splits),
                    },
                ))

        self._logger.info(
            f"Chunked {len(documents)} pages -> {len(all_chunks)} chunks "
            f"(size={self._settings.chunk_size}, overlap={self._settings.chunk_overlap})"
        )
        return all_chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        text = _SPACED_CHARS_PATTERN.sub(lambda m: m.group(0).replace(" ", ""), text)
        text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
        text = _MULTI_SPACE_PATTERN.sub(" ", text)
        return text.strip()
