"""
src/infrastructure/vector_store/sha256_vector_id_strategy.py

IVectorIdStrategy adapter. Derives a stable ID from source + page + chunk
position so re-ingesting the same file is idempotent.
"""

from __future__ import annotations

import hashlib

from src.domain.entities import EmbeddedChunk
from src.domain.interfaces import IVectorIdStrategy

_ID_LENGTH = 32


class Sha256VectorIdStrategy(IVectorIdStrategy):
    def generate_id(self, chunk: EmbeddedChunk) -> str:
        meta = chunk.document.metadata
        key = f"{meta['source']}::p{meta['page']}::c{meta['chunk_index']}"
        return hashlib.sha256(key.encode()).hexdigest()[:_ID_LENGTH]
