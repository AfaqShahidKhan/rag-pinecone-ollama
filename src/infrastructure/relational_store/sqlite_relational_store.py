"""
src/infrastructure/relational_store/sqlite_relational_store.py

IRelationalStore adapter backed by SQLite (stdlib — no extra install).

Schema:
  chunks table:
    vector_id    TEXT PRIMARY KEY  — matches the ID in the vector store
    source       TEXT              — filename
    page         INTEGER
    chunk_index  INTEGER
    content      TEXT              — original (or redacted) chunk text
    metadata_json TEXT             — full metadata dict as JSON
    ingested_at  TEXT              — ISO 8601 UTC timestamp

Enables:
  - Parent-child retrieval: look up full chunk text by vector_id after
    a similarity search returns IDs.
  - Source-level management: list or delete all chunks from a given file.
  - Audit trail: metadata_json preserves pii_redacted, pii_entities,
    word_count, has_tables, etc. for debugging and compliance.

The DB file is stored at RelationalStoreSettings.db_path (default:
./data/relational/rag_chunks.db), which is in .gitignore.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.config.settings import RelationalStoreSettings
from src.domain.entities import Document, EmbeddedChunk
from src.domain.interfaces import ILogger, IRelationalStore

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    vector_id     TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    page          INTEGER NOT NULL DEFAULT 0,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    content       TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    ingested_at   TEXT NOT NULL
)
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO chunks
    (vector_id, source, page, chunk_index, content, metadata_json, ingested_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_BY_VECTOR_ID = "SELECT content, metadata_json FROM chunks WHERE vector_id = ?"
_SELECT_BY_SOURCE    = "SELECT content, metadata_json FROM chunks WHERE source = ?"
_DELETE_BY_SOURCE    = "DELETE FROM chunks WHERE source = ?"


class SqliteRelationalStore(IRelationalStore):
    def __init__(
        self,
        logger: ILogger,
        settings: RelationalStoreSettings,
    ) -> None:
        self._logger = logger
        self._db_path = Path(settings.db_path)
        self._connection: sqlite3.Connection | None = None

    def ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        self._logger.info(
            f"SqliteRelationalStore: schema ready at '{self._db_path}'."
        )

    def save_chunks(self, chunks: list[EmbeddedChunk], vector_ids: list[str]) -> int:
        conn = self._get_connection()
        rows = []

        for chunk, vector_id in zip(chunks, vector_ids):
            meta = chunk.document.metadata
            rows.append((
                vector_id,
                meta.get("source", "unknown"),
                int(meta.get("page", 0)),
                int(meta.get("chunk_index", 0)),
                chunk.document.page_content,
                json.dumps(meta),
                meta.get("ingested_at", ""),
            ))

        conn.executemany(_INSERT_SQL, rows)
        conn.commit()
        self._logger.info(
            f"SqliteRelationalStore: saved {len(rows)} chunks."
        )
        return len(rows)

    def get_chunk_by_vector_id(self, vector_id: str) -> Document | None:
        conn = self._get_connection()
        row = conn.execute(_SELECT_BY_VECTOR_ID, (vector_id,)).fetchone()
        if row is None:
            return None
        content, metadata_json = row
        return Document(
            page_content=content,
            metadata=json.loads(metadata_json),
        )

    def search_by_source(self, source: str) -> list[Document]:
        conn = self._get_connection()
        rows = conn.execute(_SELECT_BY_SOURCE, (source,)).fetchall()
        return [
            Document(
                page_content=row[0],
                metadata=json.loads(row[1]),
            )
            for row in rows
        ]

    def delete_by_source(self, source: str) -> int:
        conn = self._get_connection()
        cursor = conn.execute(_DELETE_BY_SOURCE, (source,))
        conn.commit()
        deleted = cursor.rowcount
        self._logger.info(
            f"SqliteRelationalStore: deleted {deleted} chunks for source '{source}'."
        )
        return deleted

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection