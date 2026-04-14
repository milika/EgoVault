"""VaultStore — all SQLite read/write operations for the vault database."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from egovault.core.schema import NormalizedRecord

# Imported lazily to avoid a circular import during schema bootstrap;
# referenced only in type annotations.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from egovault.core.schema import EnrichmentStatus

_SCHEMA_VERSION = 6

# ---------------------------------------------------------------------------
# DDL — all tables, indexes, FTS5 virtual table, and triggers
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS normalized_records (
    id               TEXT PRIMARY KEY,
    platform         TEXT NOT NULL,
    record_type      TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    sender_id        TEXT NOT NULL DEFAULT '',
    sender_name      TEXT NOT NULL DEFAULT '',
    thread_id        TEXT NOT NULL DEFAULT '',
    thread_name      TEXT NOT NULL DEFAULT '',
    body             TEXT NOT NULL DEFAULT '',
    file_path        TEXT,
    mime_type        TEXT,
    attachments      TEXT NOT NULL DEFAULT '[]',
    raw              TEXT NOT NULL DEFAULT '{}',
    enriched         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    contextual_body  TEXT
);

CREATE INDEX IF NOT EXISTS idx_records_platform  ON normalized_records(platform);
CREATE INDEX IF NOT EXISTS idx_records_thread    ON normalized_records(thread_id);
CREATE INDEX IF NOT EXISTS idx_records_time      ON normalized_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_records_type      ON normalized_records(record_type);
CREATE INDEX IF NOT EXISTS idx_records_file_path ON normalized_records(file_path)
    WHERE file_path IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    body,
    thread_name,
    sender_name,
    file_path,
    content='normalized_records',
    content_rowid='rowid'
);

-- FTS5 triggers: when contextual_body is set, index it in the body column so
-- keyword search benefits from the richer context prefix (Contextual Retrieval).
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON normalized_records BEGIN
    INSERT INTO records_fts(rowid, body, thread_name, sender_name, file_path)
    VALUES (
        new.rowid,
        COALESCE(new.contextual_body, new.body),
        new.thread_name,
        new.sender_name,
        new.file_path
    );
END;

CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON normalized_records BEGIN
    INSERT INTO records_fts(records_fts, rowid, body, thread_name, sender_name, file_path)
    VALUES ('delete', old.rowid, COALESCE(old.contextual_body, old.body), old.thread_name, old.sender_name, old.file_path);
END;

CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON normalized_records BEGIN
    INSERT INTO records_fts(records_fts, rowid, body, thread_name, sender_name, file_path)
    VALUES ('delete', old.rowid, COALESCE(old.contextual_body, old.body), old.thread_name, old.sender_name, old.file_path);
    INSERT INTO records_fts(rowid, body, thread_name, sender_name, file_path)
    VALUES (
        new.rowid,
        COALESCE(new.contextual_body, new.body),
        new.thread_name,
        new.sender_name,
        new.file_path
    );
END;

CREATE TABLE IF NOT EXISTS enrichment_results (
    id          INTEGER PRIMARY KEY,
    record_id   TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    summary     TEXT NOT NULL,
    gems_raw    TEXT NOT NULL DEFAULT '',
    enriched_at TEXT NOT NULL DEFAULT (datetime('now')),
    token_count INTEGER,
    retries     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_enrichment_record ON enrichment_results(record_id);

CREATE TABLE IF NOT EXISTS extracted_gems (
    id            INTEGER PRIMARY KEY,
    record_id     TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
    gem_type      TEXT NOT NULL,
    content       TEXT NOT NULL,
    url           TEXT,
    attributed_to TEXT,
    attributed_at TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gems_record ON extracted_gems(record_id);
CREATE INDEX IF NOT EXISTS idx_gems_type   ON extracted_gems(gem_type);
CREATE INDEX IF NOT EXISTS idx_gems_url    ON extracted_gems(url) WHERE url IS NOT NULL;

CREATE TABLE IF NOT EXISTS speaker_sessions (
    id             INTEGER PRIMARY KEY,
    speaker_name   TEXT NOT NULL,
    alias          TEXT,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    session_count  INTEGER NOT NULL DEFAULT 1,
    inferred_level INTEGER NOT NULL,
    owner_override INTEGER,
    notes          TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_name ON speaker_sessions(speaker_name);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cloud_sync_log (
    id         INTEGER PRIMARY KEY,
    record_id  TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
    synced_at  TEXT NOT NULL,
    provider   TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT
);

CREATE INDEX IF NOT EXISTS idx_sync_record   ON cloud_sync_log(record_id);
CREATE INDEX IF NOT EXISTS idx_sync_provider ON cloud_sync_log(provider);
CREATE INDEX IF NOT EXISTS idx_sync_status   ON cloud_sync_log(status);

CREATE TABLE IF NOT EXISTS ingested_files (
    id           TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    mtime        REAL NOT NULL,
    size_bytes   INTEGER NOT NULL,
    platform     TEXT NOT NULL,
    content_hash TEXT,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ingested_path         ON ingested_files(path);

CREATE TABLE IF NOT EXISTS record_embeddings (
    record_id  TEXT PRIMARY KEY REFERENCES normalized_records(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON record_embeddings(model);

CREATE TABLE IF NOT EXISTS record_question_embeddings (
    id            INTEGER PRIMARY KEY,
    record_id     TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    question_text TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_qembed_record ON record_question_embeddings(record_id);
CREATE INDEX IF NOT EXISTS idx_qembed_model  ON record_question_embeddings(model);

CREATE TABLE IF NOT EXISTS record_chunks (
    id           INTEGER PRIMARY KEY,
    record_id    TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
    model        TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    BLOB NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(record_id, model, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_record ON record_chunks(record_id);
CREATE INDEX IF NOT EXISTS idx_chunks_model  ON record_chunks(model);
"""

_SETTINGS_DEFAULTS: list[tuple[str, str]] = [
    ("cloud_sync_enabled", "false"),
    ("schema_version", str(_SCHEMA_VERSION)),
]

# Column names allowed as filter keys in get_records() — guards against
# SQL injection via untrusted column names.
_ALLOWED_FILTER_COLUMNS: frozenset[str] = frozenset(
    {
        "platform",
        "record_type",
        "thread_id",
        "thread_name",
        "sender_id",
        "sender_name",
        "enriched",
        "mime_type",
    }
)


# ---------------------------------------------------------------------------
# Row → NormalizedRecord helper
# ---------------------------------------------------------------------------


def row_to_record(row: sqlite3.Row) -> NormalizedRecord:
    ts_str: str = row["timestamp"]
    ts = datetime.fromisoformat(ts_str)
    return NormalizedRecord(
        platform=row["platform"],
        record_type=row["record_type"],
        timestamp=ts,
        sender_id=row["sender_id"],
        sender_name=row["sender_name"],
        thread_id=row["thread_id"],
        thread_name=row["thread_name"],
        body=row["body"],
        attachments=json.loads(row["attachments"]),
        raw=json.loads(row["raw"]),
        file_path=row["file_path"],
        mime_type=row["mime_type"],
    )


# ---------------------------------------------------------------------------
# VaultStore
# ---------------------------------------------------------------------------


class VaultStore:
    """Manages a single SQLite vault database."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con: sqlite3.Connection = sqlite3.connect(self._path)
        self._con.row_factory = sqlite3.Row
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")

    def init_db(self) -> None:
        """Create all tables, indexes, triggers, seed default settings, and run migrations."""
        self._con.executescript(_DDL)
        for key, value in _SETTINGS_DEFAULTS:
            self._con.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        self._migrate()
        self._con.commit()

    def _migrate(self) -> None:
        """Apply incremental schema migrations to existing databases."""
        # v2 → v3: add content_hash column + index to ingested_files
        cur = self._con.execute("PRAGMA table_info(ingested_files)")
        cols = {row[1] for row in cur.fetchall()}
        if "content_hash" not in cols:
            self._con.execute(
                "ALTER TABLE ingested_files ADD COLUMN content_hash TEXT"
            )
        # Always ensure the partial index exists (idempotent, safe on all schema versions)
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingested_content_hash "
            "ON ingested_files(content_hash) WHERE content_hash IS NOT NULL"
        )
        # v3 → v4: add contextual_body column to normalized_records
        cur = self._con.execute("PRAGMA table_info(normalized_records)")
        rec_cols = {row[1] for row in cur.fetchall()}
        if "contextual_body" not in rec_cols:
            self._con.execute(
                "ALTER TABLE normalized_records ADD COLUMN contextual_body TEXT"
            )
        # v4 → v5: create record_question_embeddings table (HyPE §6)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS record_question_embeddings (
                id            INTEGER PRIMARY KEY,
                record_id     TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
                model         TEXT NOT NULL,
                question_text TEXT NOT NULL,
                embedding     BLOB NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_qembed_record "
            "ON record_question_embeddings(record_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_qembed_model "
            "ON record_question_embeddings(model)"
        )
        # v5 → v6: create record_chunks table (Sentence Window Retrieval §8)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS record_chunks (
                id           INTEGER PRIMARY KEY,
                record_id    TEXT NOT NULL REFERENCES normalized_records(id) ON DELETE CASCADE,
                model        TEXT NOT NULL,
                chunk_index  INTEGER NOT NULL,
                chunk_text   TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(record_id, model, chunk_index)
            )
            """
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_record ON record_chunks(record_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_model ON record_chunks(model)"
        )
        self._con.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )

    def upsert_record(self, record: NormalizedRecord) -> bool:
        """Insert *record* if its id is not already present.

        Returns True when a new row was inserted, False when it was a duplicate.
        """
        cur = self._con.execute(
            """
            INSERT OR IGNORE INTO normalized_records
                (id, platform, record_type, timestamp, sender_id, sender_name,
                 thread_id, thread_name, body, file_path, mime_type,
                 attachments, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.platform,
                record.record_type,
                record.timestamp.isoformat(),
                record.sender_id,
                record.sender_name,
                record.thread_id,
                record.thread_name,
                record.body,
                record.file_path,
                record.mime_type,
                json.dumps(record.attachments),
                json.dumps(record.raw),
            ),
        )
        self._con.commit()
        return cur.rowcount == 1

    def count_unenriched_records(self) -> int:
        """Return total count of records with enriched = 0 (pending)."""
        row = self._con.execute(
            "SELECT COUNT(*) FROM normalized_records WHERE enriched = 0"
        ).fetchone()
        return row[0] if row else 0

    def count_uncontextualized_records(self) -> int:
        """Return total count of records without a contextual_body set."""
        row = self._con.execute(
            "SELECT COUNT(*) FROM normalized_records WHERE contextual_body IS NULL"
        ).fetchone()
        return row[0] if row else 0

    def get_unenriched_records(self, limit: int = 100) -> list[NormalizedRecord]:
        """Return up to *limit* records with enriched = 0 (pending)."""
        cur = self._con.execute(
            "SELECT * FROM normalized_records WHERE enriched = 0 LIMIT ?",
            (limit,),
        )
        return [row_to_record(row) for row in cur.fetchall()]

    def mark_enriched(self, record_id: str, status: "int | EnrichmentStatus") -> None:
        """Update the enriched flag for a single record.

        status values: EnrichmentStatus.PENDING=0, DONE=1, FAILED=2, SKIPPED=3.
        """
        self._con.execute(
            "UPDATE normalized_records SET enriched = ? WHERE id = ?",
            (status, record_id),
        )
        self._con.commit()

    def get_records(self, filters: dict[str, object] | None = None) -> list[NormalizedRecord]:
        """Return records matching all equality *filters*.

        Allowed filter keys: platform, record_type, thread_id, thread_name,
        sender_id, sender_name, enriched, mime_type.
        """
        query = "SELECT * FROM normalized_records"
        params: list[object] = []
        if filters:
            clauses: list[str] = []
            for col, val in filters.items():
                if col not in _ALLOWED_FILTER_COLUMNS:
                    raise ValueError(
                        f"Unknown filter column {col!r}. "
                        f"Allowed: {sorted(_ALLOWED_FILTER_COLUMNS)}"
                    )
                clauses.append(f"{col} = ?")
                params.append(val)
            query += " WHERE " + " AND ".join(clauses)
        cur = self._con.execute(query, params)
        return [row_to_record(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # ingested_files — change detection for local file adapters
    # ------------------------------------------------------------------

    def is_file_known(self, file_id: str) -> bool:
        """Return True if *file_id* is already in the ingested_files table."""
        cur = self._con.execute(
            "SELECT 1 FROM ingested_files WHERE id = ?", (file_id,)
        )
        return cur.fetchone() is not None

    def is_file_known_by_hash(self, content_hash: str) -> bool:
        """Return True if a file with *content_hash* was already ingested.

        Catches renamed / re-touched files whose path-based id changed but
        whose byte content is identical to a previously ingested file.
        """
        cur = self._con.execute(
            "SELECT 1 FROM ingested_files WHERE content_hash = ?", (content_hash,)
        )
        return cur.fetchone() is not None

    def upsert_ingested_file(
        self,
        file_id: str,
        path: str,
        mtime: float,
        size_bytes: int,
        platform: str,
        content_hash: str | None = None,
    ) -> None:
        """Insert or replace a row in ingested_files."""
        self._con.execute(
            """
            INSERT OR REPLACE INTO ingested_files
                (id, path, mtime, size_bytes, platform, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, path, mtime, size_bytes, platform, content_hash),
        )
        self._con.commit()

    def record_needs_body_update(self, file_path: str) -> bool:
        """Return True if a record exists for *file_path* but has no substantive body.

        A body is considered empty when it is blank or contains only the
        auto-generated ``File: …\\n---\\n`` path header (< 60 chars of real content
        after the separator).
        """
        row = self._con.execute(
            "SELECT body FROM normalized_records WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        if row is None:
            return False
        body: str = row["body"] or ""
        # Strip the "File: …\n---\n" path header if present
        if "\n---\n" in body:
            content = body.split("\n---\n", 1)[1]
        else:
            content = body
        return len(content.strip()) < 60

    def update_body_by_file_path(self, file_path: str, new_body: str) -> bool:
        """Replace the body of the record identified by *file_path*.

        The FTS5 UPDATE trigger keeps the search index in sync automatically.
        Returns True when a row was actually updated.
        """
        cur = self._con.execute(
            "UPDATE normalized_records SET body = ? WHERE file_path = ?",
            (new_body, file_path),
        )
        self._con.commit()
        return cur.rowcount > 0

    def update_record_body(self, record_id: str, new_body: str) -> bool:
        """Replace the body of *record_id* in place.

        The FTS5 UPDATE trigger keeps the search index in sync automatically.
        Returns True when a row was actually updated.
        """
        cur = self._con.execute(
            "UPDATE normalized_records SET body = ? WHERE id = ?",
            (new_body, record_id),
        )
        self._con.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # enrichment_results + extracted_gems
    # ------------------------------------------------------------------

    def insert_enrichment_result(
        self,
        record_id: str,
        model: str,
        summary: str,
        gems_raw: str = "",
        token_count: int | None = None,
        retries: int = 0,
    ) -> None:
        """Insert or replace an enrichment result for a record."""
        with self._con:
            self._con.execute(
                """
                INSERT OR REPLACE INTO enrichment_results
                    (record_id, model, summary, gems_raw, token_count, retries)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (record_id, model, summary, gems_raw, token_count, retries),
            )

    def insert_gem(
        self,
        record_id: str,
        gem_type: str,
        content: str,
        url: str | None = None,
        attributed_to: str | None = None,
        attributed_at: str | None = None,
    ) -> None:
        """Insert a single extracted gem row."""
        with self._con:
            self._con.execute(
                """
                INSERT INTO extracted_gems
                    (record_id, gem_type, content, url, attributed_to, attributed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (record_id, gem_type, content, url, attributed_to, attributed_at),
            )

    def get_setting(self, key: str) -> str | None:
        """Return the value for *key* from the settings table, or None."""
        row = self._con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Upsert a key/value pair in the settings table."""
        self._con.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._con.commit()

    def get_owner_profile(self) -> str:
        """Return the cached owner profile string, or '' if not yet extracted."""
        return self.get_setting("owner_profile") or ""

    def set_owner_profile(self, profile: str) -> None:
        """Persist the owner profile extracted by the LLM."""
        self.set_setting("owner_profile", profile)

    def _build_date_filter(
        self,
        platform: str | None,
        since: str | None,
        until: str | None,
    ) -> tuple[list[str], list]:
        """Build WHERE-clause conditions and params for platform/date filters."""
        conditions: list[str] = []
        params: list = []
        if platform:
            conditions.append("platform = ?")
            params.append(platform)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            until_end = until if "T" in until else f"{until}T23:59:59"
            conditions.append("timestamp <= ?")
            params.append(until_end)
        return conditions, params

    def count_records(
        self,
        platform: str | None = None,
        since: str | None = None,
        until: str | None = None,
        enriched: bool | None = None,
        record_type: str | None = None,
    ) -> dict:
        """Count vault records matching optional filters.

        Returns ``{"total": int, "breakdown": [{"platform": str, "count": int}, ...]}``.
        *since* and *until* are ISO date strings (``YYYY-MM-DD``).
        *until* is inclusive for the whole day (``T23:59:59``).
        *enriched* — True: only records that have an enrichment_results row;
                      False: only unenriched records; None: all records.
        *record_type* — "image"/"photo"/"picture" matches mime_type LIKE 'image/%'
                        or common image file extensions.
        """
        conditions, params = self._build_date_filter(platform, since, until)

        # Image / picture filter
        _IMAGE_EXTS = ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif", "heic", "svg")
        _normalized_type = (record_type or "").lower().strip()
        if _normalized_type in ("image", "photo", "picture", "img"):
            ext_checks = " OR ".join(
                f"LOWER(file_path) LIKE '%.{e}'" for e in _IMAGE_EXTS
            )
            conditions.append(
                f"(mime_type LIKE 'image/%' OR {ext_checks})"
            )

        # Enriched / unenriched filter — use EXISTS subquery (no extra JOIN needed)
        if enriched is True:
            conditions.append(
                "EXISTS (SELECT 1 FROM enrichment_results er WHERE er.record_id = id)"
            )
        elif enriched is False:
            conditions.append(
                "NOT EXISTS (SELECT 1 FROM enrichment_results er WHERE er.record_id = id)"
            )

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        row = self._con.execute(
            f"SELECT COUNT(*) AS n FROM normalized_records {where}", params
        ).fetchone()
        total = row["n"] if row else 0
        breakdown_rows = self._con.execute(
            f"SELECT platform, COUNT(*) AS n FROM normalized_records {where} "
            f"GROUP BY platform ORDER BY n DESC",
            params,
        ).fetchall()
        breakdown = [{"platform": r["platform"], "count": r["n"]} for r in breakdown_rows]
        return {"total": total, "breakdown": breakdown}

    def list_records(
        self,
        platform: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
    ) -> list:
        """Return rows from normalized_records filtered by platform / date range.

        Results are ordered newest-first.  *since* / *until* are ISO date strings.
        Returns raw ``sqlite3.Row`` objects so callers can use ``row_to_record``.
        """
        conditions, params = self._build_date_filter(platform, since, until)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        return self._con.execute(
            f"SELECT * FROM normalized_records {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

    def vault_stats(self) -> dict:
        """Return a dict with basic vault statistics for context injection.

        Keys: total_records, sources (list of platform/file_path labels), date_min, date_max.
        """
        row = self._con.execute("SELECT COUNT(*) AS n FROM normalized_records").fetchone()
        total = row["n"] if row else 0

        # Unique source labels: prefer thread_name, fall back to file_path base name, then platform
        src_rows = self._con.execute(
            """
            SELECT DISTINCT
                COALESCE(NULLIF(TRIM(thread_name), ''), NULLIF(file_path, ''), platform) AS label
            FROM normalized_records
            ORDER BY label
            LIMIT 50
            """
        ).fetchall()
        sources = [r["label"] for r in src_rows]

        date_row = self._con.execute(
            "SELECT MIN(timestamp) AS d_min, MAX(timestamp) AS d_max FROM normalized_records"
        ).fetchone()
        date_min = (date_row["d_min"] or "")[:10] if date_row else ""
        date_max = (date_row["d_max"] or "")[:10] if date_row else ""

        return {"total_records": total, "sources": sources, "date_min": date_min, "date_max": date_max}

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    # ------------------------------------------------------------------
    # record_embeddings — dense vector index for semantic search
    # ------------------------------------------------------------------

    def upsert_embedding(self, record_id: str, model: str, vector: list[float]) -> None:
        """Store a dense embedding for *record_id*.

        The vector is packed as a little-endian float32 BLOB.
        An existing embedding for the same record_id is replaced.
        """
        import struct
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._con.execute(
            """
            INSERT OR REPLACE INTO record_embeddings (record_id, model, embedding)
            VALUES (?, ?, ?)
            """,
            (record_id, model, blob),
        )
        self._con.commit()

    def get_all_embeddings(self, model: str) -> list[tuple[str, bytes]]:
        """Return all stored embeddings for *model* as ``(record_id, blob)`` pairs."""
        rows = self._con.execute(
            "SELECT record_id, embedding FROM record_embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        return [(row["record_id"], bytes(row["embedding"])) for row in rows]

    def get_unembedded_record_ids(self, model: str, limit: int = 5000) -> list[str]:
        """Return IDs of records that do not yet have an embedding for *model*."""
        rows = self._con.execute(
            """
            SELECT id FROM normalized_records
            WHERE id NOT IN (
                SELECT record_id FROM record_embeddings WHERE model = ?
            )
            LIMIT ?
            """,
            (model, limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def get_record_text_by_id(self, record_id: str) -> str:
        """Return text for *record_id* suitable for embedding.

        When a contextual_body prefix has been generated (Contextual Retrieval),
        it is returned instead of the raw body so embeddings benefit from the
        document-level context blurb.
        """
        row = self._con.execute(
            "SELECT thread_name, sender_name, body, contextual_body "
            "FROM normalized_records WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return ""
        # Use contextual_body when available (Contextual Retrieval §5)
        embed_body: str = row["contextual_body"] or row["body"] or ""
        parts: list[str] = []
        if row["thread_name"]:
            parts.append(row["thread_name"])
        if row["sender_name"]:
            parts.append(row["sender_name"])
        parts.append(embed_body[:2000])
        return " ".join(parts)

    def upsert_contextual_body(self, record_id: str, contextual_body: str) -> None:
        """Persist a context-prefixed version of *record_id*'s body.

        The contextual_body is the original body prepended with a short LLM-generated
        blurb that situates the chunk within the document (Contextual Retrieval §5).
        Setting it to an empty string clears the prefix (falls back to body).
        """
        self._con.execute(
            "UPDATE normalized_records SET contextual_body = ? WHERE id = ?",
            (contextual_body or None, record_id),
        )
        self._con.commit()

    def get_uncontextualized_record_ids(self, limit: int = 5000) -> list[str]:
        """Return IDs of records that do not yet have a contextual_body set."""
        rows = self._con.execute(
            "SELECT id FROM normalized_records WHERE contextual_body IS NULL LIMIT ?",
            (limit,),
        ).fetchall()
        return [row["id"] for row in rows]

    # ------------------------------------------------------------------
    # record_question_embeddings — HyPE (Hypothetical Prompt Embeddings, §6)
    # ------------------------------------------------------------------

    def upsert_question_embedding(
        self,
        record_id: str,
        model: str,
        question_text: str,
        vector: list[float],
    ) -> None:
        """Insert a single question embedding for *record_id*.

        Multiple questions per record are supported — call once per question.
        Each (record_id, question_text) pair is unique; re-calling with the
        same pair replaces the previous embedding.
        """
        import struct
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._con.execute(
            """
            INSERT OR REPLACE INTO record_question_embeddings
                (record_id, model, question_text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (record_id, model, question_text, blob),
        )
        self._con.commit()

    def get_all_question_embeddings(
        self, model: str
    ) -> list[tuple[str, str, bytes]]:
        """Return all stored question embeddings for *model*.

        Returns a list of ``(record_id, question_text, embedding_blob)`` triples.
        """
        rows = self._con.execute(
            "SELECT record_id, question_text, embedding "
            "FROM record_question_embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        return [(row["record_id"], row["question_text"], bytes(row["embedding"])) for row in rows]

    def get_records_without_hype_questions(self, model: str, limit: int = 5000) -> list[str]:
        """Return IDs of records that have no question embeddings for *model*."""
        rows = self._con.execute(
            """
            SELECT id FROM normalized_records
            WHERE id NOT IN (
                SELECT DISTINCT record_id FROM record_question_embeddings WHERE model = ?
            )
            LIMIT ?
            """,
            (model, limit),
        ).fetchall()
        return [row["id"] for row in rows]

    # ------------------------------------------------------------------
    # record_chunks — Sentence Window Retrieval (§8)
    # ------------------------------------------------------------------

    def upsert_chunk_embedding(
        self,
        record_id: str,
        model: str,
        chunk_index: int,
        chunk_text: str,
        vector: list[float],
    ) -> None:
        """Insert or replace a single sentence-window chunk embedding.

        The ``(record_id, model, chunk_index)`` triple is unique; re-calling
        with the same triple replaces the previous embedding.
        """
        import struct
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._con.execute(
            """
            INSERT OR REPLACE INTO record_chunks
                (record_id, model, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record_id, model, chunk_index, chunk_text, blob),
        )
        self._con.commit()

    def get_all_chunk_embeddings(
        self, model: str
    ) -> list[tuple[str, int, str, bytes]]:
        """Return all stored chunk embeddings for *model*.

        Returns a list of ``(record_id, chunk_index, chunk_text, embedding_blob)``
        quadruples ordered by ``(record_id, chunk_index)``.
        """
        rows = self._con.execute(
            "SELECT record_id, chunk_index, chunk_text, embedding "
            "FROM record_chunks WHERE model = ? "
            "ORDER BY record_id, chunk_index",
            (model,),
        ).fetchall()
        return [
            (row["record_id"], row["chunk_index"], row["chunk_text"], bytes(row["embedding"]))
            for row in rows
        ]

    def get_chunks_for_record(
        self, record_id: str, model: str
    ) -> list[tuple[int, str, bytes]]:
        """Return all chunks for *record_id* ordered by chunk_index.

        Returns a list of ``(chunk_index, chunk_text, embedding_blob)`` triples.
        """
        rows = self._con.execute(
            "SELECT chunk_index, chunk_text, embedding "
            "FROM record_chunks WHERE record_id = ? AND model = ? "
            "ORDER BY chunk_index",
            (record_id, model),
        ).fetchall()
        return [(row["chunk_index"], row["chunk_text"], bytes(row["embedding"])) for row in rows]

    def get_records_without_chunks(self, model: str, limit: int = 5000) -> list[str]:
        """Return IDs of records that have no chunk embeddings for *model*."""
        rows = self._con.execute(
            """
            SELECT id FROM normalized_records
            WHERE id NOT IN (
                SELECT DISTINCT record_id FROM record_chunks WHERE model = ?
            )
            LIMIT ?
            """,
            (model, limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def fetch_records_by_ids(self, ids: list[str]) -> list:
        """Fetch full normalized_records rows for the given *ids*.

        Returns rows in arbitrary order.  Use this instead of accessing
        ``store._con`` directly for id-in-list queries.
        """
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        return self._con.execute(
            f"SELECT * FROM normalized_records WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
