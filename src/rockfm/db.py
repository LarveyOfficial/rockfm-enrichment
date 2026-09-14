"""SQLite storage.

One database, WAL mode, shared by the ingest writer and the playout/analyzer
readers. Time is stored everywhere as epoch milliseconds UTC.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Where the analyzer has read up to. Kept here rather than in analyzer.py so the
# web process can report progress without importing the numeric stack.
ANALYZER_CURSOR_KEY = "analyzer_cursor_ms"

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    pdt_ms      INTEGER PRIMARY KEY,   -- EXT-X-PROGRAM-DATE-TIME, epoch ms UTC
    seq         INTEGER NOT NULL,      -- upstream media sequence number
    duration_ms INTEGER NOT NULL,
    relpath     TEXT    NOT NULL,      -- relative to the segments dir
    bytes       INTEGER NOT NULL,
    fetched_ms  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_seq ON segments(seq);

CREATE TABLE IF NOT EXISTS gaps (
    id           INTEGER PRIMARY KEY,
    after_pdt_ms INTEGER NOT NULL,
    before_pdt_ms INTEGER NOT NULL,
    missing_ms   INTEGER NOT NULL,
    reason       TEXT,
    detected_ms  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gaps_after ON gaps(after_pdt_ms);

CREATE TABLE IF NOT EXISTS timeline (
    id         INTEGER PRIMARY KEY,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    kind       TEXT    NOT NULL,  -- cancion|publicidad|programa|noticias|sintonia|desconocido
    title      TEXT,
    artist     TEXT,
    album      TEXT,
    year       INTEGER,
    art_url    TEXT,
    show_title TEXT,               -- programme on air, from /ply/prg
    show_lead  TEXT,               -- presenter names
    show_image TEXT,
    confidence REAL,
    source     TEXT,              -- local|shazamio|audd|acrcloud|schedule|repetition
    cluster_id INTEGER,
    created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_timeline_start ON timeline(start_ms);

CREATE TABLE IF NOT EXISTS fp_tracks (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,      -- 'music' (songs) | 'nonmusic' (ads, jingles, sweepers)
    key         TEXT NOT NULL UNIQUE,
    title       TEXT,
    artist      TEXT,
    source      TEXT,               -- preview|broadcast|repetition
    occurrences INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    anchor_ms   INTEGER,            -- broadcast ms corresponding to reference offset 0
    created_ms  INTEGER NOT NULL,
    updated_ms  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fp_tracks_kind ON fp_tracks(kind);

CREATE TABLE IF NOT EXISTS fp_hashes (
    hash     INTEGER NOT NULL,
    offset   INTEGER NOT NULL,
    track_id INTEGER NOT NULL REFERENCES fp_tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fp_hashes_hash ON fp_hashes(hash);
CREATE INDEX IF NOT EXISTS idx_fp_hashes_track ON fp_hashes(track_id);

CREATE TABLE IF NOT EXISTS song_meta (
    key         TEXT PRIMARY KEY,   -- catalog_key(artist, title)
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    year        INTEGER,
    duration_ms INTEGER,
    art_url     TEXT,               -- original remote artwork URL
    art_path    TEXT,               -- locally cached path we serve
    preview_url TEXT,
    source      TEXT,
    updated_ms  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


class ThreadLocalDB:
    """One SQLite connection per thread.

    Connections cannot cross threads, and FastAPI runs sync endpoints in a
    threadpool, so each thread gets its own. WAL mode lets these readers run
    concurrently with the ingest writer.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


# --- segments ---------------------------------------------------------------


def insert_segment(
    conn: sqlite3.Connection,
    *,
    pdt_ms: int,
    seq: int,
    duration_ms: int,
    relpath: str,
    size: int,
    fetched_ms: int,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO segments"
        " (pdt_ms, seq, duration_ms, relpath, bytes, fetched_ms)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (pdt_ms, seq, duration_ms, relpath, size, fetched_ms),
    )


def has_segment(conn: sqlite3.Connection, pdt_ms: int) -> bool:
    row = conn.execute("SELECT 1 FROM segments WHERE pdt_ms = ?", (pdt_ms,)).fetchone()
    return row is not None


def latest_segment(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM segments ORDER BY pdt_ms DESC LIMIT 1"
    ).fetchone()


def earliest_segment(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM segments ORDER BY pdt_ms ASC LIMIT 1").fetchone()


def segments_from(
    conn: sqlite3.Connection, start_ms: int, limit: int
) -> list[sqlite3.Row]:
    """The `limit` segments whose PDT is at or after `start_ms`."""
    return list(
        conn.execute(
            "SELECT * FROM segments WHERE pdt_ms >= ? ORDER BY pdt_ms ASC LIMIT ?",
            (start_ms, limit),
        )
    )


def segments_between(
    conn: sqlite3.Connection, start_ms: int, end_ms: int
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM segments WHERE pdt_ms >= ? AND pdt_ms < ? ORDER BY pdt_ms ASC",
            (start_ms, end_ms),
        )
    )


def segment_at(conn: sqlite3.Connection, pdt_ms: int) -> sqlite3.Row | None:
    """The segment covering `pdt_ms` (the latest one starting at or before it)."""
    return conn.execute(
        "SELECT * FROM segments WHERE pdt_ms <= ? ORDER BY pdt_ms DESC LIMIT 1",
        (pdt_ms,),
    ).fetchone()


def segment_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0])


def delete_segments_before(conn: sqlite3.Connection, pdt_ms: int) -> list[str]:
    rows = list(
        conn.execute("SELECT relpath FROM segments WHERE pdt_ms < ?", (pdt_ms,))
    )
    conn.execute("DELETE FROM segments WHERE pdt_ms < ?", (pdt_ms,))
    conn.execute("DELETE FROM gaps WHERE before_pdt_ms < ?", (pdt_ms,))
    return [row["relpath"] for row in rows]


# --- gaps -------------------------------------------------------------------


def record_gap(
    conn: sqlite3.Connection,
    *,
    after_pdt_ms: int,
    before_pdt_ms: int,
    missing_ms: int,
    reason: str,
    detected_ms: int,
) -> None:
    conn.execute(
        "INSERT INTO gaps (after_pdt_ms, before_pdt_ms, missing_ms, reason, detected_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        (after_pdt_ms, before_pdt_ms, missing_ms, reason, detected_ms),
    )


def gaps_between(
    conn: sqlite3.Connection, start_ms: int, end_ms: int
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM gaps WHERE after_pdt_ms >= ? AND after_pdt_ms < ?"
            " ORDER BY after_pdt_ms ASC",
            (start_ms, end_ms),
        )
    )


# --- meta -------------------------------------------------------------------


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


# --- timeline ---------------------------------------------------------------

TIMELINE_COLUMNS = (
    "start_ms", "end_ms", "kind", "title", "artist", "album", "year",
    "art_url", "show_title", "show_lead", "show_image", "confidence",
    "source", "cluster_id",
)


def upsert_timeline(conn: sqlite3.Connection, item: dict, created_ms: int) -> None:
    """Insert a timeline item, replacing anything already covering that span.

    The analyzer and the classifier both write here, so the clear-then-insert
    runs as one transaction; interleaved halves would leave a hole in the
    timeline or two rows claiming the same moment.
    """
    columns = ", ".join((*TIMELINE_COLUMNS, "created_ms"))
    placeholders = ", ".join("?" * (len(TIMELINE_COLUMNS) + 1))
    values = [item.get(name) for name in TIMELINE_COLUMNS] + [created_ms]
    with transaction(conn):
        conn.execute(
            "DELETE FROM timeline WHERE start_ms < ? AND end_ms > ?",
            (item["end_ms"], item["start_ms"]),
        )
        conn.execute(f"INSERT INTO timeline ({columns}) VALUES ({placeholders})", values)


def timeline_at(conn: sqlite3.Connection, at_ms: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM timeline WHERE start_ms <= ? AND end_ms > ? ORDER BY start_ms DESC LIMIT 1",
        (at_ms, at_ms),
    ).fetchone()


def timeline_after(conn: sqlite3.Connection, at_ms: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM timeline WHERE start_ms > ? ORDER BY start_ms ASC LIMIT 1",
        (at_ms,),
    ).fetchone()


def timeline_between(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM timeline WHERE end_ms > ? AND start_ms < ? ORDER BY start_ms ASC",
            (start_ms, end_ms),
        )
    )


def delete_timeline_before(conn: sqlite3.Connection, at_ms: int) -> None:
    conn.execute("DELETE FROM timeline WHERE end_ms < ?", (at_ms,))


def segments_upto(conn: sqlite3.Connection, at_ms: int, limit: int) -> list[sqlite3.Row]:
    """The `limit` most recent segments whose PDT is at or before `at_ms`, oldest first."""
    rows = list(
        conn.execute(
            "SELECT * FROM segments WHERE pdt_ms <= ? ORDER BY pdt_ms DESC LIMIT ?",
            (at_ms, limit),
        )
    )
    rows.reverse()
    return rows


# --- song metadata cache ----------------------------------------------------

SONG_META_COLUMNS = (
    "key", "title", "artist", "album", "year", "duration_ms",
    "art_url", "art_path", "preview_url", "source",
)


def upsert_song_meta(conn: sqlite3.Connection, row: dict, updated_ms: int) -> None:
    columns = ", ".join((*SONG_META_COLUMNS, "updated_ms"))
    placeholders = ", ".join("?" * (len(SONG_META_COLUMNS) + 1))
    updates = ", ".join(f"{name} = excluded.{name}" for name in SONG_META_COLUMNS[1:])
    values = [row.get(name) for name in SONG_META_COLUMNS] + [updated_ms]
    conn.execute(
        f"INSERT INTO song_meta ({columns}) VALUES ({placeholders})"
        f" ON CONFLICT(key) DO UPDATE SET {updates}, updated_ms = excluded.updated_ms",
        values,
    )


def get_song_meta(conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM song_meta WHERE key = ?", (key,)).fetchone()
