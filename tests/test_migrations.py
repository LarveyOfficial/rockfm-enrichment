"""Upgrading an existing database must pick up later columns.

CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
without an explicit migration an upgrade keeps the old shape and the new code
fails on a missing column. This builds a database at the old shape and checks
connect() brings it forward.
"""

from __future__ import annotations

import sqlite3

from rockfm import db


def test_connect_adds_columns_to_an_existing_database(tmp_path):
    path = tmp_path / "old.db"
    # A database as it existed before song_anchored and learned_ms.
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE fp_tracks (
            id INTEGER PRIMARY KEY,
            kind TEXT NOT NULL,
            key TEXT NOT NULL UNIQUE,
            title TEXT, artist TEXT, source TEXT,
            occurrences INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER, anchor_ms INTEGER,
            created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL
        );
        """
    )
    old.execute(
        "INSERT INTO fp_tracks (kind, key, title, occurrences, created_ms, updated_ms)"
        " VALUES ('music', 'a|b', 'Denis', 3, 0, 0)"
    )
    old.commit()
    old.close()

    conn = db.connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(fp_tracks)")}
    assert {"song_anchored", "learned_ms"} <= columns

    # Existing rows survive, with sane defaults for the new columns.
    row = conn.execute("SELECT * FROM fp_tracks WHERE key = 'a|b'").fetchone()
    assert row["title"] == "Denis"
    assert row["occurrences"] == 3
    assert row["song_anchored"] == 0
    assert row["learned_ms"] is None


def test_migrating_twice_is_harmless(tmp_path):
    path = tmp_path / "x.db"
    db.connect(path).close()
    conn = db.connect(path)
    assert "song_anchored" in {r["name"] for r in conn.execute("PRAGMA table_info(fp_tracks)")}


def test_a_fresh_database_already_has_every_column(tmp_path):
    conn = db.connect(tmp_path / "new.db")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(fp_tracks)")}
    for _table, column, _definition in db.MIGRATIONS:
        assert column in columns
