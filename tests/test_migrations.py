"""Opening an existing database must never lose what is in it.

`MIGRATIONS` is empty at the moment -- the columns it used to add belonged to
the fingerprint index, which no longer exists. The mechanism stays, because the
next schema change will need it, so it is tested with a migration of its own
rather than left unexercised until the day it matters.
"""

from __future__ import annotations

import sqlite3

import pytest

from rockfm import db


def test_a_fresh_database_opens_and_holds_the_schema(tmp_path):
    conn = db.connect(tmp_path / "new.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"segments", "timeline", "song_meta", "meta"} <= tables


def test_opening_twice_is_harmless(tmp_path):
    path = tmp_path / "x.db"
    db.connect(path).close()
    conn = db.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 0


def test_an_added_column_reaches_an_existing_database(tmp_path, monkeypatch):
    """The mechanism itself, driven by a migration invented for the test."""
    path = tmp_path / "old.db"
    db.connect(path).close()

    monkeypatch.setattr(db, "MIGRATIONS", (("timeline", "mood", "TEXT"),))
    conn = db.connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(timeline)")}
    assert "mood" in columns


def test_migrating_the_same_column_twice_is_harmless(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    db.connect(path).close()
    monkeypatch.setattr(db, "MIGRATIONS", (("timeline", "mood", "TEXT"),))

    db.connect(path).close()
    db.connect(path).close()          # would raise "duplicate column" if unguarded


def test_rows_written_before_a_migration_survive_it(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.upsert_timeline(
        conn, {"start_ms": 1_000, "end_ms": 2_000, "kind": "cancion",
               "title": "Denis", "artist": "Blondie"}, 0,
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "MIGRATIONS", (("timeline", "mood", "TEXT"),))
    conn = db.connect(path)
    row = conn.execute("SELECT * FROM timeline WHERE title = 'Denis'").fetchone()
    assert row["artist"] == "Blondie"
    assert row["mood"] is None


def test_a_migration_naming_a_table_that_is_gone_is_skipped(tmp_path, monkeypatch):
    """Tables do get removed -- the fingerprint index was two of them."""
    path = tmp_path / "old.db"
    db.connect(path).close()
    monkeypatch.setattr(db, "MIGRATIONS", (("departed", "column", "TEXT"),))

    try:
        db.connect(path).close()
    except sqlite3.OperationalError as exc:  # pragma: no cover - the failure case
        pytest.fail(f"a migration for a departed table stopped startup: {exc}")
