"""The dashboard's read-only API: timeline, pipeline status, on-demand replay."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from rockfm import db
from rockfm.config import Config
from rockfm.playout import create_app


@pytest.fixture
def env(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=0)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    client = TestClient(create_app(config), raise_server_exceptions=False)
    return config, conn, client


def record(conn, start_ms: int, count: int) -> None:
    for index in range(count):
        db.insert_segment(
            conn, pdt_ms=start_ms + index * 6000, seq=1000 + index, duration_ms=6000,
            relpath=f"{index}.aac", size=1, fetched_ms=0,
        )


def test_timeline_is_empty_but_valid_before_anything_is_analyzed(env):
    _, _, client = env
    payload = client.get("/api/timeline").json()
    assert payload["items"] == []
    assert payload["range"]["start"] < payload["range"]["end"]


def test_timeline_returns_items_with_display_labels(env):
    _, conn, client = env
    now = int(time.time() * 1000)
    record(conn, now - 300_000, 50)
    db.upsert_timeline(
        conn,
        {"start_ms": now - 200_000, "end_ms": now - 20_000, "kind": "cancion",
         "title": "Denis", "artist": "Blondie", "album": "Plastic Letters",
         "year": 1977, "source": "shazamio", "confidence": 0.9},
        0,
    )
    items = client.get("/api/timeline?hours=1").json()["items"]
    assert len(items) == 1
    assert items[0]["primary"] == "Blondie — Denis"
    assert items[0]["secondary"] == "Plastic Letters · 1977"
    assert items[0]["duration"] == pytest.approx(180.0)
    assert items[0]["source"] == "shazamio"


def test_timeline_honours_an_explicit_range(env):
    _, conn, client = env
    db.upsert_timeline(conn, {"start_ms": 1_000_000, "end_ms": 1_060_000, "kind": "cancion"}, 0)
    assert len(client.get("/api/timeline?start=900000&end=1100000").json()["items"]) == 1
    assert client.get("/api/timeline?start=2000000&end=2100000").json()["items"] == []


def test_status_reports_the_pipeline(env):
    _, conn, client = env
    now = int(time.time() * 1000)
    record(conn, now - 120_000, 20)
    db.set_meta(conn, db.ANALYZER_CURSOR_KEY, str(now - 60_000))

    payload = client.get("/api/status").json()
    assert payload["counts"]["segments"] == 20
    assert payload["buffer"]["ingest_lag_seconds"] < 120
    # The analyzer is a minute behind the newest recorded audio.
    assert payload["analyzer"]["behind_live_seconds"] == pytest.approx(60, abs=10)
    assert payload["gaps"] == []


def test_status_surfaces_recorded_gaps(env):
    _, conn, client = env
    now = int(time.time() * 1000)
    record(conn, now - 120_000, 5)
    db.record_gap(
        conn, after_pdt_ms=now - 100_000, before_pdt_ms=now - 70_000,
        missing_ms=30_000, reason="missed_segments", detected_ms=0,
    )
    gaps = client.get("/api/status").json()["gaps"]
    assert len(gaps) == 1
    assert gaps[0]["missing_seconds"] == 30.0


def test_replay_is_a_complete_seekable_playlist(env):
    _, conn, client = env
    start = 1_700_000_000_000
    record(conn, start, 10)
    body = client.get(f"/replay.m3u8?start={start}&duration=60").text
    assert "#EXT-X-PLAYLIST-TYPE:VOD" in body
    assert body.rstrip().endswith("#EXT-X-ENDLIST")
    assert body.count("#EXTINF") == 10


def test_replay_marks_a_discontinuity_where_audio_is_missing(env):
    _, conn, client = env
    start = 1_700_000_000_000
    record(conn, start, 3)
    record(conn, start + 60_000, 3)  # 42s hole
    body = client.get(f"/replay.m3u8?start={start}&duration=120").text
    assert "#EXT-X-DISCONTINUITY" in body


def test_replay_404s_when_nothing_was_recorded_then(env):
    _, _, client = env
    assert client.get("/replay.m3u8?start=1700000000000&duration=60").status_code == 404


def test_dashboard_page_renders(env):
    _, _, client = env
    body = client.get("/dashboard").text
    assert "RockFM pipeline" in body
    assert "/api/status" in body
