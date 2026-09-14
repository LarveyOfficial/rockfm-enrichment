"""What playout does when a restart leaves a hole in the recording.

Nothing was recorded for that stretch, so there is no audio to serve. The
failure that matters is serving the *wrong* audio: asking for the newest
segments at or before the playout position hands back whatever was recorded
before the outage, re-stamped to look current and re-served until the hole ends.
A restart would replay its last few seconds on a loop rather than admit the gap.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from rockfm import db
from rockfm.config import Config
from rockfm.playout import Playout, create_app

SEGMENT_MS = 6_000


@pytest.fixture
def env(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return config, conn


def record(conn, config, start_ms: int, count: int) -> None:
    for index in range(count):
        pdt = start_ms + index * SEGMENT_MS
        name = f"{pdt}.aac"
        (config.segments_dir / name).write_bytes(b"\xff\xf1" + b"\x00" * 32)
        db.insert_segment(
            conn, pdt_ms=pdt, seq=index, duration_ms=SEGMENT_MS,
            relpath=name, size=34, fetched_ms=0,
        )


def test_a_hole_serves_nothing_rather_than_stale_audio(env):
    config, conn = env
    now = int(time.time() * 1000)
    position = now - 600_000  # what playout is reaching for

    # Recorded up to five minutes before the position, then an outage, then
    # recording resumed after it.
    record(conn, config, position - 300_000, 20)
    record(conn, config, position + 120_000, 20)

    playout = Playout(config)
    assert playout.window() == [], "served audio from before the outage"

    state, resumes_in = playout.buffer_state()
    assert state == "gap"
    assert resumes_in == pytest.approx(120, abs=10)


def test_the_stream_says_so_instead_of_lying(env):
    config, conn = env
    now = int(time.time() * 1000)
    position = now - 600_000
    record(conn, config, position - 300_000, 20)
    record(conn, config, position + 120_000, 20)

    client = TestClient(create_app(config), raise_server_exceptions=False)
    response = client.get("/hls/chunks.m3u8")
    assert response.status_code == 503
    assert "recorded" in response.json()["detail"]


def test_a_gap_does_not_make_the_container_look_unhealthy(env):
    """Ingest is fine; the hole is history and nothing can refill it."""
    config, conn = env
    now = int(time.time() * 1000)
    position = now - 600_000
    record(conn, config, position - 300_000, 20)
    record(conn, config, now - 30_000, 5)  # still recording, right up to now

    health = Playout(config).health()
    assert health["state"] == "gap"
    assert health["ok"] is True
    assert health["playout_ready"] is False


def test_playout_resumes_once_the_position_reaches_real_audio(env):
    config, conn = env
    now = int(time.time() * 1000)
    position = now - 600_000
    record(conn, config, position - 60_000, 40)  # covers the position

    playout = Playout(config)
    assert playout.buffer_state()[0] == "ready"
    assert playout.window(), "should serve audio once the hole is behind us"
