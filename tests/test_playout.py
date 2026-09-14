import pytest
from fastapi.testclient import TestClient

from rockfm.config import Config
from rockfm.playout import create_app


@pytest.fixture
def client(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=0)
    config.ensure_dirs()
    (config.art_dir / "abc123.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return TestClient(create_app(config), raise_server_exceptions=False)


def test_art_is_served_from_the_cache(client):
    response = client.get("/art/abc123.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_missing_art_is_a_404_not_a_crash(client):
    assert client.get("/art/deadbeef.jpg").status_code == 404


@pytest.mark.parametrize(
    "name", ["../rockfm.db", "..%2Frockfm.db", "a/b.jpg", "evil.png", "abc123.jpg.txt"]
)
def test_art_route_refuses_anything_but_a_cached_name(client, name):
    assert client.get(f"/art/{name}").status_code == 404


def test_playlist_is_unavailable_until_the_buffer_fills(client):
    assert client.get("/hls/chunks.m3u8").status_code == 503


def test_health_is_ok_while_the_buffer_is_still_filling(client):
    """A six-hour fill is the design working, not a sick container."""
    response = client.get("/api/health")
    payload = response.json()
    assert payload["state"] == "empty"
    assert payload["playout_ready"] is False
    # Nothing has been recorded at all here, so ingest really is down.
    assert payload["ok"] is False


def test_nowplaying_is_valid_even_with_nothing_recorded(client):
    payload = client.get("/api/nowplaying").json()
    assert payload["now_playing"]["kind"] == "desconocido"
    assert payload["now_playing"]["primary"] == "RockFM"
    assert payload["buffer"]["segments"] == 0


def test_master_playlist_points_at_the_media_playlist(client):
    body = client.get("/hls/playlist.m3u8").text
    assert "#EXTM3U" in body
    assert "chunks.m3u8" in body


def test_player_page_renders(client):
    assert "RockFM" in client.get("/").text


def test_nowplaying_reports_the_source_timezone(client):
    payload = client.get("/api/nowplaying").json()
    assert payload["source_timezone"] == "Europe/Madrid"
    assert payload["source_time"].endswith(("+02:00", "+01:00"))


def test_filling_is_reported_as_a_state_with_a_countdown(tmp_path):
    """Recording, but the delay has not elapsed yet: healthy, not ready."""
    import time

    from rockfm import db as database
    from rockfm.playout import Playout

    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    playout = Playout(config)
    now_ms = int(time.time() * 1000)
    for index in range(5):
        database.insert_segment(
            playout.conn, pdt_ms=now_ms - 30_000 + index * 6000, seq=index,
            duration_ms=6000, relpath=f"{index}.aac", size=1, fetched_ms=0,
        )

    state, remaining = playout.buffer_state()
    assert state == "filling"
    # Six hours of delay, half a minute recorded: just under six hours to go.
    assert 6 * 3600 - 120 < remaining <= 6 * 3600

    health = playout.health()
    assert health["ok"] is True
    assert health["playout_ready"] is False
    assert health["state"] == "filling"
