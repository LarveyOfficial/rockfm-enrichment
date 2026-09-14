"""Configurable presentation for everything that is not a song."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rockfm import appearance, db, labels
from rockfm.config import Config
from rockfm.playout import create_app


@pytest.fixture
def conn(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    return db.connect(config.db_path)


@pytest.fixture
def client(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=0)
    config.ensure_dirs()
    return TestClient(create_app(config), raise_server_exceptions=False)


def test_defaults_are_returned_when_nothing_is_configured(conn):
    current = appearance.load(conn)
    assert current["publicidad"]["title"] == "Publicidad"
    assert current["desconocido"]["title"] == "RockFM"
    # Presenter talk is left blank on purpose: the real show name is better.
    assert current["programa"]["title"] == ""


def test_gaps_are_not_configurable():
    assert "cancion" not in appearance.CONFIGURABLE
    assert set(appearance.CONFIGURABLE) == {"publicidad", "programa", "desconocido"}


def test_saving_merges_and_persists(conn):
    appearance.save(conn, {"publicidad": {"title": "Anuncios"}})
    current = appearance.load(conn)
    assert current["publicidad"]["title"] == "Anuncios"
    # Untouched fields keep their defaults.
    assert current["publicidad"]["artist"] == "Volvemos enseguida"


def test_unknown_kinds_and_fields_are_ignored(conn):
    appearance.save(conn, {"cancion": {"title": "nope"}, "publicidad": {"bogus": "x"}})
    current = appearance.load(conn)
    assert "cancion" not in current
    assert "bogus" not in current["publicidad"]


def test_corrupt_stored_json_falls_back_to_defaults(conn):
    db.set_meta(conn, appearance.META_KEY, "{not json")
    assert appearance.load(conn)["publicidad"]["title"] == "Publicidad"


def test_configured_values_override_what_is_displayed(conn):
    look = appearance.save(
        conn,
        {"publicidad": {"title": "Anuncios", "artist": "Ya volvemos", "art": "/art/ad.jpg"}},
    )
    primary, secondary, art = labels.present({"kind": "publicidad"}, "es", look)
    assert (primary, secondary, art) == ("Anuncios", "Ya volvemos", "/art/ad.jpg")


def test_a_blank_setting_falls_back_to_what_we_know(conn):
    look = appearance.save(conn, {"programa": {"title": "", "artist": "", "art": ""}})
    row = {
        "kind": "programa", "show_title": "El Pirata y su banda",
        "show_lead": "El Pirata", "art_url": "/art/show.jpg",
    }
    assert labels.present(row, "es", look) == (
        "El Pirata y su banda", "El Pirata", "/art/show.jpg"
    )


def test_songs_are_never_overridden(conn):
    look = appearance.save(conn, {"desconocido": {"title": "nope", "art": "/art/x.jpg"}})
    row = {
        "kind": "cancion", "artist": "Blondie", "title": "Denis",
        "album": "Plastic Letters", "year": 1977, "art_url": "/art/denis.jpg",
    }
    assert labels.present(row, "es", look) == (
        "Blondie — Denis", "Plastic Letters · 1977", "/art/denis.jpg"
    )


def test_rows_from_versions_with_more_kinds_still_render(conn):
    """'noticias' and 'sintonia' no longer exist but may sit in an old database."""
    for stale in ("noticias", "sintonia"):
        primary, _, _ = labels.present({"kind": stale, "show_title": "Whatever"}, "es", {})
        assert primary == "RockFM"


# --- API --------------------------------------------------------------------


def test_get_appearance_reports_defaults_and_current(client):
    payload = client.get("/api/appearance").json()
    assert set(payload["configurable"]) == {"publicidad", "programa", "desconocido"}
    assert payload["fields"] == ["title", "artist", "art"]
    assert payload["current"]["publicidad"]["title"] == "Publicidad"


def test_put_appearance_saves_and_survives_a_reread(client):
    response = client.put("/api/appearance", json={"desconocido": {"title": "Rock FM 101"}})
    assert response.status_code == 200
    assert response.json()["current"]["desconocido"]["title"] == "Rock FM 101"
    assert client.get("/api/appearance").json()["current"]["desconocido"]["title"] == "Rock FM 101"


def test_put_appearance_rejects_nonsense(client):
    assert client.put("/api/appearance", content=b"not json").status_code == 400
    assert client.put("/api/appearance", json=["a", "list"]).status_code == 400


def test_configured_appearance_reaches_the_now_playing_api(client, tmp_path):
    import time

    conn = db.connect(tmp_path / "rockfm.db")
    now = int(time.time() * 1000)
    db.upsert_timeline(
        conn,
        {"start_ms": now - 30_000, "end_ms": now + 30_000, "kind": "publicidad"},
        0,
    )
    client.put("/api/appearance", json={"publicidad": {"title": "Anuncios", "art": "/art/ad.jpg"}})
    now_playing = client.get("/api/nowplaying").json()["now_playing"]
    assert now_playing["primary"] == "Anuncios"
    assert now_playing["art"] == "/art/ad.jpg"
