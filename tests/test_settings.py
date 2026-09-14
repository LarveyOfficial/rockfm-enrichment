"""Runtime settings: editable without a restart, secrets never echoed back."""

from __future__ import annotations

import pytest

from rockfm import db, settings
from rockfm.config import Config


@pytest.fixture
def conn(tmp_path, monkeypatch):
    for _name, (_default, env, _kind) in settings.FIELDS.items():
        if env:
            monkeypatch.delenv(env, raising=False)
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    return db.connect(config.db_path)


def test_defaults_when_nothing_is_stored(conn):
    values = settings.load(conn)
    assert values["azuracast_enabled"] is False
    assert values["display_language"] == "es"
    assert values["min_nonmusic_seconds"] == 5.0


def test_environment_seeds_a_setting(conn, monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "http://tower.local:6967")
    assert settings.load(conn)["public_url"] == "http://tower.local:6967"


def test_configuring_azuracast_by_environment_turns_it_on(conn, monkeypatch):
    monkeypatch.setenv("AZURACAST_BASE_URL", "https://azuracast.example.com")
    monkeypatch.setenv("AZURACAST_API_KEY", "abc123")
    assert settings.load(conn)["azuracast_enabled"] is True


def test_saved_values_win_over_the_environment(conn, monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "http://from-env")
    settings.save(conn, {"public_url": "http://from-dashboard"})
    assert settings.load(conn)["public_url"] == "http://from-dashboard"


def test_types_are_coerced(conn):
    values = settings.save(
        conn, {"azuracast_enabled": "true", "min_nonmusic_seconds": "12"}
    )
    assert values["azuracast_enabled"] is True
    assert values["min_nonmusic_seconds"] == 12.0


def test_nonsense_falls_back_to_the_default(conn):
    assert settings.save(conn, {"min_nonmusic_seconds": "banana"})["min_nonmusic_seconds"] == 5.0


def test_unknown_keys_are_ignored(conn):
    assert "nope" not in settings.save(conn, {"nope": 1})


def test_secrets_are_masked_on_the_way_out(conn):
    settings.save(conn, {"azuracast_api_key": "supersecret"})
    shown = settings.public(settings.load(conn))
    assert shown["azuracast_api_key"] == settings.MASK
    assert "supersecret" not in str(shown)


def test_an_unset_secret_is_not_masked(conn):
    assert settings.public(settings.load(conn))["azuracast_api_key"] == ""


def test_sending_the_mask_back_does_not_overwrite_the_secret(conn):
    """The browser never sees the real value, so it cannot echo it."""
    settings.save(conn, {"azuracast_api_key": "supersecret"})
    settings.save(conn, {"azuracast_api_key": settings.MASK, "public_url": "http://x"})
    assert settings.load(conn)["azuracast_api_key"] == "supersecret"
    assert settings.load(conn)["public_url"] == "http://x"


def test_an_empty_secret_is_treated_as_leave_alone(conn):
    settings.save(conn, {"azuracast_api_key": "supersecret"})
    settings.save(conn, {"azuracast_api_key": ""})
    assert settings.load(conn)["azuracast_api_key"] == "supersecret"


def test_corrupt_stored_settings_fall_back_to_defaults(conn):
    db.set_meta(conn, settings.META_KEY, "{not json")
    assert settings.load(conn)["display_language"] == "es"


# --- the AzuraCast toggle ---------------------------------------------------


def test_turning_azuracast_off_hands_the_station_back_its_identity(tmp_path, monkeypatch):
    """Otherwise AzuraCast freezes on whatever song we last pushed."""
    from rockfm.azuracast import MetadataBridge, station_metadata
    from rockfm.config import Config

    for _name, (_default, env, _kind) in settings.FIELDS.items():
        if env:
            monkeypatch.delenv(env, raising=False)

    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)
    settings.save(
        database.conn,
        {
            "azuracast_enabled": True,
            "azuracast_base_url": "https://azuracast.example.com",
            "azuracast_station_id": "1",
            "azuracast_api_key": "k",
        },
    )

    bridge = MetadataBridge(config, database)
    pushed: list = []
    bridge.client.push = lambda m: (pushed.append(("push", m)), True)[1]
    bridge.client.push_regardless = lambda m: (pushed.append(("farewell", m)), True)[1]

    bridge.tick()  # enabled, nothing airing
    assert bridge._was_enabled is True

    settings.save(database.conn, {"azuracast_enabled": False})
    bridge.tick()

    assert [kind for kind, _ in pushed] == ["farewell"]
    sent = pushed[0][1]
    assert sent == station_metadata()
    assert sent.title == "RockFM"
    assert sent.artist == ""
    assert sent.art and "rockfm.fm" in sent.art

    pushed.clear()
    bridge.tick()
    assert pushed == [], "kept pushing after being turned off"


def test_the_farewell_is_only_sent_once(tmp_path, monkeypatch):
    from rockfm.azuracast import MetadataBridge
    from rockfm.config import Config

    for _name, (_default, env, _kind) in settings.FIELDS.items():
        if env:
            monkeypatch.delenv(env, raising=False)
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)

    bridge = MetadataBridge(config, database)
    sent: list = []
    bridge.client.push_regardless = lambda m: (sent.append(m), True)[1]

    # Never enabled: there is nothing to hand back.
    bridge.tick()
    bridge.tick()
    assert sent == []
