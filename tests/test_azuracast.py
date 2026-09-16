import pytest

from rockfm.azuracast import metadata_for

SONG = {
    "kind": "cancion", "title": "Denis", "artist": "Blondie",
    "album": "Plastic Letters", "year": 1977, "art_url": "/art/abc.jpg",
    "show_title": None, "show_lead": None,
}
# A stretch between songs that the schedule had nothing to say about.
UNNAMED = {"kind": "programa", "art_url": None, "show_title": None, "show_lead": None}
SHOW = {
    "kind": "programa", "art_url": "/art/show.jpg",
    "show_title": "El Pirata y su banda",
    "show_lead": "El Pirata, Sayago, Álex Clavero y Raquel Piqueras",
}


def test_song_maps_to_real_fields():
    meta = metadata_for(SONG, "es")
    assert meta.title == "Denis"
    assert meta.artist == "Blondie"
    assert meta.album == "Plastic Letters"


def test_an_unnamed_stretch_falls_back_to_the_station():
    """No schedule entry, so the honest label is the station itself."""
    meta = metadata_for(UNNAMED, "es")
    assert meta.title == "RockFM"


def test_programme_shows_presenters_as_the_artist():
    meta = metadata_for(SHOW, "es")
    assert meta.title == "El Pirata y su banda"
    assert "Álex Clavero" in meta.artist


def test_relative_art_becomes_absolute_when_a_public_url_is_known():
    assert metadata_for(SONG, "es", "https://radio.example.com/").art == (
        "https://radio.example.com/art/abc.jpg"
    )


def test_relative_art_is_left_alone_without_a_public_url():
    assert metadata_for(SONG, "es").art == "/art/abc.jpg"


def test_params_omit_empty_fields():
    params = metadata_for(UNNAMED, "es").as_params()
    assert "album" not in params
    assert "art" not in params
    assert params["title"] == "RockFM"


# --- how long the song runs -------------------------------------------------
#
# AzuraCast timed every song at 0:01: with no duration of our own it fell back
# to the carrier media record, which is one second of silence. `duration` is in
# ALLOWED_ANNOTATIONS, so unlike `art` and `album` it does get through.

TIMED = dict(SONG, start_ms=1_000_000, end_ms=1_204_108)


def test_a_song_carries_its_length():
    assert metadata_for(TIMED, "es").duration == pytest.approx(204.108)


def test_the_length_is_sent_to_azuracast():
    assert metadata_for(TIMED, "es").as_params()["duration"] == "204.108"


def test_a_programme_is_timed_too():
    """Shows run for hours; 0:01 looked just as wrong there."""
    timed_show = dict(SHOW, start_ms=0, end_ms=3_600_000)
    assert metadata_for(timed_show, "es").as_params()["duration"] == "3600.000"


def test_a_row_with_no_span_sends_no_length():
    """Better to leave it out than to claim a length we do not have."""
    assert "duration" not in metadata_for(SONG, "es").as_params()


@pytest.mark.parametrize("end", [1_000_000, 900_000])
def test_a_span_that_does_not_move_forward_is_not_a_length(end):
    row = dict(SONG, start_ms=1_000_000, end_ms=end)
    assert metadata_for(row, "es").duration is None


# --- artwork rides on a media record ----------------------------------------


class _Reply:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Api:
    """Stands in for AzuraCast, recording what we asked of it."""

    def __init__(self, media_id="77"):
        self.media_id, self.calls = media_id, []

    def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        if "/files" in url:
            return _Reply({"id": self.media_id})
        return _Reply({})

    def put(self, url, **kw):
        self.calls.append(("put", url, kw))
        return _Reply({})

    def get(self, url, **kw):
        self.calls.append(("get", url, kw))
        return _Reply(content=b"remote-bytes")

    def close(self):
        return None


def _carrier(tmp_path, api=None):
    from rockfm import db, settings
    from rockfm.azuracast import MediaCarrier
    from rockfm.config import Config

    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)
    settings.save(database.conn, {
        "azuracast_enabled": True,
        "azuracast_base_url": "https://radio.example.com",
        "azuracast_station_id": "4",
        "azuracast_api_key": "key",
    })
    carrier = MediaCarrier(config, database)
    carrier.client = api or _Api()
    return carrier, carrier.client, config


def test_the_carrier_record_is_created_once_and_remembered(tmp_path):
    """Uploading a placeholder on every song change would be absurd."""
    carrier, api, _ = _carrier(tmp_path)

    first = carrier.media_id()
    uploads = [c for c in api.calls if "/files" in c[1]]
    assert first == "77" and len(uploads) == 1

    assert carrier.media_id() == "77"
    assert len([c for c in api.calls if "/files" in c[1]]) == 1, "uploaded twice"


def test_our_own_artwork_is_read_from_disk_not_fetched(tmp_path):
    from rockfm.azuracast import Metadata

    carrier, api, config = _carrier(tmp_path)
    (config.art_dir / "abc.jpg").write_bytes(b"local-bytes")

    art = carrier.artwork(Metadata(title="t", artist="a",
                                   art="http://host:6967/art/abc.jpg"))
    assert art == b"local-bytes"
    assert not [c for c in api.calls if c[0] == "get"], "went to the network for a local file"


def test_artwork_we_do_not_hold_is_fetched(tmp_path):
    from rockfm.azuracast import Metadata

    carrier, api, _ = _carrier(tmp_path)
    art = carrier.artwork(Metadata(title="t", artist="a",
                                   art="https://www.rockfm.fm/programme.jpg"))
    assert art == b"remote-bytes"
    assert [c for c in api.calls if c[0] == "get"]


def test_publishing_sets_the_picture_and_the_name(tmp_path):
    from rockfm.azuracast import Metadata

    carrier, api, config = _carrier(tmp_path)
    (config.art_dir / "abc.jpg").write_bytes(b"local-bytes")
    meta = Metadata(title="Denis", artist="Blondie", album="Plastic Letters",
                    art="http://host:6967/art/abc.jpg")

    assert carrier.publish("77", meta) is True

    art_posts = [c for c in api.calls if "/art/77" in c[1]]
    assert len(art_posts) == 1
    assert art_posts[0][2]["files"]["file"][1] == b"local-bytes"

    edits = [c for c in api.calls if c[0] == "put" and "/file/77" in c[1]]
    assert len(edits) == 1
    assert edits[0][2]["json"] == {"title": "Denis", "artist": "Blondie",
                                   "album": "Plastic Letters"}


def test_the_same_picture_is_not_uploaded_twice(tmp_path):
    """A song holds for minutes and we poll every two seconds."""
    from rockfm.azuracast import Metadata

    carrier, api, config = _carrier(tmp_path)
    (config.art_dir / "abc.jpg").write_bytes(b"local-bytes")
    meta = Metadata(title="Denis", artist="Blondie",
                    art="http://host:6967/art/abc.jpg")

    carrier.publish("77", meta)
    carrier.publish("77", meta)
    assert len([c for c in api.calls if "/art/77" in c[1]]) == 1


def test_the_push_names_the_media_record(tmp_path):
    from rockfm import db, settings
    from rockfm.azuracast import Metadata, MetadataClient
    from rockfm.config import Config

    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)
    settings.save(database.conn, {
        "azuracast_enabled": True,
        "azuracast_base_url": "https://radio.example.com",
        "azuracast_station_id": "4",
        "azuracast_api_key": "key",
    })
    client = MetadataClient(config, database)
    api = _Api()
    client.client = api

    assert client.push(Metadata(title="Denis", artist="Blondie"), media_id="77") is True
    params = api.calls[-1][2]["params"]
    assert params["media_id"] == "77"
    assert params["title"] == "Denis"


# --- the push waits for the audio to catch up --------------------------------


def _bridge(tmp_path):
    from rockfm import db, settings
    from rockfm.azuracast import MetadataBridge
    from rockfm.config import Config

    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)
    settings.save(database.conn, {
        "azuracast_enabled": True,
        "azuracast_base_url": "https://radio.example.com",
        "azuracast_station_id": "4",
        "azuracast_api_key": "key",
    })
    for start, end, title in ((0, 100_000, "Denis"), (100_000, 200_000, "Basket Case")):
        db.upsert_timeline(database.conn, {
            "start_ms": start, "end_ms": end, "kind": "cancion",
            "title": title, "artist": "somebody",
        }, 0)
    database.conn.commit()
    bridge = MetadataBridge(config, database)
    bridge.playout_position = lambda: 150_000        # airing the second song
    return bridge


def test_without_a_measurement_the_bridge_uses_our_own_position(tmp_path):
    assert _bridge(tmp_path).current()["title"] == "Basket Case"


def test_the_push_steps_back_by_the_measured_delay(tmp_path):
    """AzuraCast is still on the previous song, so that is what to announce."""
    bridge = _bridge(tmp_path)
    bridge.lag.record(60_000)
    assert bridge.current()["title"] == "Denis"


def test_a_small_delay_does_not_overshoot_the_song(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.lag.record(20_000)
    assert bridge.current()["title"] == "Basket Case"


# --- how long the carrier claims to be ---------------------------------------
#
# AzuraCast reads the running time off the media record, not off the `duration`
# we push, so every song showed as 0:01 -- the carrier's own second of silence.


class _Library(_Api):
    """AzuraCast, but with opinions about being told a track length.

    `keeps` False is the quiet refusal: the PUT succeeds and the record is
    unchanged, which is indistinguishable from success without reading it back.
    """

    def __init__(self, keeps=True, refuses=False):
        super().__init__()
        self.keeps, self.refuses, self.length = keeps, refuses, 1.0

    def put(self, url, **kw):
        if self.refuses and "length" in kw.get("json", {}):
            raise __import__("httpx").HTTPError("422 Unprocessable Entity")
        if self.keeps and "length" in kw.get("json", {}):
            self.length = kw["json"]["length"]
        return super().put(url, **kw)

    def get(self, url, **kw):
        if "/file/" in url:
            self.calls.append(("get", url, kw))
            return _Reply({"length": self.length})
        return super().get(url, **kw)


def _lengths_put(api):
    return [kw["json"]["length"] for _m, url, kw in api.calls
            if _m == "put" and "/file/" in url and "length" in kw.get("json", {})]


TIMED_SONG = dict(SONG, start_ms=1_000_000, end_ms=1_204_108)


def test_the_carrier_is_told_how_long_the_song_runs(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library())
    carrier.publish("77", metadata_for(TIMED_SONG, "es"))
    assert _lengths_put(api) == [pytest.approx(204.108)]


def test_a_song_of_unknown_length_says_nothing_about_it(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library())
    carrier.publish("77", metadata_for(SONG, "es"))
    assert _lengths_put(api) == []


def test_a_refused_length_does_not_cost_us_the_artwork(tmp_path):
    """Losing the picture to gain a running time would be a bad trade."""
    carrier, api, _ = _carrier(tmp_path, _Library(refuses=True))
    assert carrier.publish("77", metadata_for(TIMED_SONG, "es")) is True
    described = [kw["json"] for _m, url, kw in api.calls if _m == "put" and "/file/" in url]
    assert described and "length" not in described[-1]
    assert described[-1]["title"] == "Denis"


def test_a_refusal_is_remembered_rather_than_retried_every_song(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library(refuses=True))
    carrier.publish("77", metadata_for(TIMED_SONG, "es"))
    before = len(api.calls)
    carrier.publish("77", metadata_for(dict(TIMED_SONG, title="Another"), "es"))
    assert _lengths_put(api) == [], "kept offering a length after it was refused"
    assert len(api.calls) > before


def test_a_length_that_quietly_does_not_stick_is_noticed(tmp_path):
    """A 200 that changes nothing looks exactly like success until you look."""
    carrier, api, _ = _carrier(tmp_path, _Library(keeps=False))
    carrier.publish("77", metadata_for(TIMED_SONG, "es"))
    assert carrier._length_kept is False
    carrier.publish("77", metadata_for(dict(TIMED_SONG, title="Another"), "es"))
    assert len(_lengths_put(api)) == 1, "kept sending a length that never stuck"


def test_a_length_that_sticks_is_trusted_from_then_on(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library(keeps=True))
    carrier.publish("77", metadata_for(TIMED_SONG, "es"))
    assert carrier._length_kept is True
    carrier.publish("77", metadata_for(dict(TIMED_SONG, title="Another"), "es"))
    assert len(_lengths_put(api)) == 2
    reads = [1 for _m, url, _kw in api.calls if _m == "get" and "/file/" in url]
    assert len(reads) == 1, "read the record back more than once"
