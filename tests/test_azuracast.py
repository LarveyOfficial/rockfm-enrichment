import shutil

import httpx
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
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload, self.content, self.status_code = payload, content, status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Api:
    """Stands in for AzuraCast, recording what we asked of it."""

    def __init__(self, media_id="77"):
        self.media_id, self.calls = media_id, []
        self._made = 0
        self.delete_status = 200
        self.delete_raises = False

    def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        if "/files" in url:
            self._made += 1
            return _Reply({"id": str(int(self.media_id) + self._made - 1)})
        return _Reply({})

    def put(self, url, **kw):
        self.calls.append(("put", url, kw))
        return _Reply({})

    def get(self, url, **kw):
        self.calls.append(("get", url, kw))
        return _Reply(content=b"remote-bytes")

    def delete(self, url, **kw):
        self.calls.append(("delete", url, kw))
        if self.delete_raises:
            raise httpx.HTTPError("connection reset")
        return _Reply({}, status_code=self.delete_status)

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


SONG_META = dict(SONG, start_ms=1_000_000, end_ms=1_204_108)


def _meta(**over):
    return metadata_for(dict(SONG_META, **over), "es")


def uploads(api):
    return [kw["json"]["path"] for m, url, kw in api.calls if m == "post" and "/files" in url]


def edits(api):
    return [kw["json"] for m, url, kw in api.calls if m == "put" and "/file/" in url]


def test_a_song_gets_a_record_of_its_own(tmp_path):
    carrier, api, _ = _carrier(tmp_path)
    assert carrier.record_for(_meta()) == "77"
    assert len(uploads(api)) == 1
    assert uploads(api)[0].startswith("rockfm/carriers/")
    assert edits(api)[0]["title"] == "Denis"
    assert edits(api)[0]["length"] == pytest.approx(204.108)


def test_asking_twice_for_one_airing_does_not_make_two_records(tmp_path):
    """We poll every two seconds; a song holds for minutes.

    Also what makes a restart mid-song safe: the key is derived, not counted,
    so the second ask finds the record the first one uploaded.
    """
    carrier, api, _ = _carrier(tmp_path)
    first = carrier.record_for(_meta())
    settled = len(api.calls)

    assert carrier.record_for(_meta()) == first
    assert len(api.calls) == settled, "touched AzuraCast again for a record it already had"


def test_the_same_song_played_again_gets_a_new_record(tmp_path):
    """Every airing is its own record, even when nothing about the song changed.

    A station cuts tracks differently each time, so a later airing may run to a
    different length -- but the reason holds even when it does not. Pointing
    AzuraCast at a record it has already seen reads as the same track still
    playing, and the elapsed time never returns to zero.
    """
    carrier, api, _ = _carrier(tmp_path)
    first = carrier.record_for(_meta())
    later = carrier.record_for(_meta(start_ms=9_000_000, end_ms=9_204_108))
    assert first != later
    assert len(uploads(api)) == 2


def test_two_songs_get_two_records(tmp_path):
    """Distinct ids are what tell AzuraCast the song actually changed."""
    carrier, api, _ = _carrier(tmp_path)
    one = carrier.record_for(_meta())
    two = carrier.record_for(_meta(title="Basket Case", artist="Green Day"))
    assert one != two
    assert len(uploads(api)) == 2


def test_the_same_song_cut_to_a_different_length_gets_its_own_record(tmp_path):
    """Stations edit tracks; a record states one length and is never rewritten."""
    carrier, _api, _ = _carrier(tmp_path)
    full = carrier.record_for(_meta())
    short = carrier.record_for(_meta(end_ms=1_150_000))
    assert full != short


def test_a_record_is_kept_across_restarts(tmp_path):
    """Remembered in the database, so a restart does not re-upload the library."""
    carrier, api, config = _carrier(tmp_path)
    carrier.record_for(_meta())

    from rockfm import db
    from rockfm.azuracast import MediaCarrier
    again = MediaCarrier(config, db.ThreadLocalDB(config.db_path))
    again.client = api
    assert again.record_for(_meta()) == "77"
    assert len(uploads(api)) == 1


def test_the_key_ignores_nothing_that_the_record_states():
    from rockfm.azuracast import Metadata, carrier_key

    base = Metadata(title="t", artist="a", album="b", duration=200.0, started_ms=5)
    same = Metadata(title="t", artist="a", album="b", duration=200.4, started_ms=5)
    assert carrier_key(base) == carrier_key(same), "the same airing asked twice"
    for different in (
        Metadata(title="other", artist="a", album="b", duration=200.0, started_ms=5),
        Metadata(title="t", artist="other", album="b", duration=200.0, started_ms=5),
        Metadata(title="t", artist="a", album="other", duration=200.0, started_ms=5),
        Metadata(title="t", artist="a", album="b", duration=170.0, started_ms=5),
        Metadata(title="t", artist="a", album="b", duration=200.0, started_ms=6),
    ):
        assert carrier_key(base) != carrier_key(different)


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


def test_a_new_record_gets_the_song_picture(tmp_path):
    carrier, api, config = _carrier(tmp_path)
    (config.art_dir / "abc.jpg").write_bytes(b"local-bytes")

    media_id = carrier.record_for(_meta())
    posted = [kw for m, url, kw in api.calls if m == "post" and f"/art/{media_id}" in url]
    assert len(posted) == 1
    assert posted[0]["files"]["file"][1] == b"local-bytes"


def test_the_same_picture_is_not_uploaded_twice(tmp_path):
    """A song holds for minutes and we poll every two seconds."""
    carrier, api, config = _carrier(tmp_path)
    (config.art_dir / "abc.jpg").write_bytes(b"local-bytes")

    carrier.record_for(_meta())
    carrier.record_for(_meta())
    assert len([1 for m, url, _ in api.calls if m == "post" and "/art/" in url]) == 1


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


# --- how long the carrier says it runs --------------------------------------
#
# AzuraCast reads the running time off the media record, not off the `duration`
# we push, so every song showed as 0:01 -- the shared carrier being one second
# of silence. Each record now states its own song's length, once.


class _Library(_Api):
    """AzuraCast, but with opinions about being told a track length."""

    def __init__(self, keeps=True, refuses=False):
        super().__init__()
        self.keeps, self.refuses, self.length = keeps, refuses, 1.0

    def put(self, url, **kw):
        if self.refuses and "length" in kw.get("json", {}):
            raise httpx.HTTPError("422 Unprocessable Entity")
        if self.keeps and "length" in kw.get("json", {}):
            self.length = kw["json"]["length"]
        return super().put(url, **kw)

    def get(self, url, **kw):
        if "/file/" in url:
            self.calls.append(("get", url, kw))
            return _Reply({"length": self.length})
        return super().get(url, **kw)


def _lengths_put(api):
    return [b["length"] for b in edits(api) if "length" in b]


def test_each_record_states_its_own_length(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library())
    carrier.record_for(_meta())
    carrier.record_for(_meta(title="Basket Case", end_ms=1_100_000))
    assert _lengths_put(api) == [pytest.approx(204.108), pytest.approx(100.0)]


def test_every_airing_is_recorded_so_its_record_can_be_found_again(tmp_path):
    """AzuraCast's library is otherwise the only place these records exist."""
    from rockfm import db
    carrier, _api, config = _carrier(tmp_path, _Library())
    carrier.record_for(_meta())
    carrier.record_for(_meta(start_ms=9_000_000, end_ms=9_204_108))
    assert db.count_carriers(db.ThreadLocalDB(config.db_path).conn) == 2


def test_a_song_of_unknown_length_says_nothing_about_it(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library())
    carrier.record_for(metadata_for(SONG, "es"))
    assert _lengths_put(api) == []


def test_a_refused_length_does_not_cost_us_the_artwork(tmp_path):
    """Losing the picture to gain a running time would be a bad trade."""
    carrier, api, _ = _carrier(tmp_path, _Library(refuses=True))
    assert carrier.record_for(_meta()) == "77"
    assert edits(api) and "length" not in edits(api)[-1]
    assert edits(api)[-1]["title"] == "Denis"


def test_a_length_that_does_not_stick_is_reported_but_still_sent(tmp_path):
    """One racy read must not switch the field off for every later song."""
    carrier, api, _ = _carrier(tmp_path, _Library(keeps=False))
    for title in ("Denis", "Another", "A third"):
        carrier.record_for(_meta(title=title))
    assert len(_lengths_put(api)) == 3, "stopped sending the length after reading it back"


def test_the_length_is_only_read_back_once(tmp_path):
    carrier, api, _ = _carrier(tmp_path, _Library())
    for title in ("Denis", "Another", "A third"):
        carrier.record_for(_meta(title=title))
    assert len([1 for m, url, _ in api.calls if m == "get" and "/file/" in url]) == 1


# --- the silence itself ------------------------------------------------------


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg renders the carrier")
def test_the_silence_runs_as_long_as_the_song():
    """So AzuraCast can read the duration off the audio, not only off a field."""
    from rockfm.audio import decode_bytes
    from rockfm.azuracast import _silent_mp3

    rendered = _silent_mp3(12.0)
    assert rendered
    heard = decode_bytes(rendered, rate=8000)
    assert heard.size / 8000 == pytest.approx(12.0, abs=0.5)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg renders the carrier")
def test_an_hours_long_programme_is_not_rendered_in_full():
    """Hours of silence is not worth uploading; the length still goes on record."""
    from rockfm.audio import decode_bytes
    from rockfm.azuracast import MAX_RENDER_SECONDS, _silent_mp3

    heard = decode_bytes(_silent_mp3(4 * 3600.0), rate=8000)
    assert heard.size / 8000 == pytest.approx(MAX_RENDER_SECONDS, abs=1.0)


# --- clearing up after ourselves ---------------------------------------------
#
# One record per airing fills a library at the rate songs are played, so each
# new one takes the chance to remove those that are neither playing now nor the
# one before. Only records we made: they are tracked in our own table, and
# nothing else in the station's library is looked at.


def _airings(carrier, how_many):
    return [
        carrier.record_for(_meta(start_ms=1_000_000 * n, end_ms=1_000_000 * n + 204_108))
        for n in range(1, how_many + 1)
    ]


def deleted(api):
    return [url.rsplit("/", 1)[-1] for m, url, _kw in api.calls if m == "delete"]


def test_nothing_is_removed_while_there_are_only_two(tmp_path):
    carrier, api, _ = _carrier(tmp_path)
    _airings(carrier, 2)
    assert deleted(api) == []


def test_the_one_before_the_previous_is_removed(tmp_path):
    """Keep what is playing and the one before it; let go of the rest."""
    carrier, api, config = _carrier(tmp_path)
    made = _airings(carrier, 3)

    assert deleted(api) == [made[0]]
    from rockfm import db
    conn = db.ThreadLocalDB(config.db_path).conn
    assert db.count_carriers(conn) == 2


def test_the_library_does_not_grow_with_the_day(tmp_path):
    carrier, api, config = _carrier(tmp_path)
    made = _airings(carrier, 12)

    from rockfm import db
    assert db.count_carriers(db.ThreadLocalDB(config.db_path).conn) == 2
    assert deleted(api) == made[:10], "removed the wrong records, or the wrong order"


def test_a_removal_that_fails_is_tried_again(tmp_path):
    """Forgetting a record we failed to delete would strand it in the library."""
    carrier, api, config = _carrier(tmp_path)
    api.delete_raises = True
    _airings(carrier, 3)

    from rockfm import db
    conn = db.ThreadLocalDB(config.db_path).conn
    assert db.count_carriers(conn) == 3, "forgot a record that is still there"

    api.delete_raises = False
    _airings(carrier, 4)
    assert db.count_carriers(db.ThreadLocalDB(config.db_path).conn) == 2


def test_a_record_already_gone_is_not_a_failure(tmp_path):
    """404 is the outcome we wanted, however it came about."""
    carrier, api, config = _carrier(tmp_path)
    api.delete_status = 404
    _airings(carrier, 3)

    from rockfm import db
    assert db.count_carriers(db.ThreadLocalDB(config.db_path).conn) == 2
