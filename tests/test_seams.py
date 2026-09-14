"""Crossfades leave a seam between songs that belongs to neither.

Each edge is located independently, so a few seconds can fall between the end of
one song and the start of the next. It is too short to be an advert or anything
else worth naming, but long enough that a player keeps showing the previous song
through it -- the metadata bridge only pushes when the item under the playout
position changes, and with no item there it pushes nothing.
"""

from __future__ import annotations

import pytest

from rockfm import db, settings
from rockfm.analyzer import Analyzer
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer


@pytest.fixture
def analyzer(tmp_path, monkeypatch):
    for _name, (_default, env, _kind) in settings.FIELDS.items():
        if env:
            monkeypatch.delenv(env, raising=False)
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


def place(conn, start_ms, end_ms, title="A"):
    db.upsert_timeline(
        conn,
        {"start_ms": start_ms, "end_ms": end_ms, "kind": "cancion",
         "title": title, "artist": "x"},
        0,
    )


def test_a_short_seam_is_split_down_the_middle(analyzer):
    settings.save(analyzer.conn, {"max_seam_seconds": 10})
    place(analyzer.conn, 0, 100_000)
    # The next song starts 6.5s later -- the crossfade the edges disagreed over.
    assert analyzer._close_seam(106_500) == 103_250

    previous = db.timeline_at(analyzer.conn, 102_000)
    assert previous["end_ms"] == 103_250, "the earlier song should have been extended"


def test_the_two_songs_end_up_touching(analyzer):
    settings.save(analyzer.conn, {"max_seam_seconds": 10})
    place(analyzer.conn, 0, 100_000)
    start = analyzer._close_seam(106_500)
    previous = db.previous_timeline(analyzer.conn, start + 1)
    assert previous["end_ms"] == start, "a gap survived"


def test_a_real_break_is_left_alone(analyzer):
    """Longer than the threshold is an advert break, not a seam."""
    settings.save(analyzer.conn, {"max_seam_seconds": 10})
    place(analyzer.conn, 0, 100_000)
    assert analyzer._close_seam(160_000) == 160_000
    assert db.timeline_at(analyzer.conn, 99_000)["end_ms"] == 100_000


def test_the_threshold_follows_its_own_setting(analyzer):
    settings.save(analyzer.conn, {"max_seam_seconds": 10})
    place(analyzer.conn, 0, 100_000)
    assert analyzer._close_seam(108_000) != 108_000   # 8s seam, under 10

    settings.save(analyzer.conn, {"max_seam_seconds": 5})
    place(analyzer.conn, 200_000, 300_000)
    assert analyzer._close_seam(308_000) == 308_000   # 8s seam, over 5


def test_seam_closing_is_independent_of_the_non_music_floor(analyzer):
    """They were one dial doing two jobs; tuning one must not move the other."""
    settings.save(analyzer.conn, {"max_seam_seconds": 12, "min_nonmusic_seconds": 2})
    place(analyzer.conn, 0, 100_000)
    assert analyzer._close_seam(108_000) == 104_000


def test_nothing_to_meet_leaves_the_start_alone(analyzer):
    assert analyzer._close_seam(50_000) == 50_000


def test_an_overlap_is_not_treated_as_a_seam(analyzer):
    place(analyzer.conn, 0, 100_000)
    assert analyzer._close_seam(95_000) == 95_000
