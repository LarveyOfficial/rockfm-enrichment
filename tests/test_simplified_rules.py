"""The two rules that replaced the classifier.

Adverts, jingles and presenter talk used to be told apart by a CNN and a
repetition index. They are not distinguishable in practice, and the attempt
cost a TensorFlow dependency, a background process and a great deal of time.
What is left is duration: a song has to have played, and a gap between songs
either belongs to the songs around it or is the programme that was on air.
"""

from __future__ import annotations

from rockfm import db, settings
from rockfm import strings_es as S
from rockfm.analyzer import MIN_SONG_MS, Analyzer, Label, Run
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer

RATE = 8000


def build(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


def _run(key="song", first=0, last=100_000):
    label = Label(key=key, artist="a", title="t", source="s", confidence=1.0, track_id=1)
    return Run(key=key, label=label, first_ms=first, last_ms=last)


# --- a song has to have played ---------------------------------------------


def test_a_snippet_of_a_song_does_not_count_as_playing_it(tmp_path):
    """A track trailed over a link matches exactly as well as the real airing.

    It is the same recording, so no amount of fingerprinting separates them.
    Length does.
    """
    analyzer = build(tmp_path)
    db.upsert_song_meta(analyzer.conn, {"key": "song", "duration_ms": 200_000}, 0)

    assert not analyzer._plays_long_enough("song", 20_000)
    assert analyzer._plays_long_enough("song", 40_000)


def test_an_unknown_length_falls_back_to_a_flat_floor(tmp_path):
    analyzer = build(tmp_path)
    assert not analyzer._plays_long_enough("never-heard-of-it", MIN_SONG_MS - 1)
    assert analyzer._plays_long_enough("never-heard-of-it", MIN_SONG_MS + 1)


def test_nothing_is_not_a_song(tmp_path):
    assert not build(tmp_path)._plays_long_enough(None, 10 ** 9)


# --- the gaps between songs -------------------------------------------------


def test_a_gap_long_enough_to_name_becomes_the_programme(tmp_path):
    analyzer = build(tmp_path)
    analyzer._programme = lambda _at: type(
        "P", (), {"title": "El Pirata y su banda", "lead": "El Pirata", "image_url": "/art/p.jpg"}
    )()

    item = analyzer._between_songs(0, 30_000)
    assert item.kind == S.KIND_PROGRAMA
    assert item.show_title == "El Pirata y su banda"
    assert item.show_lead == "El Pirata"
    assert item.source == "schedule"


def test_a_gap_the_schedule_cannot_name_stays_unknown(tmp_path):
    analyzer = build(tmp_path)
    analyzer._programme = lambda _at: None

    item = analyzer._between_songs(0, 30_000)
    assert item.kind == S.KIND_DESCONOCIDO
    assert item.show_title is None


def test_a_short_gap_is_split_so_the_songs_meet(tmp_path):
    analyzer = build(tmp_path)
    a, gap, b = _run("a", 0, 100_000), _run(None, 100_000, 100_000), _run("b", 106_000, 300_000)
    gap.key, gap.label = None, None

    cleaned = analyzer._absorb_slivers(
        [(a, 0, 100_000), (gap, 100_000, 106_000), (b, 106_000, 300_000)],
        seam_ms=10_000,
        nonmusic_ms=10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", "b"]
    assert cleaned[0][2] == cleaned[1][1] == 103_000


def test_a_gap_past_the_threshold_survives_to_be_named(tmp_path):
    analyzer = build(tmp_path)
    a, gap, b = _run("a", 0, 100_000), _run(None, 100_000, 100_000), _run("b", 130_000, 300_000)
    gap.key, gap.label = None, None

    cleaned = analyzer._absorb_slivers(
        [(a, 0, 100_000), (gap, 100_000, 130_000), (b, 130_000, 300_000)],
        seam_ms=10_000,
        nonmusic_ms=10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", None, "b"]


def test_a_gap_between_the_two_thresholds_is_absorbed(tmp_path):
    """Too long to be a crossfade, too short to be worth naming."""
    analyzer = build(tmp_path)
    a, gap, b = _run("a", 0, 100_000), _run(None, 100_000, 100_000), _run("b", 109_000, 300_000)
    gap.key, gap.label = None, None

    cleaned = analyzer._absorb_slivers(
        [(a, 0, 100_000), (gap, 100_000, 109_000), (b, 109_000, 300_000)],
        seam_ms=5_000,
        nonmusic_ms=15_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", "b"]


def test_the_thresholds_are_read_live(tmp_path):
    analyzer = build(tmp_path)
    settings.save(analyzer.conn, {"min_nonmusic_seconds": 30.0, "max_seam_seconds": 4.0})
    live = settings.load(analyzer.conn)
    assert live["min_nonmusic_seconds"] == 30.0
    assert live["max_seam_seconds"] == 4.0


def test_no_advert_kind_remains() -> None:
    assert S.ALL_KINDS == (S.KIND_CANCION, S.KIND_PROGRAMA, S.KIND_DESCONOCIDO)
    assert not hasattr(S, "KIND_PUBLICIDAD")


# --- one programme is one row ----------------------------------------------


def _gap(analyzer, start, end, show="Nano Jaquotot"):
    from rockfm.analyzer import Item

    return Item(start_ms=start, end_ms=end, kind=S.KIND_PROGRAMA,
                show_title=show, show_lead="x", source="schedule")


def test_a_programme_settled_in_pieces_is_one_row(tmp_path):
    """The scan settles a gap in pieces; each piece arrives separately.

    Written as it comes, one continuous programme became three abutting rows
    with the same name.
    """
    analyzer = build(tmp_path)
    analyzer._commit(_gap(analyzer, 0, 118_641))
    analyzer._commit(_gap(analyzer, 118_641, 165_609))
    analyzer._commit(_gap(analyzer, 165_609, 213_609))

    rows = list(analyzer.conn.execute("SELECT * FROM timeline ORDER BY start_ms"))
    assert len(rows) == 1
    assert rows[0]["start_ms"] == 0
    assert rows[0]["end_ms"] == 213_609


def test_a_different_programme_starts_a_new_row(tmp_path):
    analyzer = build(tmp_path)
    analyzer._commit(_gap(analyzer, 0, 60_000, show="Nano Jaquotot"))
    analyzer._commit(_gap(analyzer, 60_000, 120_000, show="El Pirata y su banda"))

    rows = list(analyzer.conn.execute("SELECT * FROM timeline ORDER BY start_ms"))
    assert len(rows) == 2


def test_a_programme_after_a_real_break_starts_a_new_row(tmp_path):
    """Far enough apart to be two separate stretches, not one interrupted."""
    analyzer = build(tmp_path)
    analyzer._commit(_gap(analyzer, 0, 60_000))
    analyzer._commit(_gap(analyzer, 400_000, 460_000))

    rows = list(analyzer.conn.execute("SELECT * FROM timeline ORDER BY start_ms"))
    assert len(rows) == 2


def test_two_airings_of_one_song_stay_two_rows(tmp_path):
    """Merging a repeat would turn it into one impossibly long play."""
    from rockfm.analyzer import Item

    analyzer = build(tmp_path)
    for start, end in ((0, 200_000), (200_000, 400_000)):
        analyzer._commit(Item(start_ms=start, end_ms=end, kind=S.KIND_CANCION,
                              title="Whatever", artist="Oasis", source="local"))

    rows = list(analyzer.conn.execute("SELECT * FROM timeline ORDER BY start_ms"))
    assert len(rows) == 2
