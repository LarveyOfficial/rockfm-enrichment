"""Audio nobody recorded belongs to nobody.

RockFM's master playlist answered 404 for fourteen minutes one night. The
recorder retried throughout and got nothing, which is the honest outcome: the
upstream live window is about twenty-four seconds, so anything longer than that
is gone from the source before it can be asked for again.

What must not happen is a song being written across it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm import db
from rockfm.analyzer import PROBE_MS, STEP_MS, Analyzer, Label, Window
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer

RATE = 16_000
HOLE_START, HOLE_END = 300_000, 600_000


def build(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


def window():
    return Window(0, 1_200_000, np.zeros(16, dtype=np.float32))


# --- finding the hole -------------------------------------------------------


def _record_gap(conn, hole_start, hole_end):
    conn.execute(
        "INSERT INTO gaps (after_pdt_ms, before_pdt_ms, missing_ms, reason, detected_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        (hole_start - 6000, hole_end, hole_end - hole_start, "missed_segments", 0),
    )
    conn.commit()


def test_a_gap_that_began_before_the_window_is_still_found(tmp_path):
    """Fourteen minutes of missing audio starts once and overlaps what follows."""
    analyzer = build(tmp_path)
    _record_gap(analyzer.conn, 100_000, 900_000)

    found = db.gaps_overlapping(analyzer.conn, 500_000, 1_000_000)
    assert found, "a gap running through the window was missed entirely"
    assert not db.gaps_overlapping(analyzer.conn, 1_000_000, 1_200_000)


# --- not asking about silence ----------------------------------------------


def test_probes_inside_a_hole_are_never_sent_to_the_recogniser(tmp_path):
    analyzer = build(tmp_path)
    _record_gap(analyzer.conn, HOLE_START, HOLE_END)
    asked: list[int] = []
    analyzer._external = lambda _w, at: asked.append(at)

    analyzer.process_window(window())

    inside = [at for at in asked if HOLE_START <= at < HOLE_END]
    assert not inside, f"asked about audio that was never recorded: {inside}"
    assert asked, "the scan stopped asking about everything"


# --- and not claiming it ----------------------------------------------------


def test_a_song_cannot_end_after_the_recording_stopped(tmp_path):
    analyzer = build(tmp_path)
    analyzer._holes = [(HOLE_START, HOLE_END)]
    assert analyzer._outside_holes(200_000, 700_000) == (200_000, HOLE_START)


def test_a_song_cannot_start_before_the_recording_resumed(tmp_path):
    analyzer = build(tmp_path)
    analyzer._holes = [(HOLE_START, HOLE_END)]
    assert analyzer._outside_holes(400_000, 700_000) == (HOLE_END, 700_000)


def test_a_span_clear_of_the_hole_is_left_alone(tmp_path):
    analyzer = build(tmp_path)
    analyzer._holes = [(HOLE_START, HOLE_END)]
    assert analyzer._outside_holes(650_000, 800_000) == (650_000, 800_000)


def test_no_song_is_written_across_a_hole(tmp_path):
    """The whole point: the timeline must not claim the missing minutes."""
    analyzer = build(tmp_path)
    _record_gap(analyzer.conn, HOLE_START, HOLE_END)
    analyzer._enriched = lambda item: item
    analyzer._programme = lambda _at: None

    def identify(_w, at, **_kw):
        # A song heard either side of the hole, telling the same start both times.
        return Label(key="k", artist="Mr. Big", title="To Be with You",
                     source="shazamio", confidence=1.0, started_ms=240_000)

    analyzer._identify = identify
    analyzer.process_window(window())

    songs = [
        dict(row)
        for row in analyzer.conn.execute(
            "SELECT * FROM timeline WHERE kind = 'cancion' ORDER BY start_ms"
        )
    ]
    trespassing = [
        (row["start_ms"], row["end_ms"])
        for row in songs
        if row["start_ms"] < HOLE_END and row["end_ms"] > HOLE_START
    ]
    assert not trespassing, f"songs written over audio nobody recorded: {trespassing}"
    assert STEP_MS and PROBE_MS  # the grid the scan walks, for context


@pytest.mark.parametrize("span", [(0, HOLE_START), (HOLE_END, 900_000)])
def test_audio_either_side_of_a_hole_is_untouched(tmp_path, span):
    analyzer = build(tmp_path)
    analyzer._holes = [(HOLE_START, HOLE_END)]
    assert analyzer._outside_holes(*span) == span
