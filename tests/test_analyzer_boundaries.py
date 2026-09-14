"""Song boundaries must come from the audio, not from the scan geometry.

A live instance once emitted three consecutive songs at exactly 198.0s each --
Foreigner (5:00), Kaiser Chiefs (3:25) and AC/DC (3:29) -- perfectly abutting.
Nothing about that came from the music: 198 is 8 x the 24s scan step plus half a
probe. Two faults combined. Windows were short enough for one song to fill one,
so its edges fell on the window rather than on a change; and the only reference
a freshly identified song had was the single 12s probe that found it, so the
bisection that is supposed to locate the edge could not confirm the song
anywhere else and collapsed onto the last coarse probe.

This drives the real grouping and boundary code over a synthetic timeline, with
recognition stubbed, and checks the edges track the songs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm import db
from rockfm.analyzer import (
    MIN_WINDOW_MS,
    PROBE_MS,
    Analyzer,
    Label,
    Window,
)
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer

RATE = 16_000
# Deliberately different lengths; the bug made them all identical.
SONGS = [("foreigner", 300), ("ruby", 205), ("tnt", 209), ("meatloaf", 302)]


def build(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


@pytest.fixture
def played():
    spans, at = [], 0
    for name, seconds in SONGS:
        spans.append((at, at + seconds * 1000, name))
        at += seconds * 1000
    return spans


def song_at(spans, ms):
    for start, end, name in spans:
        if start <= ms < end:
            return name
    return None


def run_analyzer(tmp_path, spans, passes=4):
    analyzer = build(tmp_path)

    def identify(_window, at_ms):
        name = song_at(spans, at_ms)
        if name is None:
            return None
        return Label(key=name, artist="x", title=name, source="stub",
                     confidence=1.0, track_id=1)

    # Once a run is learned, "is this still the same song?" is answerable
    # anywhere inside it: a probe is that song if most of it lies within.
    def same_track(_window, at_ms, key, min_score=0.0):
        return song_at(spans, at_ms + PROBE_MS // 2) == key

    committed: list = []
    analyzer._identify = identify
    analyzer._same_track = same_track
    analyzer._enriched = lambda item: item
    analyzer._commit = committed.append
    analyzer._relearn = lambda *args: None

    cursor = 0
    for _ in range(passes):
        end = cursor + MIN_WINDOW_MS
        samples = np.zeros(int((end - cursor) / 1000 * RATE), dtype=np.float32)
        cursor = analyzer.process_window(Window(cursor, end, samples))
    return committed


def test_durations_are_not_locked_to_the_scan_geometry(tmp_path, played):
    songs = [i for i in run_analyzer(tmp_path, played) if i.title]
    durations = {round(i.duration_ms / 1000, 1) for i in songs}
    assert len(songs) >= 3
    assert len(durations) > 1, f"every song came out the same length: {durations}"


def test_each_song_lands_close_to_its_real_length(tmp_path, played):
    truth = dict(SONGS)
    for item in run_analyzer(tmp_path, played):
        if not item.title or item.title not in truth:
            continue
        assert item.duration_ms / 1000 == pytest.approx(truth[item.title], abs=15)


def test_items_never_overlap_and_leave_no_holes(tmp_path, played):
    items = sorted(run_analyzer(tmp_path, played), key=lambda i: i.start_ms)
    for earlier, later in zip(items, items[1:], strict=False):
        assert later.start_ms == earlier.end_ms, "timeline must be contiguous"


def test_no_slivers_are_emitted_between_tracks(tmp_path, played):
    """A rewound window re-sees a couple of seconds of the song just committed."""
    items = run_analyzer(tmp_path, played)
    interior = [i for i in items if i.start_ms > 0 and i.end_ms < 1_000_000]
    assert all(i.duration_ms >= 20_000 for i in interior), [
        (i.title or i.kind, i.duration_ms / 1000) for i in interior
    ]


def test_absorb_slivers_folds_a_leading_tail_into_what_follows():
    run_a = Label(key="a", artist="x", title="a", source="s", confidence=1, track_id=1)
    from rockfm.analyzer import Run

    tail = Run(key="a", label=run_a, first_ms=0, last_ms=0)
    real = Run(key="b", label=run_a, first_ms=6_000, last_ms=200_000)
    cleaned = Analyzer._absorb_slivers([(tail, 0, 2_000), (real, 2_000, 200_000)])
    assert len(cleaned) == 1
    assert cleaned[0][1] == 0 and cleaned[0][2] == 200_000
