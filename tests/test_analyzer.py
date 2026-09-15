"""Tests for the analyzer's pure logic: grouping and window slicing.

End-to-end behaviour against live radio lives in scripts/demo_timeline.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm.analyzer import PROBE_MS, Analyzer, Label, Window

RECOGNIZE_RATE = 16000


def label(key: str) -> Label:
    return Label(key=key, artist="A", title=key, source="test", confidence=1.0)


def test_group_merges_consecutive_identical_probes():
    probes = [
        (0, label("a")),
        (24_000, label("a")),
        (48_000, label("a")),
        (72_000, label("b")),
    ]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", "b"]
    assert runs[0].first_ms == 0
    assert runs[0].last_ms == 48_000
    assert runs[1].first_ms == 72_000


def test_group_keeps_a_substantial_unidentified_stretch():
    probes = [
        (0, label("a")),
        *[(24_000 * (i + 1), None) for i in range(4)],
        (120_000, label("b")),
    ]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", None, "b"]


def test_group_merges_adjacent_unidentified_probes():
    probes = [(0, None), (24_000, None), (48_000, label("a"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == [None, "a"]
    assert runs[0].last_ms == 24_000


def test_group_separates_a_song_that_returns_later():
    probes = [(0, label("a")), (24_000, label("b")), (48_000, label("a"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", "b", "a"]


def test_group_of_nothing_is_empty():
    assert Analyzer._group([]) == []


@pytest.fixture
def window() -> Window:
    seconds = 120
    samples = np.linspace(0, 1, seconds * RECOGNIZE_RATE, dtype=np.float32)
    return Window(1_000_000, 1_000_000 + seconds * 1000, samples)


def test_window_slices_at_the_right_offset(window):
    probe = window.probe16(1_000_000 + 10_000, 12_000)
    assert probe.size == 12 * RECOGNIZE_RATE
    expected = window.samples16[10 * RECOGNIZE_RATE]
    assert probe[0] == pytest.approx(expected)


def test_window_refuses_slices_outside_itself(window):
    assert window.probe16(999_000, PROBE_MS).size == 0
    assert window.probe16(1_000_000 + 119_000, PROBE_MS).size == 0


def test_window_covers_reports_what_can_be_probed(window):
    assert window.covers(1_000_000)
    assert window.covers(1_000_000 + 100_000)
    assert not window.covers(1_000_000 + 119_000)
    assert not window.covers(999_000)


def test_group_bridges_a_single_unmatched_probe_inside_a_song():
    # One probe that named nothing, flanked by the same song either side --
    # a quiet passage, not a boundary.
    probes = [(0, label("a")), (24_000, None), (48_000, label("a"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a"]
    assert runs[0].first_ms == 0
    assert runs[0].last_ms == 48_000


def test_group_bridges_a_short_unidentified_stretch_inside_a_song():
    """Meat Loaf arrived as 72s + 30s of nothing + 168s. It is one song."""
    probes = [(0, label("a")), (24_000, None), (48_000, None), (72_000, label("a"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a"]
    assert runs[0].last_ms == 72_000


def test_group_does_not_bridge_a_real_break():
    """Long enough and it is an advert break, not a passage of the song."""
    from rockfm.analyzer import MAX_BRIDGE_MS, STEP_MS

    gap = [(STEP_MS * (i + 1), None) for i in range(MAX_BRIDGE_MS // STEP_MS + 2)]
    probes = [(0, label("a")), *gap, (gap[-1][0] + STEP_MS, label("a"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", None, "a"]


def test_group_never_merges_two_different_songs_together():
    probes = [(0, label("a")), (24_000, None), (48_000, label("b"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs].count("a") == 1
    assert [run.key for run in runs].count("b") == 1


def test_grouping_keeps_an_unidentified_stretch_between_two_songs():
    """Whether it is a crossfade or a presenter link depends on how long it
    lasted, which grouping cannot know -- that is decided once boundaries are
    placed. Discarding it here lost real content."""
    probes = [(0, label("a")), (24_000, None), (48_000, label("b"))]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", None, "b"]


def test_a_real_break_between_two_songs_survives():
    probes = [
        (0, label("a")), (24_000, None), (48_000, None), (72_000, None),
        (96_000, None), (120_000, label("b")),
    ]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == ["a", None, "b"]


def test_leading_and_trailing_unidentified_runs_are_kept():
    probes = [(0, None), (24_000, label("a")), (48_000, None)]
    runs = Analyzer._group(probes)
    assert [run.key for run in runs] == [None, "a", None]
