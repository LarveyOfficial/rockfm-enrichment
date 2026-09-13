"""Tests for the analyzer's pure logic: grouping and window slicing.

End-to-end behaviour against live radio lives in scripts/demo_timeline.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm.analyzer import PROBE_MS, Analyzer, Label, Window
from rockfm.audio import ANALYSIS_RATE

RECOGNIZE_RATE = 16000


def label(key: str) -> Label:
    return Label(key=key, artist="A", title=key, source="test", confidence=1.0, track_id=1)


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


def test_group_keeps_unidentified_stretches_separate():
    probes = [(0, label("a")), (24_000, None), (48_000, label("b"))]
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


def test_window_downsamples_for_fingerprinting(window):
    assert window.samples8.size == pytest.approx(
        window.samples16.size * ANALYSIS_RATE / RECOGNIZE_RATE, rel=0.01
    )
    assert window.probe8(1_000_000, 12_000).size == 12 * ANALYSIS_RATE


def test_window_refuses_slices_outside_itself(window):
    assert window.probe16(999_000, PROBE_MS).size == 0
    assert window.probe16(1_000_000 + 119_000, PROBE_MS).size == 0


def test_window_covers_reports_what_can_be_probed(window):
    assert window.covers(1_000_000)
    assert window.covers(1_000_000 + 100_000)
    assert not window.covers(1_000_000 + 119_000)
    assert not window.covers(999_000)
