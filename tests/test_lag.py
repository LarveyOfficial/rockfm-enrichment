"""Timing the broadcast against our own recording of it.

The title used to change up to half a minute before the song did, because the
API push overtakes the audio's own buffering. These cover the measurement that
closes that gap: find the broadcast inside the buffer, and the distance is the
answer.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from rockfm.lag import (
    MAX_LAG_SECONDS,
    PROBE_SECONDS,
    RATE,
    LagProbe,
    align,
)

NOW_MS = 1_789_500_000_000
PROBE_MS = int(PROBE_SECONDS * 1000)
BROADCAST_LAG_MS = 27_400

# A stand-in for the recording: deterministic, and unlike real music it never
# repeats, so a correct match is unambiguous.
_WORLD_START_MS = NOW_MS - int((MAX_LAG_SECONDS + 600) * 1000)
_WORLD = np.random.default_rng(7).standard_normal(
    int((MAX_LAG_SECONDS + 900) * RATE)
).astype(np.float32)


def world(start_ms: int, span_ms: int) -> np.ndarray:
    """Whatever was on air over that stretch of wall clock."""
    first = int((start_ms - _WORLD_START_MS) * RATE / 1000)
    return _WORLD[max(0, first) : max(0, first) + int(span_ms * RATE / 1000)]


# --- finding one recording inside another -----------------------------------


def test_a_probe_is_found_where_it_was_taken_from():
    reference = _WORLD[: 40 * RATE]
    offset = 13 * RATE
    found = align(reference, reference[offset : offset + 5 * RATE])
    assert found is not None
    assert found.index == offset
    assert found.score > 0.99


def test_a_quieter_broadcast_still_matches():
    """Icecast re-encodes at a different level; that must not cost us the lock."""
    reference = _WORLD[: 40 * RATE]
    offset = 9 * RATE
    quiet = reference[offset : offset + 5 * RATE] * 0.05
    found = align(reference, quiet)
    assert found is not None and found.index == offset


def test_a_constant_offset_in_the_signal_does_not_shift_the_answer():
    reference = _WORLD[: 40 * RATE]
    offset = 21 * RATE
    shifted = reference[offset : offset + 5 * RATE] + 0.4
    found = align(reference, shifted)
    assert found is not None and found.index == offset


def test_audio_that_is_not_in_the_reference_is_refused():
    reference = _WORLD[: 40 * RATE]
    stranger = np.random.default_rng(99).standard_normal(5 * RATE).astype(np.float32)
    assert align(reference, stranger) is None


def test_silence_places_nothing():
    """A silent probe correlates with everything equally, which is no answer."""
    reference = _WORLD[: 40 * RATE]
    assert align(reference, np.zeros(5 * RATE, dtype=np.float32)) is None


def test_a_passage_that_repeats_is_refused_rather_than_guessed():
    """Two equally good homes for the probe means we do not know which."""
    piece = _WORLD[: 5 * RATE]
    filler = np.random.default_rng(3).standard_normal(10 * RATE).astype(np.float32)
    reference = np.concatenate([piece, filler, piece, filler])
    assert align(reference, piece) is None


def test_a_probe_longer_than_the_reference_is_refused():
    assert align(_WORLD[: 2 * RATE], _WORLD[: 5 * RATE]) is None


# --- turning that into a delay ----------------------------------------------


class Reader:
    def __init__(self):
        self.asked = []

    def read(self, start_ms, span_ms, rate=RATE):
        self.asked.append((start_ms, span_ms, rate))
        return world(start_ms, span_ms)


def probe_for(lag_ms=BROADCAST_LAG_MS, reader=None):
    """A LagProbe listening to a broadcast planted exactly `lag_ms` behind."""
    def capture(_url, rate=RATE, duration=PROBE_SECONDS, timeout=0.0):
        assert rate == RATE and duration == PROBE_SECONDS and timeout
        return world(NOW_MS - lag_ms - PROBE_MS, PROBE_MS)

    return LagProbe(
        reader or Reader(),
        position=lambda: NOW_MS,
        listen_url=lambda: "http://example.com/listen.mp3",
        capture=capture,
    )


def test_the_delay_is_measured_from_the_audio():
    measured = probe_for().measure_once()
    assert measured == pytest.approx(BROADCAST_LAG_MS, abs=200)


@pytest.mark.parametrize("planted", [0, 5_000, 60_000, 120_000])
def test_delays_across_the_range_are_all_found(planted):
    assert probe_for(planted).measure_once() == pytest.approx(planted, abs=200)


def test_nothing_is_claimed_before_a_measurement_lands():
    assert probe_for().lag_ms == 0


def test_a_broadcast_that_cannot_be_reached_measures_nothing():
    probe = LagProbe(
        Reader(), position=lambda: NOW_MS, listen_url=lambda: "", capture=None
    )
    assert probe.measure_once() is None


def test_a_capture_that_throws_is_survived():
    def capture(*_a, **_kw):
        raise OSError("ffmpeg went away")

    probe = LagProbe(
        Reader(),
        position=lambda: NOW_MS,
        listen_url=lambda: "http://example.com/listen.mp3",
        capture=capture,
    )
    assert probe.measure_once() is None
    assert probe.lag_ms == 0


def test_one_bad_reading_cannot_move_the_answer():
    """The median is why a single mis-lock does not drag the timeline with it."""
    probe = probe_for()
    for good in (27_000, 27_400, 27_200, 27_300):
        probe.record(good)
    probe.record(150_000)
    assert probe.lag_ms == pytest.approx(27_300, abs=250)


def test_the_search_covers_the_whole_range_it_promises():
    reader = Reader()
    probe_for(reader=reader).measure_once()
    start_ms, span_ms, _rate = reader.asked[0]
    assert start_ms <= NOW_MS - MAX_LAG_SECONDS * 1000
    assert start_ms + span_ms >= NOW_MS


def test_state_reports_what_it_knows():
    probe = probe_for()
    assert probe.state["measurements"] == 0
    assert probe.state["measured_seconds_ago"] is None
    probe.record(27_400)
    assert probe.state["lag_seconds"] == pytest.approx(27.4)
    assert probe.state["measured_seconds_ago"] is not None


def test_reading_the_state_does_not_deadlock():
    """`state` and `lag_ms` both want the lock, and a Lock is not reentrant.

    Written after `state` hung the suite outright: a deadlock is not a failing
    test, it is a run that never ends, so this gives it a deadline.
    """
    probe = probe_for()
    probe.record(27_400)
    done = threading.Event()
    threading.Thread(target=lambda: (probe.state, done.set()), daemon=True).start()
    assert done.wait(5), "reading the probe's state deadlocked"
