"""External recogniser guards.

These cover the failure that matters most in production: not a wrong answer,
but no answer at all. ShazamIO blocks rather than raising, so a lookup that
never returns is invisible to ordinary error handling -- it simply stops the
analyser, one probe at a time.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from rockfm.recognize.base import Recognition, Throttled
from rockfm.recognize.shazam import ShazamRecognizer, _parse


def _hanging_shazam(delay: float):
    class _Fake:
        async def recognize(self, wav, options=None):
            await asyncio.sleep(delay)
            return {"track": {"title": "never", "subtitle": "arrives"}}

    return _Fake()


def _recognizer(delay: float, timeout: float) -> ShazamRecognizer:
    """A ShazamRecognizer without its constructor, so shazamio stays optional."""
    r = object.__new__(ShazamRecognizer)
    r.timeout = timeout
    r._shazam = _hanging_shazam(delay)
    r._options = None
    return r


def test_a_lookup_that_never_answers_gives_up() -> None:
    r = _recognizer(delay=30.0, timeout=0.2)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(r._recognize(b"wav"))
    assert time.monotonic() - started < 5.0


def test_a_lookup_that_answers_in_time_is_returned() -> None:
    r = _recognizer(delay=0.0, timeout=5.0)
    payload = asyncio.run(r._recognize(b"wav"))
    assert payload["track"]["title"] == "never"


class _Recorder:
    """Stands in for a recogniser, returning or raising whatever it is told."""

    name = "recorder"

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def recognize(self, samples, rate):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        return None


def test_a_failed_lookup_backs_off() -> None:
    inner = _Recorder(TimeoutError("no answer"))
    throttled = Throttled(inner, min_interval=1.0)

    assert throttled.recognize(np.zeros(8), 8000) is None
    # Two failures should wait longer than one; a dead service must not be
    # retried at the same rate as a healthy one.
    first = throttled._next_allowed - time.monotonic()
    throttled._next_allowed = 0.0
    throttled.recognize(np.zeros(8), 8000)
    second = throttled._next_allowed - time.monotonic()
    assert second > first


def test_a_genuine_no_match_does_not_back_off() -> None:
    """Shazam answering "I don't know this" is a success, not a failure."""
    throttled = Throttled(_Recorder(None), min_interval=1.0)

    throttled.recognize(np.zeros(8), 8000)
    throttled._next_allowed = 0.0
    throttled.recognize(np.zeros(8), 8000)

    assert throttled._consecutive_errors == 0
    assert throttled._next_allowed - time.monotonic() == pytest.approx(1.0, abs=0.5)


def test_a_recognised_track_is_passed_through() -> None:
    found = Recognition(artist="The Clash", title="Should I Stay or Should I Go")
    assert Throttled(_Recorder(found), min_interval=0.0).recognize(np.zeros(8), 8000) is found


def test_an_empty_payload_is_not_a_match() -> None:
    assert _parse(None) is None
    assert _parse({}) is None
    assert _parse({"track": {"title": "", "subtitle": "The Clash"}}) is None


def test_a_dead_recogniser_is_dropped_rather_than_waited_on() -> None:
    """The failure that stalled production: never answering, never raising fast."""
    inner = _Recorder(TimeoutError("no answer"))
    throttled = Throttled(inner, min_interval=1.0, open_after=3, cool_off=600.0)

    # Each failure carries a penalty that must lapse before the next attempt,
    # so time is advanced by hand rather than waited out.
    for _ in range(3):
        throttled._next_allowed = 0.0
        assert throttled.recognize(np.zeros(8), 8000) is None
    assert inner.calls == 3

    # Once the circuit is open, further lookups cost nothing at all: no call to
    # the dead service, and -- crucially -- no sleeping the caller either.
    started = time.monotonic()
    for _ in range(20):
        assert throttled.recognize(np.zeros(8), 8000) is None
    assert inner.calls == 3
    assert throttled.skipped == 20
    assert time.monotonic() - started < 1.0


def test_an_open_breaker_answers_at_once() -> None:
    """No prospect of an answer, and `degraded` tells the caller so."""
    throttled = Throttled(
        _Recorder(TimeoutError("no answer")), min_interval=0.0, open_after=1,
        cool_off=600.0,
    )
    throttled.recognize(np.zeros(8), 8000)
    assert throttled.degraded

    started = time.monotonic()
    assert throttled.recognize(np.zeros(8), 8000) is None
    assert time.monotonic() - started < 1.0
    assert throttled.skipped == 1


def test_one_blip_is_not_an_outage() -> None:
    """Callers read `degraded` as "nobody asked", and act on it drastically.

    On one transient failure the analyzer stopped extending songs and held
    seventeen minutes of audio unlabelled -- including a track the recogniser
    names at every offset. Only a recogniser that has stopped being consulted
    counts.
    """
    throttled = Throttled(_Recorder(TimeoutError("no answer")), min_interval=0.0, open_after=3)
    assert not throttled.degraded

    throttled._next_allowed = 0.0
    throttled.recognize(np.zeros(8), 8000)
    assert not throttled.degraded, "one failure must not read as an outage"

    for _ in range(2):
        throttled._next_allowed = 0.0
        throttled.recognize(np.zeros(8), 8000)
    assert throttled.degraded, "a recogniser that stopped answering must say so"


def test_recovery_clears_the_penalty() -> None:
    inner = _Recorder(TimeoutError("no answer"))
    throttled = Throttled(inner, min_interval=0.0, open_after=2)
    for _ in range(2):
        throttled._next_allowed = 0.0
        throttled.recognize(np.zeros(8), 8000)
    assert throttled.degraded

    inner.outcome = Recognition(artist="Foreigner", title="Cold as Ice")
    throttled._next_allowed = 0.0
    assert throttled.recognize(np.zeros(8), 8000) is inner.outcome
    assert not throttled.degraded
