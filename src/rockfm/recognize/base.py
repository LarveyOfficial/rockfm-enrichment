"""Recogniser interface and factory.

The local fingerprint index answers most lookups for free; an external
recogniser is only consulted for audio we have never identified before. Keeping
that behind one small interface means the default (ShazamIO, free but
unofficial) can be swapped for a supported paid API by changing one env var.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("rockfm.recognize")


@dataclass(frozen=True)
class Recognition:
    artist: str
    title: str
    album: str | None = None
    art_url: str | None = None
    provider: str = "unknown"
    confidence: float = 1.0


@runtime_checkable
class Recognizer(Protocol):
    name: str

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        """Identify a short excerpt of 8 kHz mono audio, or return None."""

    def close(self) -> None: ...


# Seconds between calls when the recogniser is healthy. See
# `settings.recognizer_interval_seconds`, which overrides this at runtime.
#
# Small, and measured rather than guessed. Unpaced, the analyzer makes about
# nine calls a second and Shazam starts returning throttle pages instead of
# JSON within seconds; probed by hand at roughly one a second it answered
# thirty-six times out of thirty-six. A third of a second keeps a full window
# under a minute of lookups -- against twenty minutes of audio -- while staying
# far away from the rate that breaks it. An earlier twelve seconds here was the
# single largest cost in a scan; this is not that.
DEFAULT_MIN_INTERVAL = 0.35

# Consecutive failures before the external recogniser is left alone entirely.
# Counted against real backoff, so reaching this takes half a minute of trying
# rather than the few milliseconds it took when the backoff was accidentally
# zero.
OPEN_AFTER_ERRORS = 6

# Seconds to wait after the first failure, doubling each time. Deliberately not
# derived from `min_interval`: pacing is about how fast to go when things work,
# backoff is about how to behave when they do not, and tying them together
# meant setting the pace to zero silently disabled the backoff entirely.
BACKOFF_BASE_SECONDS = 1.0

# How long to stop calling it for once the breaker opens. Short, because the
# analyzer now holds unidentified audio rather than guessing at it: a long
# blackout no longer produces wrong labels, it produces no progress. Long
# enough to let a burst of throttling pass, short enough to recover from one.
COOL_OFF_SECONDS = 120.0


class Throttled:
    """Wraps a recogniser with pacing, failure backoff, and a circuit breaker.

    The external recogniser is a shared, unmetered service we do not own. The
    local index already absorbs repeats, so calls should be occasional -- this
    makes that a guarantee rather than a hope.

    The circuit breaker is the part that matters when things go wrong. A service
    that refuses fast is harmless; one that accepts the connection and never
    answers is not, because the caller pays the full timeout every time. Left to
    grind, that is what turned a twenty-minute window into a seven-hour one.
    After a few consecutive failures the recogniser is dropped for a spell and
    every lookup returns immediately, so the analyser keeps moving on the local
    index instead of queueing behind a service that is not going to reply.

    Pacing and backoff both wait; only an open breaker returns at once. The
    difference matters because a caller reads None as "no song here". While
    there is any prospect of an answer it is worth waiting for one -- there are
    hours of buffer and nothing is going out on air for six of them. Once the
    breaker is open there is no prospect, `degraded` says so, and the caller
    holds the audio instead of drawing a conclusion from silence.
    """

    def __init__(
        self,
        inner: Recognizer,
        min_interval: float = 3.0,
        max_backoff: float = 120.0,
        open_after: int = OPEN_AFTER_ERRORS,
        cool_off: float = COOL_OFF_SECONDS,
    ):
        self.inner = inner
        self.name = inner.name
        self.min_interval = min_interval
        self.max_backoff = max_backoff
        self.open_after = open_after
        self.cool_off = cool_off
        self._next_allowed = 0.0
        self._consecutive_errors = 0
        self.calls = 0
        self.skipped = 0

    @property
    def degraded(self) -> bool:
        """True only when the recogniser is not being consulted at all.

        Not merely "the last call failed". Callers use this to decide whether a
        silence means the audio holds no song or that nobody asked, and a single
        transient failure -- Shazam handing back a throttle page instead of JSON
        -- is not that. Treating one blip as an outage held seventeen minutes of
        audio, including a song the recogniser names at every offset.
        """
        return self._consecutive_errors >= self.open_after

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        import time

        now = time.monotonic()
        wait = self._next_allowed - now
        if wait > 0:
            if self.degraded:
                # The breaker is open: we are deliberately not calling at all,
                # and the caller knows it, because `degraded` says so.
                self.skipped += 1
                return None
            # Backoff waits rather than answering. Returning None here costs
            # nothing in time and everything in meaning: the caller cannot tell
            # it from "no song in this audio", and writes that down as fact.
            # Seventeen minutes of radio, one song of it playing throughout,
            # went to the timeline as unknown that way.
            time.sleep(min(wait, self.max_backoff))

        self.calls += 1
        try:
            result = self.inner.recognize(samples, rate)
        except Exception as exc:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.open_after:
                log.error(
                    "%s has failed %d times in a row (%s); leaving it alone for"
                    " %.0f min and relying on the local index",
                    self.name, self._consecutive_errors, exc, self.cool_off / 60,
                )
                self._next_allowed = time.monotonic() + self.cool_off
            else:
                delay = min(
                    BACKOFF_BASE_SECONDS * 2 ** (self._consecutive_errors - 1),
                    self.max_backoff,
                )
                log.warning("recogniser error (%s); backing off %.0fs", exc, delay)
                self._next_allowed = time.monotonic() + delay
            return None

        if self._consecutive_errors:
            log.info("%s is answering again", self.name)
        self._consecutive_errors = 0
        self._next_allowed = time.monotonic() + self.min_interval
        return result

    def close(self) -> None:
        self.inner.close()


class NullRecognizer:
    """Used when no external recogniser is configured or available."""

    name = "null"

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        return None

    def close(self) -> None:
        return None


def build(name: str, min_interval: float = DEFAULT_MIN_INTERVAL) -> Recognizer:
    key = (name or "").strip().lower()
    try:
        if key in {"shazam", "shazamio"}:
            from .shazam import ShazamRecognizer

            return Throttled(ShazamRecognizer(), min_interval)
        if key == "audd":
            from .audd import AudDRecognizer

            return Throttled(AudDRecognizer(), min_interval)
        if key == "acrcloud":
            from .acrcloud import AcrCloudRecognizer

            return Throttled(AcrCloudRecognizer(), min_interval)
        if key in {"", "none", "null"}:
            return NullRecognizer()
    except Exception as exc:
        log.error("recogniser %r unavailable (%s); falling back to none", name, exc)
        return NullRecognizer()

    log.error("unknown recogniser %r; falling back to none", name)
    return NullRecognizer()
