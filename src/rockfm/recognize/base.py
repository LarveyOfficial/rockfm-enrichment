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
DEFAULT_MIN_INTERVAL = 12.0

# Consecutive failures before the external recogniser is left alone entirely.
OPEN_AFTER_ERRORS = 4

# How long to stop calling it for once that happens. The local index keeps
# working throughout, so the cost of waiting is only the songs it has not
# learned yet -- far less than the cost of blocking on a service that is down.
COOL_OFF_SECONDS = 900.0


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

    Only the healthy pacing interval sleeps. Backoff and cool-off return at once:
    making the analyser wait out a penalty it cannot shorten just moves the
    stall from the recogniser into the scan.
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
        """True when lookups are failing, so callers can stop asking twice."""
        return self._consecutive_errors > 0

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        import time

        now = time.monotonic()
        wait = self._next_allowed - now
        if wait > 0:
            if self._consecutive_errors:
                # Penalty time. Answer now and let the local index carry it.
                self.skipped += 1
                return None
            time.sleep(wait)

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
                delay = min(self.min_interval * 2**self._consecutive_errors, self.max_backoff)
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
