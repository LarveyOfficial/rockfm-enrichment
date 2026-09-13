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


class Throttled:
    """Wraps a recogniser with a minimum call interval and failure backoff.

    The external recogniser is a shared, unmetered service we do not own. The
    local index already absorbs repeats, so calls should be occasional -- this
    makes that a guarantee rather than a hope.
    """

    def __init__(self, inner: "Recognizer", min_interval: float = 3.0, max_backoff: float = 120.0):
        self.inner = inner
        self.name = inner.name
        self.min_interval = min_interval
        self.max_backoff = max_backoff
        self._next_allowed = 0.0
        self._consecutive_errors = 0
        self.calls = 0

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        import time

        wait = self._next_allowed - time.monotonic()
        if wait > 0:
            time.sleep(wait)

        self.calls += 1
        try:
            result = self.inner.recognize(samples, rate)
        except Exception as exc:
            self._consecutive_errors += 1
            delay = min(self.min_interval * 2**self._consecutive_errors, self.max_backoff)
            log.warning("recogniser error (%s); backing off %.0fs", exc, delay)
            self._next_allowed = time.monotonic() + delay
            return None

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


def build(name: str, min_interval: float = 3.0) -> Recognizer:
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
