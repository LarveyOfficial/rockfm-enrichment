"""Work out how far AzuraCast's broadcast trails the buffer, by listening to it.

Audio and metadata reach AzuraCast by different roads. The audio goes through
HLS into Liquidsoap and out of Icecast, buffering at every hop; the title goes
straight into Liquidsoap's output over the API, buffering nowhere. So the title
arrives first, by however much the audio path is holding -- tens of seconds,
and not a number anybody can look up.

It can be measured, though, because we still have the audio. Record a few
seconds of what AzuraCast is actually broadcasting, slide it along our own
recording of the same stretch, and the position it locks into says exactly how
far behind the broadcast is running. Feed that back into which row we call
"now" and the title changes when the song does.

The alternative was a hand-tuned constant that silently goes wrong every time
Liquidsoap reconnects.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import audio
from .buffer import BufferReader

log = logging.getLogger("rockfm.lag")

RATE = audio.ANALYSIS_RATE

# How much broadcast to capture for one measurement. Long enough to be unique
# in a rock station's catalogue, short enough not to span a song boundary.
PROBE_SECONDS = 12.0
# The furthest behind we will look. Beyond this something is wrong that an
# offset would only paper over.
MAX_LAG_SECONDS = 180.0
# Slack for clock skew and for ffmpeg handing us the last sample a moment late.
MARGIN_SECONDS = 10.0

# A normalised correlation below this is not a match, it is a coincidence.
MIN_SCORE = 0.30
# And the peak has to stand clear of the best match anywhere else, or we have
# found a repeated passage rather than the position.
MIN_LEAD = 1.6
# Ignore rivals this close to the peak: they are the same match, one sample over.
GUARD_SECONDS = 1.0

# Measurements are pooled and the median used, so one bad lock cannot move the
# timeline on its own.
HISTORY = 5
INTERVAL_SECONDS = 300.0
RETRY_SECONDS = 60.0


@dataclass(frozen=True)
class Alignment:
    """Where a probe sits inside a longer reference recording."""

    index: int      # sample offset of the probe's first sample
    score: float    # normalised correlation there, 0..1


def align(reference: np.ndarray, probe: np.ndarray, rate: int = RATE) -> Alignment | None:
    """Find `probe` inside `reference`, or return None if it is not confidently there.

    Normalised cross-correlation, computed through the FFT because the
    reference runs to a couple of million samples and the direct form would
    take minutes. Both sides have their mean removed, so a loudness difference
    between our recording and the broadcast does not affect the score.
    """
    if probe.size == 0 or reference.size < probe.size:
        return None
    ref = reference.astype(np.float64)
    pro = probe.astype(np.float64)
    pro = pro - pro.mean()
    energy = float(pro @ pro)
    if energy <= 0:
        return None                      # silence matches everywhere, so nowhere

    lags = ref.size - pro.size + 1
    size = 1 << int(np.ceil(np.log2(ref.size + pro.size)))
    spectrum = np.fft.rfft(ref, size) * np.fft.rfft(pro[::-1], size)
    raw = np.fft.irfft(spectrum, size)[pro.size - 1 : pro.size - 1 + lags]

    # Every window of the reference the probe could occupy, normalised by that
    # window's own variance -- a running sum rather than a pass per position.
    squares = np.concatenate([[0.0], np.cumsum(ref * ref)])
    totals = np.concatenate([[0.0], np.cumsum(ref)])
    window_sq = squares[pro.size :] - squares[:lags]
    window_sum = totals[pro.size :] - totals[:lags]
    variance = window_sq - (window_sum * window_sum) / pro.size
    norm = np.sqrt(np.maximum(variance, 0.0) * energy)
    score = np.divide(raw, norm, out=np.zeros(lags), where=norm > 1e-9)

    best = int(np.argmax(score))
    peak = float(score[best])
    if peak < MIN_SCORE:
        return None

    guard = max(1, int(rate * GUARD_SECONDS))
    rivals = np.concatenate([score[: max(0, best - guard)], score[best + guard + 1 :]])
    if rivals.size and peak < float(rivals.max()) * MIN_LEAD:
        return None
    return Alignment(index=best, score=peak)


class LagProbe:
    """Keeps a running answer to "how far behind is the broadcast?".

    `position` reports the buffer instant that should be airing now, and
    `listen_url` the mount to listen to; both are called fresh each time so a
    settings change or a DST shift is picked up without restarting anything.
    """

    def __init__(
        self,
        reader: BufferReader,
        *,
        position,
        listen_url,
        capture=audio.decode_url,
    ) -> None:
        self.reader = reader
        self._position = position
        self._listen_url = listen_url
        self._capture = capture
        self._lock = threading.Lock()
        self._history: deque[int] = deque(maxlen=HISTORY)
        self._measured_at: float | None = None
        self.stop_event = threading.Event()

    def _settled(self) -> int:
        """The pooled answer. Callers hold the lock; Lock is not reentrant."""
        if not self._history:
            return 0
        return int(statistics.median(self._history))

    @property
    def lag_ms(self) -> int:
        """The broadcast's delay, or zero until something has been measured."""
        with self._lock:
            return self._settled()

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "lag_seconds": round(self._settled() / 1000, 3),
                "measurements": len(self._history),
                "measured_seconds_ago": (
                    None if self._measured_at is None
                    else round(time.monotonic() - self._measured_at, 1)
                ),
            }

    def measure_once(self) -> int | None:
        """One measurement, or None if the broadcast could not be placed."""
        url = self._listen_url()
        if not url:
            return None
        try:
            probe = self._capture(
                url, rate=RATE, duration=PROBE_SECONDS, timeout=PROBE_SECONDS * 4
            )
        except Exception as exc:                     # noqa: BLE001 - ffmpeg, network, anything
            log.warning("could not listen to the broadcast at %s: %s", url, exc)
            return None
        # The moment capture ended is the moment the last sample was on air.
        airing_now = self._position()
        if probe.size < int(RATE * PROBE_SECONDS * 0.5):
            log.debug("broadcast sample too short to place (%d frames)", probe.size)
            return None

        probe_ms = int(probe.size * 1000 / RATE)
        begins_ms = airing_now - int(MAX_LAG_SECONDS * 1000) - probe_ms
        span_ms = int((MAX_LAG_SECONDS + MARGIN_SECONDS) * 1000) + probe_ms
        reference = self.reader.read(begins_ms, span_ms, rate=RATE)
        if reference.size < probe.size:
            return None

        found = align(reference, probe, rate=RATE)
        if found is None:
            log.debug("broadcast sample did not match the buffer")
            return None

        ends_ms = begins_ms + int((found.index + probe.size) * 1000 / RATE)
        lag = airing_now - ends_ms
        if lag < -MARGIN_SECONDS * 1000 or lag > MAX_LAG_SECONDS * 1000:
            log.debug("placed the broadcast %.1fs away, which is not believable", lag / 1000)
            return None
        return max(0, lag)

    def record(self, lag_ms: int) -> None:
        with self._lock:
            self._history.append(lag_ms)
            self._measured_at = time.monotonic()

    def run(self) -> None:
        while not self.stop_event.is_set():
            measured = self.measure_once()
            if measured is None:
                self.stop_event.wait(RETRY_SECONDS)
                continue
            self.record(measured)
            log.info(
                "broadcast is running %.1fs behind (using %.1fs)",
                measured / 1000,
                self.lag_ms / 1000,
            )
            self.stop_event.wait(INTERVAL_SECONDS)

    def stop(self) -> None:
        self.stop_event.set()
