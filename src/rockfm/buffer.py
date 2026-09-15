"""Random access into the recorded segment buffer by wall-clock time."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import db
from .audio import ANALYSIS_RATE, decode_segment_files
from .config import Config

log = logging.getLogger("rockfm.buffer")

# Segment timestamps drift by a millisecond or two; anything under this is the
# next segment following on, not a hole in the recording.
CONTIGUOUS_TOLERANCE_MS = 1500


@dataclass(frozen=True)
class Range:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


class BufferReader:
    def __init__(self, conn: sqlite3.Connection, config: Config) -> None:
        self.conn = conn
        self.config = config

    def available(self) -> Range | None:
        earliest = db.earliest_segment(self.conn)
        latest = db.latest_segment(self.conn)
        if earliest is None or latest is None:
            return None
        return Range(earliest["pdt_ms"], latest["pdt_ms"] + latest["duration_ms"])

    def read(
        self, start_ms: int, duration_ms: int, rate: int = ANALYSIS_RATE
    ) -> np.ndarray:
        """Decode `duration_ms` of audio starting at wall-clock `start_ms`.

        Every sample sits where its wall clock says it should. Segments are laid
        into the span at their own timestamps and anything missing stays silent,
        rather than the recording being closed up around it.

        Closing it up is what this used to do, and it moved everything after a
        hole earlier by the length of the hole. Fourteen minutes went missing
        one night and the songs that followed were written into the timeline
        inside the gap they came after -- correctly identified, an hour of them
        at the wrong time, with one track appearing twice.

        Audio missing from the *end* is not padded: a short read is how the
        analyzer knows it has reached the edge of what was recorded.
        """
        if duration_ms <= 0:
            return np.zeros(0, dtype=np.float32)
        end_ms = start_ms + duration_ms
        rows = list(
            self.conn.execute(
                "SELECT * FROM segments WHERE pdt_ms + duration_ms > ? AND pdt_ms < ?"
                " ORDER BY pdt_ms ASC",
                (start_ms, end_ms),
            )
        )
        if not rows:
            return np.zeros(0, dtype=np.float32)

        # Segments that follow on from one another decode together, as one
        # stream; a break starts a new run placed at its own timestamp.
        runs: list[list] = []
        for row in rows:
            previous = runs[-1][-1] if runs else None
            follows_on = previous is not None and (
                abs(row["pdt_ms"] - (previous["pdt_ms"] + previous["duration_ms"]))
                <= CONTIGUOUS_TOLERANCE_MS
            )
            if follows_on:
                runs[-1].append(row)
            else:
                runs.append([row])

        want = int(duration_ms * rate / 1000)
        span = np.zeros(want, dtype=np.float32)
        filled_to = 0
        for run in runs:
            samples = decode_segment_files(
                [self.config.segments_dir / row["relpath"] for row in run], rate=rate
            )
            if samples.size == 0:
                continue
            at = int((run[0]["pdt_ms"] - start_ms) * rate / 1000)
            if at < 0:                      # the run began before the span
                samples = samples[-at:]
                at = 0
            room = want - at
            if room <= 0:
                continue
            samples = samples[:room]
            span[at : at + samples.size] = samples
            filled_to = max(filled_to, at + samples.size)

        return span[:filled_to]

    def has_gap(self, start_ms: int, end_ms: int) -> bool:
        return bool(db.gaps_between(self.conn, start_ms, end_ms))
