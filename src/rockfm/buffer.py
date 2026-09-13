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
        """Decode `duration_ms` of audio starting at wall-clock `start_ms`."""
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

        paths = [self.config.segments_dir / row["relpath"] for row in rows]
        samples = decode_segment_files(paths, rate=rate)
        if samples.size == 0:
            return samples

        # Trim to the exact requested span; decoding starts at the first segment.
        lead_ms = max(0, start_ms - rows[0]["pdt_ms"])
        begin = int(lead_ms * rate / 1000)
        want = int(duration_ms * rate / 1000)
        return samples[begin : begin + want]

    def has_gap(self, start_ms: int, end_ms: int) -> bool:
        return bool(db.gaps_between(self.conn, start_ms, end_ms))
