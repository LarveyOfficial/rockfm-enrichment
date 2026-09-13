"""Landmark ("constellation") audio fingerprinting.

Shazam-style: find spectral peaks, pair each anchor peak with nearby peaks, and
hash (freq1, freq2, dt). Matching histograms the time offset between query and
reference hashes -- a real match produces a sharp spike at one offset, whereas
coincidental hash collisions scatter.

Used for two jobs:
  * `music`     -- identify songs (seeded from previews, then learned from air)
  * `nonmusic`  -- cluster repeating ads, jingles and sweepers, which is how we
                   tell an advert apart from live DJ talk
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from .audio import ANALYSIS_RATE

N_FFT = 1024
HOP = 256
FRAMES_PER_SECOND = ANALYSIS_RATE / HOP  # 31.25

PEAK_NEIGHBOURHOOD = (21, 13)  # (frequency bins, time frames)
PEAKS_PER_SECOND = 26
FAN_OUT = 12
MIN_DT, MAX_DT = 1, 63
MAX_DF = 96
FREQ_BITS = 9
FREQ_MASK = (1 << FREQ_BITS) - 1

DEFAULT_MIN_VOTES = 8
SQLITE_PARAM_CHUNK = 900


def _spectrogram(samples: np.ndarray) -> np.ndarray:
    if samples.size < N_FFT:
        return np.zeros((N_FFT // 2 + 1, 0), dtype=np.float32)
    frame_count = 1 + (samples.size - N_FFT) // HOP
    strides = (samples.strides[0] * HOP, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(frame_count, N_FFT), strides=strides, writeable=False
    )
    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum).T.astype(np.float32)
    return 20.0 * np.log10(magnitude + 1e-6)


def _peaks(spec: np.ndarray) -> list[tuple[int, int]]:
    """Local maxima, thinned to a steady rate so loud passages don't dominate."""
    if spec.size == 0:
        return []
    local_max = maximum_filter(spec, size=PEAK_NEIGHBOURHOOD, mode="nearest")
    candidates = np.argwhere((spec == local_max) & (spec > spec.mean()))
    if candidates.size == 0:
        return []

    frames = spec.shape[1]
    budget = max(1, int(round(PEAKS_PER_SECOND / FRAMES_PER_SECOND * frames)))
    if len(candidates) > budget:
        strength = spec[candidates[:, 0], candidates[:, 1]]
        keep = np.argpartition(strength, -budget)[-budget:]
        candidates = candidates[keep]

    peaks = [(int(t), int(f)) for f, t in candidates]
    peaks.sort()
    return peaks


def compute(samples: np.ndarray) -> list[tuple[int, int]]:
    """Return (hash, time-offset-in-frames) pairs for 8 kHz mono audio."""
    peaks = _peaks(_spectrogram(samples))
    hashes: list[tuple[int, int]] = []
    for index, (t1, f1) in enumerate(peaks):
        paired = 0
        for t2, f2 in peaks[index + 1 :]:
            delta = t2 - t1
            if delta < MIN_DT:
                continue
            if delta > MAX_DT:
                break
            if abs(f2 - f1) > MAX_DF:
                continue
            value = (
                ((f1 & FREQ_MASK) << (FREQ_BITS + 6))
                | ((f2 & FREQ_MASK) << 6)
                | (delta & 0x3F)
            )
            hashes.append((value, t1))
            paired += 1
            if paired >= FAN_OUT:
                break
    return hashes


def frames_to_seconds(frames: int) -> float:
    return frames / FRAMES_PER_SECOND


@dataclass(frozen=True)
class Match:
    track_id: int
    key: str
    kind: str
    title: str | None
    artist: str | None
    votes: int
    offset_frames: int
    score: float  # votes as a fraction of query hashes
    margin: float  # best votes relative to the runner-up

    @property
    def offset_seconds(self) -> float:
        return frames_to_seconds(self.offset_frames)


class FingerprintIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- writing ---

    def add(
        self,
        *,
        kind: str,
        key: str,
        hashes: list[tuple[int, int]],
        title: str | None = None,
        artist: str | None = None,
        source: str = "broadcast",
        duration_ms: int | None = None,
        anchor_ms: int | None = None,
    ) -> int:
        now = int(time.time() * 1000)
        cursor = self.conn.execute(
            "INSERT INTO fp_tracks (kind, key, title, artist, source, occurrences,"
            " duration_ms, anchor_ms, created_ms, updated_ms)"
            " VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET occurrences = occurrences + 1,"
            " updated_ms = excluded.updated_ms",
            (kind, key, title, artist, source, duration_ms, anchor_ms, now, now),
        )
        row = self.conn.execute(
            "SELECT id, occurrences FROM fp_tracks WHERE key = ?", (key,)
        ).fetchone()
        track_id = int(row["id"])
        if row["occurrences"] > 1 and self._hash_count(track_id):
            return track_id  # already fingerprinted; just counted another airing
        self.conn.executemany(
            "INSERT INTO fp_hashes (hash, offset, track_id) VALUES (?, ?, ?)",
            [(value, offset, track_id) for value, offset in hashes],
        )
        del cursor
        return track_id

    def _hash_count(self, track_id: int) -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) FROM fp_hashes WHERE track_id = ?", (track_id,)
            ).fetchone()[0]
        )

    def extend(
        self, track_id: int, hashes: list[tuple[int, int]], shift_frames: int
    ) -> None:
        """Add more of a track to an existing reference.

        `shift_frames` places these hashes at their true position relative to the
        track's anchor, so the offset histogram stays coherent as the reference
        grows outwards from the first probe that identified it.
        """
        if not hashes:
            return
        self.conn.executemany(
            "INSERT INTO fp_hashes (hash, offset, track_id) VALUES (?, ?, ?)",
            [(value, offset + shift_frames, track_id) for value, offset in hashes],
        )

    def replace(
        self, track_id: int, hashes: list[tuple[int, int]], anchor_ms: int | None = None
    ) -> None:
        """Swap a reference for a better one.

        Once a song's real boundaries are known we re-fingerprint the whole
        aired span and replace whatever we had (a 30 s preview, or the single
        probe that first identified it). The broadcast version -- radio edit,
        processing and all -- is what we will hear next time.
        """
        self.conn.execute("DELETE FROM fp_hashes WHERE track_id = ?", (track_id,))
        self.conn.executemany(
            "INSERT INTO fp_hashes (hash, offset, track_id) VALUES (?, ?, ?)",
            [(value, offset, track_id) for value, offset in hashes],
        )
        self.conn.execute(
            "UPDATE fp_tracks SET anchor_ms = ?, source = 'broadcast', updated_ms = ?"
            " WHERE id = ?",
            (anchor_ms, int(time.time() * 1000), track_id),
        )

    def anchor_ms(self, track_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT anchor_ms FROM fp_tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return row["anchor_ms"] if row and row["anchor_ms"] is not None else None

    def find(self, key: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM fp_tracks WHERE key = ?", (key,)).fetchone()

    def bump(self, track_id: int) -> None:
        self.conn.execute(
            "UPDATE fp_tracks SET occurrences = occurrences + 1, updated_ms = ?"
            " WHERE id = ?",
            (int(time.time() * 1000), track_id),
        )

    def occurrences(self, track_id: int) -> int:
        row = self.conn.execute(
            "SELECT occurrences FROM fp_tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return int(row["occurrences"]) if row else 0

    def track_count(self, kind: str | None = None) -> int:
        if kind:
            return int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM fp_tracks WHERE kind = ?", (kind,)
                ).fetchone()[0]
            )
        return int(self.conn.execute("SELECT COUNT(*) FROM fp_tracks").fetchone()[0])

    # --- matching ---

    def match(
        self,
        hashes: list[tuple[int, int]],
        *,
        kind: str | None = None,
        min_votes: int = DEFAULT_MIN_VOTES,
    ) -> Match | None:
        if not hashes:
            return None

        by_hash: dict[int, list[int]] = defaultdict(list)
        for value, offset in hashes:
            by_hash[value].append(offset)
        unique = list(by_hash)

        votes: dict[tuple[int, int], int] = defaultdict(int)
        for start in range(0, len(unique), SQLITE_PARAM_CHUNK):
            chunk = unique[start : start + SQLITE_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            sql = (
                "SELECT h.hash, h.offset, h.track_id FROM fp_hashes h"
                f" WHERE h.hash IN ({placeholders})"
            )
            params: list = list(chunk)
            if kind:
                sql += " AND h.track_id IN (SELECT id FROM fp_tracks WHERE kind = ?)"
                params.append(kind)
            for row in self.conn.execute(sql, params):
                for query_offset in by_hash[row["hash"]]:
                    votes[(row["track_id"], row["offset"] - query_offset)] += 1

        if not votes:
            return None

        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        (track_id, offset_frames), best = ranked[0]
        if best < min_votes:
            return None

        runner_up = next(
            (count for (other, _), count in ranked[1:] if other != track_id), 0
        )
        track = self.conn.execute(
            "SELECT * FROM fp_tracks WHERE id = ?", (track_id,)
        ).fetchone()
        if track is None:
            return None

        return Match(
            track_id=track_id,
            key=track["key"],
            kind=track["kind"],
            title=track["title"],
            artist=track["artist"],
            votes=best,
            offset_frames=offset_frames,
            score=best / max(1, len(hashes)),
            margin=best / max(1, runner_up),
        )
