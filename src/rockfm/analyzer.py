"""Work out what actually aired, and when.

Runs on recorded audio roughly six hours before it goes out, which is what makes
the expensive part affordable: there is time to probe, bisect boundaries and
look things up.

The pass over a window is:

  1. scan at a coarse stride, identifying each probe *independently* -- local
     fingerprint index first (free), external recogniser only on a miss
  2. group consecutive probes that named the same track into runs
  3. refine each run's edges by bisecting, which is free: both neighbouring
     tracks have just been learned, so the local index can answer "A or B?"
  4. enrich and commit
  5. re-fingerprint the whole aired span and replace the reference, so every
     later airing of that song is recognised locally at zero cost

Identifying each probe independently matters. An earlier version grew a run
outwards and learned as it went, which quietly ran away: a probe straddling a
song change still half-matches the outgoing song, gets learned, and drags the
reference into the next track. Independent identification cannot drift.

Probes are 12 s because that is what Shazam's signature format requires; see
recognize/shazam.py. A probe that straddles a song change matches neither side,
which is a useful signal rather than a failure -- it marks a boundary.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import UTC

import numpy as np
from scipy.signal import resample_poly

from . import db, fingerprint
from . import strings_es as S
from .audio import ANALYSIS_RATE
from .buffer import BufferReader
from .config import Config
from .enrich import Enricher
from .fingerprint import FingerprintIndex
from .recognize import Recognition
from .recognize import build as build_recognizer
from .rockfm_api import RockFmApi, catalog_key
from .timeshift import target_delay_seconds

log = logging.getLogger("rockfm.analyzer")

RECOGNIZE_RATE = 16000
PROBE_MS = 12_000          # fixed by Shazam's signature format
STEP_MS = 24_000           # coarse scan stride; songs are far longer than this
BISECT_LIMIT_MS = 2_000    # boundary precision we stop refining at
MIN_SONG_MS = 45_000       # shorter runs are treated as non-music
# Windows must comfortably hold several songs. At five minutes a single track
# fills one, so its boundaries land on the window edges instead of being found
# in the audio -- which is how every song came out the same length.
WINDOW_MS = 1_200_000      # decoded per pass
MIN_WINDOW_MS = 900_000    # never nibble at the live edge -- see run_once

# An unidentified stretch this short between two runs of the same song is a
# passage the recogniser could not place, not a break. Meat Loaf arrived as
# 72s + 30s of nothing + 168s; it is one song.
MAX_BRIDGE_MS = 60_000
# How far a run may exceed the release length before we stop trusting it.
DURATION_TOLERANCE = 1.4
TAIL_GUARD_MS = 90_000     # leave the newest audio alone; it may still be arriving
MIN_LOCAL_SCORE = 0.02
# Boundary tests need a stricter bar than identification. A probe only has to
# overlap a song slightly to be identified, but for locating an edge we want the
# probe to be *mostly* that song -- so the last probe that passes sits about half
# a probe-length before the real transition.
EDGE_MIN_SCORE = 0.25
CURSOR_KEY = db.ANALYZER_CURSOR_KEY


@dataclass
class Item:
    start_ms: int
    end_ms: int
    kind: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    year: int | None = None
    art_url: str | None = None
    confidence: float = 0.0
    source: str = "unknown"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class Label:
    key: str
    artist: str
    title: str
    source: str
    confidence: float
    track_id: int


@dataclass
class Run:
    key: str | None
    label: Label | None
    first_ms: int
    last_ms: int


class Window:
    """One decoded pass of the buffer, sliced in memory for every probe."""

    def __init__(self, start_ms: int, end_ms: int, samples16: np.ndarray) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.samples16 = samples16
        # Anti-aliased downsample; fingerprinting works at 8 kHz.
        self.samples8 = (
            resample_poly(samples16, 1, RECOGNIZE_RATE // ANALYSIS_RATE).astype(np.float32)
            if samples16.size
            else samples16
        )

    def _slice(self, samples: np.ndarray, rate: int, at_ms: int, length_ms: int) -> np.ndarray:
        begin = int((at_ms - self.start_ms) * rate / 1000)
        if begin < 0:
            return np.zeros(0, dtype=np.float32)
        end = begin + int(length_ms * rate / 1000)
        if end > samples.size:
            return np.zeros(0, dtype=np.float32)
        return samples[begin:end]

    def probe8(self, at_ms: int, length_ms: int = PROBE_MS) -> np.ndarray:
        return self._slice(self.samples8, ANALYSIS_RATE, at_ms, length_ms)

    def probe16(self, at_ms: int, length_ms: int = PROBE_MS) -> np.ndarray:
        return self._slice(self.samples16, RECOGNIZE_RATE, at_ms, length_ms)

    def covers(self, at_ms: int, length_ms: int = PROBE_MS) -> bool:
        return self.start_ms <= at_ms and at_ms + length_ms <= self.end_ms


class Analyzer:
    def __init__(
        self,
        config: Config,
        conn: sqlite3.Connection,
        recognizer=None,
        enricher: Enricher | None = None,
    ) -> None:
        self.config = config
        self.conn = conn
        self.reader = BufferReader(conn, config)
        self.index = FingerprintIndex(conn)
        self.recognizer = recognizer if recognizer is not None else build_recognizer(config.recognizer)
        if enricher is not None:
            self.enricher = enricher
        else:
            try:
                with RockFmApi() as api:
                    catalog = api.tracks()
            except Exception as exc:
                log.warning("catalog unavailable (%s); artwork falls back to iTunes/Deezer", exc)
                catalog = []
            self.enricher = Enricher(config.art_dir, catalog)
        self.external_calls = 0

    # --- cursor ---

    def cursor(self) -> int | None:
        raw = db.get_meta(self.conn, CURSOR_KEY)
        if raw:
            return int(raw)
        available = self.reader.available()
        return available.start_ms if available else None

    def set_cursor(self, value: int) -> None:
        db.set_meta(self.conn, CURSOR_KEY, str(value))

    def _note(self, state: str) -> None:
        """Record what the analyzer is doing, for the dashboard.

        A cold first pass probes a whole window before committing anything, so
        without this the only visible sign of life arrives minutes late and the
        thing looks dead.
        """
        db.set_meta(self.conn, db.ANALYZER_STATE_KEY, state)
        db.set_meta(self.conn, db.ANALYZER_HEARTBEAT_KEY, str(int(time.time() * 1000)))

    # --- identification and boundary refinement ---

    def _local(
        self, window: Window, at_ms: int, min_score: float = MIN_LOCAL_SCORE
    ) -> fingerprint.Match | None:
        probe = window.probe8(at_ms)
        if probe.size == 0:
            return None
        match = self.index.match(fingerprint.compute(probe), kind="music")
        if match and match.score >= min_score:
            return match
        return None

    def _external(self, window: Window, at_ms: int) -> Recognition | None:
        probe = window.probe16(at_ms)
        if probe.size == 0:
            return None
        self.external_calls += 1
        return self.recognizer.recognize(probe, RECOGNIZE_RATE)

    def _expected_ms(self, key: str) -> int | None:
        """How long this song is supposed to run, per iTunes/Deezer."""
        row = db.get_song_meta(self.conn, key)
        value = row["duration_ms"] if row else None
        return int(value) if value else None

    def _same_track(
        self, window: Window, at_ms: int, key: str, min_score: float = MIN_LOCAL_SCORE
    ) -> bool:
        match = self._local(window, at_ms, min_score)
        return match is not None and match.key == key

    # --- committing ---

    def _commit(self, item: Item) -> None:
        db.upsert_timeline(
            self.conn,
            {
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "kind": item.kind,
                "title": item.title,
                "artist": item.artist,
                "album": item.album,
                "year": item.year,
                "art_url": item.art_url,
                "show_title": None,
                "show_lead": None,
                "show_image": None,
                "confidence": item.confidence,
                "source": item.source,
                "cluster_id": None,
            },
            int(time.time() * 1000),
        )

    def _enriched(self, item: Item) -> Item:
        if not (item.artist and item.title):
            return item
        cached = db.get_song_meta(self.conn, catalog_key(item.artist, item.title))
        if cached is not None:
            return replace(
                item,
                album=item.album or cached["album"],
                year=cached["year"],
                art_url=cached["art_path"] or item.art_url,
            )
        info = self.enricher.lookup(item.artist, item.title)
        art_path = self.enricher.cache_art(info.art_url or item.art_url)
        db.upsert_song_meta(
            self.conn,
            {
                "key": catalog_key(item.artist, item.title),
                "title": item.title,
                "artist": item.artist,
                "album": info.album or item.album,
                "year": info.year,
                "duration_ms": info.duration_ms,
                "art_url": info.art_url or item.art_url,
                "art_path": art_path,
                "preview_url": info.preview_url,
                "source": info.source,
            },
            int(time.time() * 1000),
        )
        return replace(
            item,
            album=info.album or item.album,
            year=info.year,
            art_url=art_path or info.art_url or item.art_url,
        )

    # --- main pass ---

    def _identify(self, window: Window, at_ms: int) -> Label | None:
        """Name the audio at `at_ms`, independently of any neighbouring probe."""
        match = self._local(window, at_ms)
        if match is not None:
            self.index.bump(match.track_id)
            return Label(
                key=match.key,
                artist=match.artist or "",
                title=match.title or "",
                source="local",
                confidence=min(1.0, match.score * 4),
                track_id=match.track_id,
            )

        found = self._external(window, at_ms)
        if found is None:
            return None
        key = catalog_key(found.artist, found.title)
        track = self.index.find(key)
        if track is None:
            track_id = self.index.add(
                kind="music",
                key=key,
                hashes=fingerprint.compute(window.probe8(at_ms)),
                title=found.title,
                artist=found.artist,
                source="broadcast",
                anchor_ms=at_ms,
            )
        else:
            track_id = int(track["id"])
        return Label(
            key=key,
            artist=found.artist,
            title=found.title,
            source=found.provider,
            confidence=found.confidence,
            track_id=track_id,
        )

    def _boundary(self, window: Window, earlier: Run, later: Run) -> int:
        """Locate the transition between two consecutive runs.

        Bisects for the last probe start that is still mostly the outgoing
        track, then adds half a probe length: a 12 s probe stops being "mostly A"
        roughly six seconds before A actually ends.
        """
        if earlier.key is None:
            if later.key is None:
                return later.first_ms
            # Nothing identifiable before, a song after: find where the song
            # starts by searching backwards for the earliest probe that is
            # still it. Without this a track coming out of an advert break just
            # snapped to the next scan position, up to a step late.
            low, high = earlier.last_ms, later.first_ms
            while high - low > BISECT_LIMIT_MS:
                middle = (low + high) // 2
                if not window.covers(middle):
                    break
                if self._same_track(window, middle, later.key, EDGE_MIN_SCORE):
                    high = middle
                else:
                    low = middle
            # The earliest probe that reads as the new song begins half a probe
            # before the song itself does, so the boundary sits half a probe
            # after that -- the mirror of the rule used for endings below.
            return max(high + PROBE_MS // 2, earlier.first_ms)

        key = earlier.key
        low, high = earlier.last_ms, later.first_ms
        while high - low > BISECT_LIMIT_MS:
            middle = (low + high) // 2
            if not window.covers(middle):
                break
            if self._same_track(window, middle, key, EDGE_MIN_SCORE):
                low = middle
            else:
                high = middle
        return min(low + PROBE_MS // 2, later.first_ms + PROBE_MS)

    def _scan(self, window: Window) -> list[tuple[int, Label | None]]:
        probes: list[tuple[int, Label | None]] = []
        position = window.start_ms
        while window.covers(position):
            probes.append((position, self._identify(window, position)))
            position += STEP_MS
        return probes

    @staticmethod
    def _group(probes: list[tuple[int, Label | None]]) -> list[Run]:
        runs: list[Run] = []
        for position, label in probes:
            key = label.key if label else None
            if runs and runs[-1].key == key:
                runs[-1].last_ms = position
            else:
                runs.append(Run(key=key, label=label, first_ms=position, last_ms=position))

        # An unidentified stretch between two runs of the same song is part of
        # that song -- a quiet passage, or probes that landed somewhere the
        # recogniser could not place. Punching a hole through the middle of a
        # track is worse than bridging it, so long as the hole is short.
        merged: list[Run] = []
        for run in runs:
            if (
                len(merged) >= 2
                and run.key is not None
                and merged[-1].key is None
                and merged[-2].key == run.key
                and merged[-1].last_ms - merged[-1].first_ms <= MAX_BRIDGE_MS
            ):
                merged.pop()
                merged[-1].last_ms = run.last_ms
                continue
            merged.append(run)

        # A brief unidentified stretch between two different songs is the
        # transition itself -- probes that straddled the change and matched
        # neither side. Emitting that as a few seconds of "RockFM" between every
        # pair of tracks is noise; let the two songs share a boundary instead.
        # Anything longer than a scan step is a real break and is kept.
        without_seams: list[Run] = []
        for index, run in enumerate(merged):
            straddles_a_change = (
                run.key is None
                and 0 < index < len(merged) - 1
                and merged[index - 1].key is not None
                and merged[index + 1].key is not None
                and run.last_ms - run.first_ms <= STEP_MS
            )
            if not straddles_a_change:
                without_seams.append(run)
        return without_seams

    def _relearn(self, window: Window, run: Run, start_ms: int, end_ms: int) -> None:
        """Replace the reference with the whole aired span of this song."""
        if run.label is None:
            return
        span = window.probe8(start_ms, end_ms - start_ms)
        if span.size == 0:
            return
        self.index.replace(run.label.track_id, fingerprint.compute(span), anchor_ms=start_ms)

    def process_window(self, window: Window) -> int:
        """Commit everything in the window and return the new cursor.

        The final run is deliberately left uncommitted and the cursor rewound to
        its start: it is very likely a song still in progress at the window edge,
        and re-reading it next pass keeps songs whole rather than split in two.
        """
        runs = self._group(self._scan(window))
        if not runs:
            return window.end_ms

        # Teach the index each song over the whole stretch it was heard across,
        # before going looking for its edges. Bisection asks "is this still the
        # same song?", and until now the only reference was the single twelve
        # second probe that first identified it -- so the answer was no almost
        # everywhere, and the boundary collapsed onto the last coarse probe.
        # That is what pinned every song to a multiple of the scan step.
        for run in runs:
            if run.key is not None:
                self._relearn(window, run, run.first_ms, run.last_ms + PROBE_MS)

        # One shared boundary per adjacent pair, so items are contiguous and
        # never overlap -- a radio stream has no gaps between one thing and the next.
        edges = [window.start_ms]
        for position in range(len(runs) - 1):
            edges.append(self._boundary(window, runs[position], runs[position + 1]))
        edges.append(window.end_ms)

        committed = runs[:-1] if len(runs) > 1 else runs
        segments = [
            (run, edges[position], edges[position + 1])
            for position, run in enumerate(committed)
        ]
        for run, start_ms, end_ms in self._absorb_slivers(segments):
            self._commit_run(window, run, start_ms, end_ms)

        if len(runs) > 1 and edges[-2] > window.start_ms:
            return edges[-2]
        return window.end_ms

    @staticmethod
    def _absorb_slivers(
        segments: list[tuple[Run, int, int]],
    ) -> list[tuple[Run, int, int]]:
        """Fold anything too short to be a song into its neighbour.

        A window is rewound to the boundary before its last run, and that
        boundary is only accurate to a couple of seconds -- so the next window
        opens on a two second tail of the song just committed. Emitted on its
        own it became a sliver of "RockFM" wedged between every pair of tracks.
        It belongs to whichever item it abuts, and the timeline stays gapless
        either way.
        """
        cleaned: list[tuple[Run, int, int]] = []
        carried_start: int | None = None
        for index, (run, start_ms, end_ms) in enumerate(segments):
            if carried_start is not None:
                start_ms, carried_start = carried_start, None
            if end_ms - start_ms < MIN_SONG_MS:
                if index + 1 < len(segments):
                    carried_start = start_ms  # hand the span to what follows
                    continue
                if cleaned:  # trailing sliver: give it to what came before
                    previous_run, previous_start, _ = cleaned[-1]
                    cleaned[-1] = (previous_run, previous_start, end_ms)
                    continue
            cleaned.append((run, start_ms, end_ms))
        return cleaned

    def _commit_run(self, window: Window, run: Run, start_ms: int, end_ms: int) -> None:
        if end_ms <= start_ms:
            return
        if run.key is None or end_ms - start_ms < MIN_SONG_MS:
            self._commit(
                Item(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    kind=S.KIND_DESCONOCIDO,
                    source="unresolved",
                )
            )
            return

        assert run.label is not None
        item = self._enriched(
            Item(
                start_ms=start_ms,
                end_ms=end_ms,
                kind=S.KIND_CANCION,
                title=run.label.title,
                artist=run.label.artist,
                confidence=run.label.confidence,
                source=run.label.source,
            )
        )
        self._commit(item)
        self._relearn(window, run, start_ms, end_ms)
        log.info(
            "%s  %s - %s  (%.0fs, %s)",
            _clock(item.start_ms), item.artist, item.title,
            item.duration_ms / 1000, item.source,
        )

    def _must_analyze_now(self, start_ms: int) -> bool:
        """True once this audio is close enough to airing that we cannot wait."""
        delay_ms = target_delay_seconds(self.config) * 1000
        lead_ms = self.config.lead_time_minutes * 60_000
        return time.time() * 1000 >= start_ms + delay_ms - lead_ms

    def run_once(self) -> int:
        available = self.reader.available()
        if available is None:
            return 0
        start = self.cursor() or available.start_ms
        start = max(start, available.start_ms)
        limit = available.end_ms - TAIL_GUARD_MS
        if limit - start < PROBE_MS * 2:
            self._note("waiting")
            return 0

        # Wait for a decent block of audio before analysing. Having caught up
        # with the live edge, ingest only adds six seconds at a time, and a
        # window barely longer than one probe gets exactly one probe -- so a
        # single unmatched probe condemns the whole stretch to "unknown", and
        # the cursor moves past it for good. With hours of delay in hand there
        # is no reason to work that way; the only exception is audio close
        # enough to airing that waiting would miss the deadline.
        if limit - start < MIN_WINDOW_MS and not self._must_analyze_now(start):
            self._note("waiting")
            return 0

        end = min(start + WINDOW_MS, limit)
        samples = self.reader.read(start, end - start, rate=RECOGNIZE_RATE)
        if samples.size == 0:
            self.set_cursor(end)
            return 0

        real_end = start + int(samples.size / RECOGNIZE_RATE * 1000)
        window = Window(start, min(end, real_end), samples)
        self._note("scanning")
        position = self.process_window(window)
        self._note("idle")
        advance = position - start
        if advance < PROBE_MS:
            # The window held nothing we could resolve yet -- most likely a song
            # still in progress at the live edge. Leave the cursor and retry once
            # more audio has arrived.
            return 0
        self.set_cursor(position)
        return advance


def _clock(epoch_ms: int) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%H:%M:%S")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = Config()
    config.ensure_dirs()
    analyzer = Analyzer(config, db.connect(config.db_path))
    while True:
        advanced = analyzer.run_once()
        if advanced <= 0:
            time.sleep(30)


if __name__ == "__main__":
    main()
