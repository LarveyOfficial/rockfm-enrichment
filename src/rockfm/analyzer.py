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
from datetime import UTC, datetime

import numpy as np
from scipy.signal import resample_poly

from . import db, fingerprint, settings
from . import strings_es as S
from .audio import ANALYSIS_RATE
from .buffer import BufferReader
from .classify import segmenter as seg
from .config import Config
from .enrich import Enricher
from .fingerprint import FingerprintIndex
from .recognize import Recognition
from .recognize import build as build_recognizer
from .rockfm_api import RockFmApi, catalog_key
from .schedule import Schedule
from .timeshift import target_delay_seconds

log = logging.getLogger("rockfm.analyzer")

RECOGNIZE_RATE = 16000
PROBE_MS = 12_000          # fixed by Shazam's signature format
STEP_MS = 24_000           # coarse scan stride; songs are far longer than this
# Bisection never calls the external recogniser -- it queries our own index --
# so it is not bound by Shazam's sample length. Measured against a reference
# learned from the same broadcast audio, a 2 s probe still matched 100% of the
# time and reported its position to within 0.016 s, so the probe can be short.
# That matters: the boundary correction is half the probe length, so a 12 s
# probe put a six second guess into every edge.
BISECT_PROBE_MS = 2_000
BISECT_MIN_VOTES = 5
# Offset-derived start estimates should agree closely; if they scatter, the
# reference is not trustworthy and bisection is the safer answer.
OFFSET_AGREEMENT_MS = 4_000
# Where to look again when a probe comes back empty, before concluding there is
# no song there.
RETRY_OFFSETS_MS = (5_000, 10_000)
# How far a run may stray from the release length before we distrust it.
DURATION_DISAGREEMENT = 0.35
# How far an offset-derived start may sit from the edge we searched for before
# we distrust it and keep the searched one.
OFFSET_TRUST_MS = 15_000
# How much a still-growing run must gain before its reference is refreshed.
RELEARN_GROWTH_MS = 90_000
# The coarse scan steps 24s, so a song can end up to a step past its last
# matching probe. The reference is learned from that probe extent, and the edge
# search cannot confirm a song beyond what the reference covers -- so the end
# landed systematically early, and the song's tail leaked into whatever came
# next. Walking the extent out in finer steps first costs a few probes and
# removes the bias.
EXTEND_STEP_MS = 6_000
EXTEND_LIMIT_MS = 24_000
# External calls an edge may spend when the index cannot speak for the audio.
# Extension runs before the song is learned, so the reference is whatever named
# it -- a 12 s probe, or a 30 s catalogue preview. Asking only the index there
# stops at the reference's coverage rather than the song's end, which put every
# edge short and invented gaps between songs that actually abut.
#
# Enough to walk the whole way out. Grounding covers to one probe past the last
# match, and a probe *starting* there already reaches beyond it, so the index
# answers for a step at most -- which made a smaller budget, not EXTEND_LIMIT_MS,
# the real stopping point. A song ended thirteen seconds early on a budget worth
# twelve. The limit above is the bound; this must not quietly undercut it.
EXTEND_EXTERNAL_BUDGET = EXTEND_LIMIT_MS // EXTEND_STEP_MS
BISECT_LIMIT_MS = 400      # boundary precision we stop refining at
MIN_SONG_MS = 45_000       # floor for songs of unknown length
# A song has to run for a decent share of itself to count as having been played.
# Stations trail tracks over links and play stings built from them, and those
# match as confidently as the real airing -- length is what separates them.
MIN_SONG_FRACTION = 0.15
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
    show_title: str | None = None
    show_lead: str | None = None
    show_image: str | None = None

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
        # Names the stretches between songs. The station publishes its own
        # weekly grid, so a gap does not have to be guessed at -- it can be
        # looked up.
        self.schedule = Schedule()
        self.external_calls = 0
        # Extents already taught to the index this window, so that planning --
        # which now runs after nearly every probe -- does not re-fingerprint
        # minutes of audio each time.
        self._learned: dict[str, tuple[int, int]] = {}
        # Songs already committed this window. Their reference has been taught
        # from boundary-refined edges, which is strictly better than the raw
        # probe extent -- so planning must not overwrite it with the worse one.
        self._settled: set[str] = set()
        self._grounded: set[str] = set()
        # Set when a window ends with audio nobody could identify because the
        # recogniser was down. The cursor stops there rather than past it.
        self._held = False
        # Pure numpy, ~0.2ms per probe. Deliberately not the CNN: this runs on
        # every probe that misses locally, and it only has to be right when it
        # is certain.
        self._gate = seg.LightSegmenter()

    # --- cursor ---

    def cursor(self) -> int | None:
        raw = db.get_meta(self.conn, CURSOR_KEY)
        if raw:
            return int(raw)
        available = self.reader.available()
        return available.start_ms if available else None

    def set_cursor(self, value: int) -> None:
        db.set_meta(self.conn, CURSOR_KEY, str(value))

    def _note(self, state: str, done: int | None = None, total: int | None = None) -> None:
        """Record what the analyzer is doing, for the dashboard.

        The heartbeat is written on every probe, not just on state changes.
        Without that there is no way to tell a slow cold scan -- where each of
        forty-odd probes is a throttled network call -- apart from a hung one,
        and the first window of a fresh install looks dead for several minutes.
        """
        db.set_meta(self.conn, db.ANALYZER_STATE_KEY, state)
        db.set_meta(self.conn, db.ANALYZER_HEARTBEAT_KEY, str(int(time.time() * 1000)))
        db.set_meta(
            self.conn,
            db.ANALYZER_PROGRESS_KEY,
            f"{done}/{total}" if done is not None and total else "",
        )

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

    def _is_speech(self, window: Window, at_ms: int) -> bool:
        """Is this probe clearly not music, and so not worth a music lookup?

        Asking a music recogniser about presenter talk costs three network calls
        -- a miss, then both retry offsets -- and the answer was never going to
        be a song. On a talk-led morning show that is most of the probes in a
        window, which is why the scan could not keep pace with the stream that
        fed it.

        The reading has to be free and it has to abstain. LightSegmenter is
        both: no model to load, and it reports speech only above a threshold it
        is biased against reaching, so anything ambiguous still goes out to the
        recogniser. Being wrong here costs one missed identification of a song
        the local index will learn on its next airing anyway.
        """
        if not settings.load(self.conn)["skip_lookups_for_speech"]:
            return False
        probe = window.probe8(at_ms)
        if probe.size == 0:
            return False
        return self._gate.classify(probe, ANALYSIS_RATE) == seg.SPEECH

    def _recognizer_degraded(self) -> bool:
        return bool(getattr(self.recognizer, "degraded", False))

    def _pace_recognizer(self) -> None:
        """Pick up the configured call interval without a restart."""
        if not hasattr(self.recognizer, "min_interval"):
            return
        wanted = float(settings.load(self.conn)["recognizer_interval_seconds"])
        if wanted != self.recognizer.min_interval:
            log.info("external lookups now paced at %.0fs apart", wanted)
            self.recognizer.min_interval = wanted

    def _programme(self, at_ms: int):
        """Whatever the station says is on air then, or None."""
        try:
            moment = datetime.fromtimestamp(at_ms / 1000, tz=UTC).astimezone(
                self.config.source_tz
            )
            return self.schedule.at(moment)
        except Exception as exc:  # the grid is a nicety, never a blocker
            log.debug("schedule unavailable: %s", exc)
            return None

    def _plays_long_enough(self, key: str | None, duration_ms: int) -> bool:
        """Did this song actually play, or was it only touched on?

        A sting, a bed under a link, or a track trailed before the break all
        match as confidently as the real airing -- the fingerprint cannot tell
        them apart, because they are the same recording. What separates them is
        how much of the song ran.
        """
        if key is None:
            return False
        expected = self._expected_ms(key)
        if expected:
            return duration_ms >= expected * MIN_SONG_FRACTION
        return duration_ms >= MIN_SONG_MS

    def _expected_ms(self, key: str) -> int | None:
        """How long this song is supposed to run, per iTunes/Deezer."""
        row = db.get_song_meta(self.conn, key)
        value = row["duration_ms"] if row else None
        return int(value) if value else None

    def _edge_match(self, window: Window, at_ms: int, key: str) -> bool:
        """Is this short probe still `key`? Used only for locating boundaries."""
        probe = window.probe8(at_ms, BISECT_PROBE_MS)
        if probe.size == 0:
            return False
        match = self.index.match(
            fingerprint.compute(probe), kind="music", min_votes=BISECT_MIN_VOTES
        )
        return match is not None and match.key == key

    def _same_track(
        self, window: Window, at_ms: int, key: str, min_score: float = MIN_LOCAL_SCORE
    ) -> bool:
        match = self._local(window, at_ms, min_score)
        return match is not None and match.key == key

    def _start_from_offsets(self, window: Window, run: Run) -> int | None:
        """Where the song started, read straight off the match offsets.

        Once a reference is anchored at a song's start, a match reports how far
        into the song the probe was -- measured accurate to about one frame. So
        every probe in the run independently says where the song began, and the
        median of them beats any binary search. Only works from a song's second
        airing on; the first has nothing anchored to measure against.
        """
        if run.label is None:
            return None
        track = self.index.get(run.label.track_id)
        if track is None or not track["song_anchored"]:
            return None

        estimates: list[int] = []
        position = run.first_ms
        while position <= run.last_ms and window.covers(position):
            match = self._local(window, position)
            if match is not None and match.key == run.key:
                estimates.append(position - int(match.offset_seconds * 1000))
            position += STEP_MS
        if len(estimates) < 3:
            return None

        estimates.sort()
        spread = estimates[-1] - estimates[0]
        if spread > OFFSET_AGREEMENT_MS:
            log.debug("offset estimates disagree by %.1fs; falling back", spread / 1000)
            return None
        return estimates[len(estimates) // 2]

    # --- committing ---

    def _commit(self, item: Item) -> None:
        if self._extends_previous(item):
            return
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
                "show_title": item.show_title,
                "show_lead": item.show_lead,
                "show_image": item.show_image,
                "confidence": item.confidence,
                "source": item.source,
                "cluster_id": None,
            },
            int(time.time() * 1000),
        )

    def _extends_previous(self, item: Item) -> bool:
        """Grow the item before this one rather than writing its twin.

        The scan settles a gap in pieces -- a sting it identified and then
        rejected as too short to be a play, a stretch split across two passes --
        and each piece arrives here separately. Written as it comes, one
        continuous programme becomes three rows with the same name, abutting.

        Songs are exempt. Two airings of the same track are two events, and
        merging them would turn a repeat into one impossibly long play.
        """
        if item.kind == S.KIND_CANCION:
            return False
        previous = db.previous_timeline(self.conn, item.start_ms)
        if previous is None or previous["kind"] != item.kind:
            return False
        # Only a gap small enough to be a seam counts as touching.
        seam_ms = int(settings.load(self.conn)["max_seam_seconds"] * 1000)
        if item.start_ms - previous["end_ms"] > seam_ms:
            return False
        if (previous["show_title"] or None) != (item.show_title or None):
            return False
        if previous["end_ms"] >= item.end_ms:
            return True  # already covered; writing it again would duplicate
        db.set_timeline_end(self.conn, previous["id"], item.end_ms)
        return True

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

    def _identify(
        self, window: Window, at_ms: int, *, retry: bool = True, gate: bool = True
    ) -> Label | None:
        """Name the audio at `at_ms`, independently of any neighbouring probe.

        `retry` is for callers who are asking a question a miss already answers.
        Shifting along and asking again earns its keep during the scan, where a
        miss writes off a whole 24s stretch -- but not where the miss is the
        expected result.

        `gate` skips the speech check. It belongs to the scan, which is asking
        the open question "is anything playing here?", and speech is a fair
        answer to that. It does not belong to a caller asking whether one known
        song is still running: a presenter talking over an outro reads as speech
        while the song plays on underneath, and ending the song there is the
        error the check would cause.
        """
        match = self._local(window, at_ms)
        if match is not None:
            self.index.bump(match.track_id)
            return Label(
                key=match.key,
                artist=match.artist or "",
                title=match.title or "",
                source="local",
                confidence=_confidence(match),
                track_id=match.track_id,
            )

        if gate and self._is_speech(window, at_ms):
            return None

        found = self._external(window, at_ms)
        if found is None and retry and not self._recognizer_degraded():
            # The same song can match at one offset and miss at another, so a
            # single miss is not evidence of silence. Shift along and ask again
            # before writing the stretch off.
            #
            # Only worth doing while the recogniser is actually answering. When
            # it is not, asking twice more turns every probe into three dead
            # calls, and the scan pays the timeout three times over for nothing.
            for retry in RETRY_OFFSETS_MS:
                if not window.covers(at_ms + retry):
                    break
                found = self._external(window, at_ms + retry)
                if found is not None:
                    break
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

    def _last_match(self, window: Window, low: int, high: int, key: str) -> int:
        """Latest position in [low, high] that still reads as `key`."""
        while high - low > BISECT_LIMIT_MS:
            middle = (low + high) // 2
            if not window.covers(middle, BISECT_PROBE_MS):
                break
            if self._edge_match(window, middle, key):
                low = middle
            else:
                high = middle
        return low

    def _first_match(self, window: Window, low: int, high: int, key: str) -> int:
        """Earliest position in [low, high] that reads as `key`."""
        while high - low > BISECT_LIMIT_MS:
            middle = (low + high) // 2
            if not window.covers(middle, BISECT_PROBE_MS):
                break
            if self._edge_match(window, middle, key):
                high = middle
            else:
                low = middle
        return high

    def _boundary(self, window: Window, earlier: Run, later: Run) -> int:
        """Where one run gives way to the next.

        Both edges are located, not one: the last moment the outgoing song can
        still be heard and the first the incoming song can. On a hard cut those
        coincide. Across a crossfade they do not -- both songs are genuinely
        present for a few seconds -- and the honest boundary is the middle of
        that overlap rather than an edge picked arbitrarily from one side.

        A probe starting at t covers the next BISECT_PROBE_MS, so it stops
        reading as the outgoing song about half a probe before the change; half
        a probe is added back. With a two second probe that correction is one
        second, where the twelve second probe this used to take made it six.
        """
        # Run extents come from twelve second probes, which read as a song while
        # most of the probe lies inside it -- so each extent is soft by up to
        # half a probe, and the true change can sit outside the bracket they
        # form. Searching only between them caps the answer at whichever extent
        # overshot: a backward extension reaching half a probe into the outgoing
        # song pulled every boundary that far early, shortening the song by five
        # seconds however precise the bisection. The edge probes are two seconds
        # long, so looking half a probe further out costs a step and no accuracy.
        margin = PROBE_MS // 2
        low = max(earlier.last_ms - margin, earlier.first_ms)
        high = min(later.first_ms + margin, later.last_ms + PROBE_MS)
        if high < low:
            low, high = earlier.last_ms, later.first_ms
        correction = BISECT_PROBE_MS // 2

        if earlier.key is None and later.key is None:
            return later.first_ms
        if earlier.key is None:
            return max(self._first_match(window, low, high, later.key) + correction,
                       earlier.first_ms)
        if later.key is None:
            return min(self._last_match(window, low, high, earlier.key) + correction,
                       later.first_ms + PROBE_MS)

        last_outgoing = self._last_match(window, low, high, earlier.key)
        first_incoming = self._first_match(window, low, high, later.key)
        if first_incoming < last_outgoing:
            log.debug(
                "crossfade of %.1fs between %s and %s",
                (last_outgoing - first_incoming) / 1000, earlier.key, later.key,
            )
        middle = (last_outgoing + first_incoming) // 2
        return max(low, min(middle + correction, high + PROBE_MS))

    def _probe_count(self, window: Window) -> int:
        span = window.end_ms - window.start_ms - PROBE_MS
        return max(1, span // STEP_MS + 1)

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

        return merged

    def _ground_reference(self, window: Window, run: Run) -> None:
        """Teach the stretch already heard, before asking where it reaches.

        Extension asks "is this still the same song?" every six seconds. That is
        a question the local index answers in milliseconds -- but only about
        audio it has a reference for, and at this point a freshly identified
        song has nothing but the twelve second probe that named it. So every
        step missed locally and fell through to the network: eight lookups per
        song, per window, to establish an edge.

        Learning the run's own extent first costs one pass over audio already in
        memory and turns all eight into local answers. Once per song per window
        is enough; the extension that follows only widens what is learned here,
        and the commit relearns the refined span properly.
        """
        if run.key is None or run.label is None or run.key in self._grounded:
            return
        self._grounded.add(run.key)

        track = self.index.get(run.label.track_id)
        if track is None:
            return
        # A song carrying a reference learned from real boundaries already
        # answers locally, and adding the coarse extent on top of it would only
        # blur what the edge search depends on.
        if track["song_anchored"] and track["learned_ms"]:
            return

        start_ms = run.first_ms
        end_ms = min(run.last_ms + PROBE_MS, window.end_ms)
        if end_ms <= start_ms:
            return
        span = window.probe8(start_ms, end_ms - start_ms)
        if span.size == 0:
            return

        # Added, not swapped in. This is provisional coverage to make extension
        # answerable locally -- it is not a claim about where the song begins or
        # ends, so it must not mark the reference song-anchored or record itself
        # as the learned span. Doing either would let this coarse extent stand
        # in for the refined one the commit teaches, which is how a reference
        # erodes: each pass learning from the last pass's approximation.
        anchor_ms = int(track["anchor_ms"] or 0)
        shift = int(round((start_ms - anchor_ms) / 1000 * fingerprint.FRAMES_PER_SECOND))
        self.index.extend(run.label.track_id, fingerprint.compute(span), shift)

    def _extend_run(self, window: Window, run: Run) -> None:
        """Find where a run really reaches, not where the 24s grid landed.

        The scan only asks every 24 seconds, so a song's last matching probe can
        sit a full step short of its actual end. Everything downstream inherits
        that: the reference is learned from this extent, and the edge search
        cannot confirm the song past what the reference covers. Stepping out in
        six second increments until identification stops costs a handful of
        probes -- local ones, once the song is known -- and removes the bias.

        Each edge asks the index first and pays for an answer only when the
        index has none -- see `_continues`. Each direction gets its own small
        budget, so a song costs a few calls to bound properly rather than the
        eight it once spent asking questions it already had answers to.
        """
        if run.key is None:
            return

        budget = [EXTEND_EXTERNAL_BUDGET]
        reach = run.last_ms
        while reach - run.last_ms < EXTEND_LIMIT_MS:
            candidate = reach + EXTEND_STEP_MS
            if not window.covers(candidate):
                break
            if not self._continues(window, candidate, run.key, budget):
                break
            reach = candidate
        run.last_ms = reach

        budget = [EXTEND_EXTERNAL_BUDGET]
        start = run.first_ms
        while run.first_ms - start < EXTEND_LIMIT_MS:
            candidate = start - EXTEND_STEP_MS
            if candidate < window.start_ms or not window.covers(candidate):
                break
            if not self._continues(window, candidate, run.key, budget):
                break
            start = candidate
        run.first_ms = start

    def _continues(
        self, window: Window, at_ms: int, key: str, budget: list[int]
    ) -> bool:
        """Is `key` still playing at `at_ms`?

        The index answers for free wherever it has been taught. At an edge it
        usually has not been: extension runs before the song is learned, so the
        reference is only whatever named it. A local miss there is therefore not
        evidence the song ended -- it may just mean we have run out of
        reference, and treating the two the same is what put every edge short.

        So when the index has nothing to say, buy one answer, up to a fixed
        budget per edge. Once the run is committed the whole aired span is
        learned, and later airings resolve the same edges locally throughout.
        """
        if self._same_track(window, at_ms, key):
            return True
        if budget[0] <= 0 or self._recognizer_degraded():
            return False
        budget[0] -= 1
        found = self._identify(window, at_ms, retry=False, gate=False)
        return found is not None and found.key == key

    def _learn_once(self, window: Window, run: Run, start_ms: int, end_ms: int) -> None:
        """Teach a span, unless this window already taught one covering it."""
        if run.key is None or run.key in self._settled:
            return
        known = self._learned.get(run.key)
        if known is not None and known[0] <= start_ms and known[1] >= end_ms:
            return
        # Never trade a longer reference for a shorter one. replace() clears
        # what is there, so a downgrade permanently loses coverage the edge
        # search depends on.
        if run.label is not None:
            existing = self.index.get(run.label.track_id)
            if (
                existing is not None
                and existing["song_anchored"]
                and existing["learned_ms"]
                and existing["learned_ms"] > end_ms - start_ms
            ):
                log.debug(
                    "keeping the longer reference for %s (%.0fs over %.0fs)",
                    run.key, existing["learned_ms"] / 1000, (end_ms - start_ms) / 1000,
                )
                return
        self._relearn(window, run, start_ms, end_ms)
        self._learned[run.key] = (start_ms, end_ms)

    def _relearn(self, window: Window, run: Run, start_ms: int, end_ms: int) -> None:
        """Replace the reference with the whole aired span of this song."""
        if run.label is None:
            return
        span = window.probe8(start_ms, end_ms - start_ms)
        if span.size == 0:
            return
        self.index.replace(
            run.label.track_id,
            fingerprint.compute(span),
            anchor_ms=start_ms,
            learned_ms=end_ms - start_ms,
        )

    def _plan(self, window: Window, probes: list[tuple[int, Label | None]]) -> list[tuple[Run, int, int]]:
        """Group probes into runs and work out where each one starts and ends."""
        runs = self._group(probes)
        if not runs:
            return []

        # Teach the index each song over the whole stretch it was heard across,
        # before going looking for its edges. Bisection asks "is this still the
        # same song?", and with only the twelve second probe that first
        # identified it as a reference the answer was no almost everywhere, so
        # the boundary collapsed onto the last coarse probe -- which is what
        # pinned every song to a multiple of the scan step.
        for run in runs:
            self._ground_reference(window, run)
            self._extend_run(window, run)

        for position, run in enumerate(runs):
            if run.key is None:
                continue
            # A song already learned from refined boundaries -- on an earlier
            # airing or an earlier pass -- has a better reference than anything
            # the raw probe extent could teach. Overwriting it with the coarse
            # one shrinks what the edge search can confirm, so the song comes
            # out shorter, and shorter again the next time round.
            if run.label is not None:
                existing = self.index.get(run.label.track_id)
                if existing is not None and existing["song_anchored"]:
                    continue
            extent = (run.first_ms, run.last_ms + PROBE_MS)
            known = self._learned.get(run.key)
            # Already covered -- including by the wider, boundary-refined span
            # a commit teaches, which supersedes the run's raw probe extent.
            if known is not None and known[0] <= extent[0] and known[1] >= extent[1]:
                continue
            # The run at the end of the scan grows with every probe. Relearning
            # it each time would re-fingerprint minutes of audio on every step,
            # costing far more than the scan; it only has to be good enough to
            # answer boundary questions, so it is refreshed in chunks. A run the
            # scan has moved past is final and worth learning exactly.
            still_growing = position == len(runs) - 1
            if still_growing and known is not None and extent[1] - known[1] < RELEARN_GROWTH_MS:
                continue
            self._learn_once(window, run, *extent)

        # One shared boundary per adjacent pair, so items are contiguous and
        # never overlap -- a radio stream has no gaps between one thing and the next.
        edges = [window.start_ms]
        for position in range(len(runs) - 1):
            edges.append(self._boundary(window, runs[position], runs[position + 1]))
        edges.append(window.end_ms)

        # A song heard before can say where it started directly, by reading the
        # offset off its own match rather than searching for the edge. Measured
        # accurate to about one frame, so it wins wherever it is available and
        # broadly agrees with the edge we found. Adjusting the shared edge keeps
        # the timeline contiguous.
        for position in range(1, len(runs)):
            told = self._start_from_offsets(window, runs[position])
            if told is None:
                continue
            if abs(told - edges[position]) > OFFSET_TRUST_MS:
                log.debug(
                    "offset start %s disagrees with edge %s by %.1fs; keeping the edge",
                    told, edges[position], abs(told - edges[position]) / 1000,
                )
                continue
            if edges[position - 1] < told < edges[position + 1]:
                edges[position] = told

        keep = runs[:-1] if len(runs) > 1 else runs
        segments = [(run, edges[i], edges[i + 1]) for i, run in enumerate(keep)]
        live = settings.load(self.conn)
        return self._absorb_slivers(
            segments,
            int(live["max_seam_seconds"] * 1000),
            int(live["min_nonmusic_seconds"] * 1000),
        )

    def process_window(self, window: Window) -> int:
        """Scan the window, publishing each item as soon as it is settled.

        The scan is the slow part on a cold index -- forty-odd probes, each a
        throttled network call. Waiting for all of them before writing anything
        meant a fresh install showed an empty dashboard for several minutes and
        then everything at once. Runs far enough behind the scan position can no
        longer change, so they are committed as we go.
        """
        self._pace_recognizer()
        probes: list[tuple[int, Label | None]] = []
        self._learned.clear()
        self._settled.clear()
        self._grounded.clear()
        self._held = False
        total = self._probe_count(window)
        emitted_to = window.start_ms
        position = window.start_ms
        done = 0

        while window.covers(position):
            probes.append((position, self._identify(window, position)))
            done += 1
            position += STEP_MS
            self._note("scanning", done, total)
            # Grouping can still merge a run backwards across a short gap, so
            # only publish what the scan has moved safely past.
            settled = position - MAX_BRIDGE_MS - STEP_MS
            if settled > emitted_to:
                emitted_to = self._publish(window, probes, emitted_to, settled)

        emitted_to = self._publish(window, probes, emitted_to, None)

        # Held audio is unfinished business. Leaving the cursor short of it
        # means the next pass looks again, by which time the recogniser has
        # usually recovered -- and with hours of buffer in hand, waiting costs
        # nothing that writing the wrong answer would not cost more.
        if self._held:
            return emitted_to

        runs = self._group(probes)
        if len(runs) > 1:
            plan = self._plan(window, probes)
            if plan and plan[-1][2] > window.start_ms:
                return plan[-1][2]
        return window.end_ms

    def _publish(
        self,
        window: Window,
        probes: list[tuple[int, Label | None]],
        emitted_to: int,
        settled_before: int | None,
    ) -> int:
        """Commit every planned item that is finished and not already written.

        Except anything unnamed while the recogniser is down. A stretch nobody
        could identify looks exactly like a stretch with no song in it, and the
        difference is the whole meaning of the item: one is the programme, the
        other is a song we failed to ask about. Committing the first when it was
        really the second writes the failure into the timeline as fact, and the
        cursor moves past it, so it is never revisited.
        """
        for run, start_ms, end_ms in self._plan(window, probes):
            if start_ms < emitted_to:
                continue
            if settled_before is not None and end_ms > settled_before:
                break
            if run.key is None and self._recognizer_degraded():
                log.info(
                    "holding %.0fs at %s: the recogniser is down and this may be a song",
                    (end_ms - start_ms) / 1000, _clock(start_ms),
                )
                self._held = True
                break
            self._commit_run(window, run, start_ms, end_ms)
            emitted_to = end_ms
        return emitted_to

    def _absorb_slivers(
        self, segments: list[tuple[Run, int, int]], seam_ms: int, nonmusic_ms: int
    ) -> list[tuple[Run, int, int]]:
        """Fold away anything too small to stand on its own.

        What survives is decided by duration alone. A song has to have run for a
        real share of itself; a stretch between songs has to last longer than a
        presenter draws breath. Everything else is the join between two tracks,
        and belongs to them rather than to itself.

        Short gaps are closed by splitting them down the middle so the songs
        abut. Longer ones -- still too short to name -- are given to the song
        that follows, which is where a fade more often belongs.

        Judging this by how many probes a stretch spanned rather than how long
        it lasted is what once made an eighteen second link vanish: it fell
        inside a single probe, so it measured as zero.
        """
        cleaned: list[tuple[Run, int, int]] = []
        carried_start: int | None = None
        for index, (run, start_ms, end_ms) in enumerate(segments):
            if carried_start is not None:
                start_ms, carried_start = carried_start, None

            duration = end_ms - start_ms
            if run.key is not None:
                fragment = not self._plays_long_enough(run.key, duration)
            else:
                fragment = duration < nonmusic_ms

            if fragment:
                has_next = index + 1 < len(segments)
                if has_next and cleaned and duration <= seam_ms:
                    middle = start_ms + duration // 2
                    previous_run, previous_start, _ = cleaned[-1]
                    cleaned[-1] = (previous_run, previous_start, middle)
                    carried_start = middle
                    continue
                if has_next:
                    carried_start = start_ms
                    continue
                if cleaned:
                    previous_run, previous_start, _ = cleaned[-1]
                    cleaned[-1] = (previous_run, previous_start, end_ms)
                    continue
            cleaned.append((run, start_ms, end_ms))
        return cleaned

    def _close_seam(self, start_ms: int) -> int:
        """Meet the previous item in the middle if only a seam separates us.

        Songs are separated by crossfades, and locating each edge independently
        leaves a few seconds belonging to neither -- too short to be an advert
        or anything else worth naming, but long enough that a player shows the
        previous song through it, since nothing tells it otherwise. Anything
        below max_seam_seconds is split down the middle so the two songs abut.
        """
        previous = db.previous_timeline(self.conn, start_ms)
        if previous is None:
            return start_ms
        seam = start_ms - previous["end_ms"]
        threshold = int(settings.load(self.conn)["max_seam_seconds"] * 1000)
        if not 0 < seam <= threshold:
            return start_ms
        middle = previous["end_ms"] + seam // 2
        db.set_timeline_end(self.conn, previous["id"], middle)
        log.debug("closed a %.1fs seam before %s", seam / 1000, _clock(start_ms))
        return middle

    def _between_songs(self, start_ms: int, end_ms: int) -> Item:
        """Name a stretch that is not a song.

        It is whatever programme the station says was on air. That is the whole
        judgement: no attempt to tell an advert from a presenter link, because
        the two cannot be told apart reliably and naming the show is true
        either way. A listener seeing the right programme during an advert is a
        far smaller error than seeing "Publicidad" over the presenter talking.
        """
        programme = self._programme(start_ms)
        if programme is None:
            return Item(
                start_ms=start_ms,
                end_ms=end_ms,
                kind=S.KIND_DESCONOCIDO,
                source="gap",
            )
        return Item(
            start_ms=start_ms,
            end_ms=end_ms,
            kind=S.KIND_PROGRAMA,
            art_url=programme.image_url,
            show_title=programme.title,
            show_lead=programme.lead,
            show_image=programme.image_url,
            confidence=0.5,
            source="schedule",
        )

    def _commit_run(self, window: Window, run: Run, start_ms: int, end_ms: int) -> None:
        if end_ms <= start_ms:
            return
        start_ms = self._close_seam(start_ms)
        if not self._plays_long_enough(run.key, end_ms - start_ms):
            self._commit(
                self._between_songs(start_ms, end_ms)
            )
            return

        assert run.label is not None
        confidence = run.label.confidence
        expected = self._expected_ms(run.key) if run.key else None
        if expected:
            drift = abs((end_ms - start_ms) - expected) / expected
            if drift > DURATION_DISAGREEMENT:
                # Radio edits are genuinely shorter than the release, so this is
                # a reason to doubt the boundaries rather than to move them.
                confidence = round(confidence * 0.6, 3)
                log.info(
                    "%s - %s ran %.0fs against a release of %.0fs (%.0f%% out)",
                    run.label.artist, run.label.title,
                    (end_ms - start_ms) / 1000, expected / 1000, drift * 100,
                )
        item = self._enriched(
            Item(
                start_ms=start_ms,
                end_ms=end_ms,
                kind=S.KIND_CANCION,
                title=run.label.title,
                artist=run.label.artist,
                confidence=confidence,
                source=run.label.source,
            )
        )
        self._commit(item)
        self._learn_once(window, run, start_ms, end_ms)
        if run.key:
            self._settled.add(run.key)
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


def _confidence(match: fingerprint.Match) -> float:
    """How much to trust a local match.

    Raw score alone is misleading: a short probe of a long reference scores low
    even when it is unmistakably right. What separates a real match from a
    coincidence is the margin -- how far ahead the winner is of the next best
    alignment. A handful of votes spread evenly is noise; a hundred votes all
    agreeing on one offset is not.
    """
    by_score = min(1.0, match.score / 0.5)
    by_margin = min(1.0, match.margin / 5.0)
    return round(0.4 * by_score + 0.6 * by_margin, 3)


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
