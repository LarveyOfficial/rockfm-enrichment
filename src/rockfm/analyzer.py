"""Work out what actually aired, and when.

Runs on recorded audio roughly six hours before it goes out, which is what
makes asking anything at all affordable: there is time to probe and look things
up long before any of it is heard.

The pass over a window is:

  1. scan at a coarse stride, asking the recogniser what is playing
  2. group consecutive probes naming the same recording into runs
  3. lay each run out: it begins where the recogniser said it began, and ends
     at its release length or where the next song starts, whichever comes first
  4. whatever is left between songs is not a song
  5. enrich and commit

Step 3 used to be most of this file. A local fingerprint index was taught each
song's extent, walked outwards to find where the song continued, and bisected
with short probes to place the edge. It resolved to 400 ms, only worked
properly from a song's second airing, and cost about a thousand local matches
per probe -- thirty seconds of arithmetic to place twenty-four seconds of
audio, with the network not involved at all.

None of that was necessary. A probe that names a song also reports how far into
it the probe was: measured across one airing, probes thirty seconds apart came
back 30.001 s apart, with three milliseconds of spread over the whole song. The
start is arithmetic, from the first probe that names the song, and it is far
more precise than the search ever was.

The index also made mistakes the search could not: the same recording could sit
in it twice, learned from air and seeded from the station's catalogue under a
different name, and a run broke in half wherever a probe matched the other
copy. Identity now comes from the recogniser's own id for the recording, which
does not vary between releases.

Probes are 12 s because that is what Shazam's signature format wants; see
recognize/shazam.py. A probe straddling a song change matches neither side,
which is a signal rather than a failure -- the scan shifts along and asks
again, and measures the answer from wherever it actually asked.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import numpy as np

from . import db, settings
from . import strings_es as S
from .buffer import BufferReader
from .config import Config
from .enrich import Enricher
from .recognize import Recognition
from .recognize import build as build_recognizer
from .rockfm_api import RockFmApi, catalog_key
from .schedule import Schedule
from .timeshift import target_delay_seconds

log = logging.getLogger("rockfm.analyzer")

RECOGNIZE_RATE = 16000
PROBE_MS = 12_000          # fixed by Shazam's signature format
STEP_MS = 24_000           # coarse scan stride; songs are far longer than this
# Where to look again when a probe comes back empty, before concluding there is
# no song there.
RETRY_OFFSETS_MS = (5_000, 10_000)
# How far a run may stray from the release length before we distrust it.
DURATION_DISAGREEMENT = 0.35
MIN_SONG_MS = 45_000       # floor for songs of unknown length
# A song has to run for a decent share of itself to count as having been played.
# Stations trail tracks over links and play stings built from them, and those
# match as confidently as the real airing -- length is what separates them.
MIN_SONG_FRACTION = 0.15
# How much longer than the release two adjacent, same-named items may be and
# still count as one airing split in two rather than the song played twice.
# Well under 2.0, which is what a genuine repeat would measure.
MERGE_LENGTH_TOLERANCE = 1.3
# How alike two names must read before they are treated as the same recording.
# Loose enough for a reissue's punctuation and parenthetical, tight enough that
# two different songs by one artist stay apart.
MERGE_NAME_RATIO = 0.82
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
    # Absolute wall clock of the moment this recording began, as the recogniser
    # reports it. This is the whole of the boundary machinery now: a probe that
    # names a song also says how far into it we are, to about three
    # milliseconds, so the start is arithmetic rather than a search.
    started_ms: int | None = None


@dataclass
class Run:
    key: str | None
    label: Label | None
    first_ms: int
    last_ms: int
    # Why extension stopped at each edge. Recorded because the difference
    # between "the song ended" and "we ran out of ways to ask" is invisible in
    # the finished timeline, and that is exactly what goes wrong.
    # Every probe in the run independently reports where the song began. They
    # agree to milliseconds, so the median of them is the start.
    starts: list[int] = field(default_factory=list)


class Window:
    """One decoded pass of the buffer, sliced in memory for every probe."""

    def __init__(self, start_ms: int, end_ms: int, samples16: np.ndarray) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.samples16 = samples16
    def _slice(self, samples: np.ndarray, rate: int, at_ms: int, length_ms: int) -> np.ndarray:
        begin = int((at_ms - self.start_ms) * rate / 1000)
        if begin < 0:
            return np.zeros(0, dtype=np.float32)
        end = begin + int(length_ms * rate / 1000)
        if end > samples.size:
            return np.zeros(0, dtype=np.float32)
        return samples[begin:end]

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
        # Set when a window ends with audio nobody could identify because the
        # recogniser was down. The cursor stops there rather than past it.
        self._held = False

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

    def _external(self, window: Window, at_ms: int) -> Recognition | None:
        probe = window.probe16(at_ms)
        if probe.size == 0:
            return None
        self.external_calls += 1
        return self.recognizer.recognize(probe, RECOGNIZE_RATE)

    def _recognizer_degraded(self) -> bool:
        return bool(getattr(self.recognizer, "degraded", False))

    def _publish_recognizer_state(self) -> None:
        """Say how the recogniser is doing, where another process can see it.

        It lives in this process and nothing else can ask it, so a stalled or
        refusing recogniser is invisible from the dashboard -- which is where
        the question always gets asked first.
        """
        import json

        recognizer = self.recognizer
        db.set_meta(
            self.conn,
            db.RECOGNIZER_STATE_KEY,
            json.dumps(
                {
                    "name": getattr(recognizer, "name", "unknown"),
                    "degraded": bool(getattr(recognizer, "degraded", False)),
                    "errors_in_a_row": int(getattr(recognizer, "_consecutive_errors", 0)),
                    "calls": int(getattr(recognizer, "calls", 0)),
                    "skipped": int(getattr(recognizer, "skipped", 0)),
                    "updated_ms": int(time.time() * 1000),
                }
            ),
        )

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

        Songs merge only under the stricter rule in `_one_airing_split_in_two`:
        two airings of the same track are two events, and merging those would
        turn a repeat into one impossibly long play.
        """
        if item.kind == S.KIND_CANCION:
            return self._one_airing_split_in_two(item)
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

    def _one_airing_split_in_two(self, item: Item) -> bool:
        """Is this the rest of the song already on the timeline?

        A single airing can come back under more than one name. "Dead Ringer
        for Love" arrived as three items, the middle one a different release of
        the same recording -- the fingerprint matched a reissue for a few
        probes and the run broke in two around it.

        Two tests, and both are needed. Similar names alone would merge a track
        played twice in an hour into one impossibly long play. Length alone
        would merge two different songs that happen to abut. Together they say:
        these are the same song, adjacent, and together they still only add up
        to about one airing of it.
        """
        previous = db.previous_timeline(self.conn, item.start_ms)
        if previous is None or previous["kind"] != S.KIND_CANCION:
            return False
        if not (item.artist and item.title and previous["artist"] and previous["title"]):
            return False

        seam_ms = int(settings.load(self.conn)["max_seam_seconds"] * 1000)
        if item.start_ms - previous["end_ms"] > seam_ms:
            return False
        if previous["end_ms"] >= item.end_ms:
            return True

        if not _alike(
            f'{previous["artist"]} {previous["title"]}', f"{item.artist} {item.title}"
        ):
            return False

        # One airing, or two? The release length decides.
        combined = item.end_ms - previous["start_ms"]
        expected = self._expected_ms(catalog_key(item.artist, item.title)) or (
            self._expected_ms(catalog_key(previous["artist"], previous["title"]))
        )
        if not expected:
            # Without a release length there is no way to tell one airing split
            # in two from the song played twice, and merging a repeat is the
            # worse mistake. Leave them alone.
            return False
        if combined > expected * MERGE_LENGTH_TOLERANCE:
            log.debug(
                "not merging %s: %.0fs together against a release of %.0fs",
                item.title, combined / 1000, expected / 1000,
            )
            return False

        db.set_timeline_end(self.conn, previous["id"], item.end_ms)
        log.info(
            "%s  merged a split airing of %s - %s (%.0fs)",
            _clock(previous["start_ms"]), previous["artist"], previous["title"],
            combined / 1000,
        )
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

    def _identify(self, window: Window, at_ms: int, *, retry: bool = True) -> Label | None:
        """Name the audio at `at_ms`, and say where that recording began.

        One question to one recogniser. There used to be a local fingerprint
        index in front of this, and most of the analyzer existed to serve it:
        teach it a song's extent, walk outwards asking whether the song
        continued, bisect the edges with short probes. All of that was a way of
        finding a boundary -- and the recogniser reports the boundary directly,
        in the same answer that names the song, an order of magnitude more
        precisely than bisection ever resolved.

        The index also made its own mistakes. The same recording could sit in
        it twice, learned from air and seeded from the station's catalogue under
        a different name, and a run would break in half where a probe matched
        the other copy. Identity now comes from the recogniser's own id for the
        recording, which does not vary between releases.
        """
        asked_at = at_ms
        found = self._external(window, at_ms)
        if found is None and retry and not self._recognizer_degraded():
            # A probe that straddles a change matches neither side. Shift along
            # and ask again before writing the stretch off.
            for shift in RETRY_OFFSETS_MS:
                if not window.covers(at_ms + shift):
                    break
                found = self._external(window, at_ms + shift)
                if found is not None:
                    asked_at = at_ms + shift
                    break
        if found is None:
            return None

        started_ms = None
        if found.offset_seconds is not None:
            started_ms = asked_at - int(round(found.offset_seconds * 1000))

        return Label(
            key=found.track_id or catalog_key(found.artist, found.title),
            artist=found.artist,
            title=found.title,
            source=found.provider,
            confidence=found.confidence,
            started_ms=started_ms,
        )

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
            if label is not None and label.started_ms is not None:
                runs[-1].starts.append(label.started_ms)

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

    def _run_start(self, run: Run, window: Window) -> int:
        """Where the song began, as its own probes reported it."""
        told = sorted(value for value in run.starts if value is not None)
        if told:
            start = told[len(told) // 2]
            # It has to sit at or before the probe that heard it and inside the
            # window. A wild answer is not worth trusting over the probe grid.
            if window.start_ms <= start <= run.first_ms:
                return start
        return max(run.first_ms, window.start_ms)

    def _run_end(self, run: Run, start_ms: int, window: Window) -> int:
        """Where it stopped: its release length, or at least what we heard.

        The recogniser places a start exactly but says nothing about length, so
        the end is the release length where one is known. Radio edits run
        shorter than the release and presenters talk over outros, so the next
        song's own start trims this wherever the two disagree -- and that start
        is measured, not predicted.
        """
        heard_to = min(run.last_ms + PROBE_MS, window.end_ms)
        expected = (
            self._expected_ms(catalog_key(run.label.artist, run.label.title))
            if run.label
            else None
        )
        if expected:
            return min(max(start_ms + expected, heard_to), window.end_ms)
        return heard_to

    def _plan(self, window: Window, probes: list[tuple[int, Label | None]]) -> list[tuple[Run, int, int]]:
        """Lay the window out from what the recogniser said.

        Songs are placed where they said they began. Whatever is left between
        them is not a song, and needs no examination to say so -- the previous
        design searched for every edge and spent a thousand local matches a
        probe doing it, thirty seconds of arithmetic to place twenty-four
        seconds of audio.
        """
        runs = self._group(probes)
        if not runs:
            return []

        songs: list[list] = []
        for run in runs:
            if run.key is None or run.label is None:
                continue
            start_ms = self._run_start(run, window)
            end_ms = self._run_end(run, start_ms, window)
            if end_ms > start_ms:
                songs.append([run, start_ms, end_ms])

        songs.sort(key=lambda item: item[1])
        # Two songs cannot play at once. Where a release length overran the next
        # song's start, the next song's own report of itself wins.
        for index in range(len(songs) - 1):
            if songs[index][2] > songs[index + 1][1]:
                songs[index][2] = songs[index + 1][1]

        segments: list[tuple[Run, int, int]] = []
        cursor = window.start_ms
        for run, start_ms, end_ms in songs:
            if start_ms > cursor:
                segments.append((_gap(cursor, start_ms), cursor, start_ms))
            segments.append((run, start_ms, end_ms))
            cursor = end_ms
        if cursor < window.end_ms:
            segments.append((_gap(cursor, window.end_ms), cursor, window.end_ms))

        # The stretch running to the window edge is still growing; the next pass
        # sees more audio and finishes it.
        if segments and segments[-1][2] >= window.end_ms:
            segments = segments[:-1]

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
        self._publish_recognizer_state()
        probes: list[tuple[int, Label | None]] = []
        self._held = False
        total = self._probe_count(window)
        emitted_to = window.start_ms
        position = window.start_ms
        done = 0

        while window.covers(position):
            if self._recognizer_degraded():
                log.info(
                    "pausing the scan at %s: the recogniser is out, and probing on"
                    " would spend calls we cannot make on answers we cannot trust",
                    _clock(position),
                )
                self._held = True
                break
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
                if not self._held:
                    log.info(
                        "holding %.0fs at %s: the recogniser is down and this may"
                        " be a song", (end_ms - start_ms) / 1000, _clock(start_ms),
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
        expected_note = f" of {expected / 1000:.0f}s" if expected else ""
        log.info(
            "%s  %s - %s  (%.0fs%s, %s)  edges: start=%s end=%s",
            _clock(item.start_ms), item.artist, item.title,
            item.duration_ms / 1000, expected_note, item.source,
            run.start_stopped or "-", run.end_stopped or "-",
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


def _alike(left: str, right: str) -> bool:
    """Do these read as the same recording under two names?

    Releases differ in punctuation, casing and parentheticals -- a remaster, a
    single edit, a compilation -- while naming the same audio.
    """
    from difflib import SequenceMatcher

    def tidy(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).strip()

    a, b = tidy(left), tidy(right)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= MERGE_NAME_RATIO


def _gap(start_ms: int, end_ms: int) -> Run:
    """A stretch with no song in it."""
    return Run(key=None, label=None, first_ms=start_ms, last_ms=end_ms)


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
