"""Song boundaries come from the recogniser, not from a search.

The analyzer used to find edges by hunting for them: teach a local fingerprint
index a song's extent, walk outwards asking whether the song continued, then
bisect with short probes. It resolved to 400 ms, cost about a thousand local
matches per probe, and only ever worked properly from a song's second airing.

The recogniser reports the position directly, in the same answer that names the
song. Measured across one airing of a Mr. Big track, probes thirty seconds
apart came back with offsets 30.001 s apart and the spread over the whole song
was three milliseconds. So a start is arithmetic now, and these tests drive the
real grouping and layout code over stubbed recognition to check it holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm import db
from rockfm.analyzer import (
    MIN_WINDOW_MS,
    PROBE_MS,
    STEP_MS,
    Analyzer,
    Label,
    Window,
)
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer, Recognition
from rockfm.rockfm_api import catalog_key

RATE = 16_000


def build(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


def window():
    samples = np.zeros(int(MIN_WINDOW_MS / 1000 * RATE), dtype=np.float32)
    return Window(0, MIN_WINDOW_MS, samples)


def label(key, started_ms, artist="Mr. Big", title="To Be with You"):
    return Label(key=key, artist=artist, title=title, source="shazamio",
                 confidence=1.0, started_ms=started_ms)


def release(analyzer, artist, title, ms):
    db.upsert_song_meta(
        analyzer.conn,
        {"key": catalog_key(artist, title), "artist": artist, "title": title,
         "duration_ms": ms},
        0,
    )


# --- the probe reports where the song began --------------------------------


def test_a_named_probe_says_where_the_recording_started(tmp_path):
    analyzer = build(tmp_path)
    analyzer._external = lambda _w, at: Recognition(
        artist="Mr. Big", title="To Be with You", provider="shazamio",
        offset_seconds=51.35, track_id="284861603",
    )

    found = analyzer._identify(window(), 60_000)
    assert found is not None
    assert found.key == "284861603", "identity must be the recogniser's own id"
    assert found.started_ms == 60_000 - 51_350


def test_a_probe_with_no_offset_still_names_the_song(tmp_path):
    """Not every recogniser reports a position; the song is still the song."""
    analyzer = build(tmp_path)
    analyzer._external = lambda _w, at: Recognition(
        artist="Mr. Big", title="To Be with You", provider="audd",
    )

    found = analyzer._identify(window(), 60_000)
    assert found is not None and found.started_ms is None


def test_a_shifted_retry_measures_from_where_it_asked(tmp_path):
    """The offset is relative to the probe that answered, not the one that failed.

    A probe straddling a change matches neither side, so the scan shifts along
    and asks again. Measuring the answer from the original position would put
    the start wrong by exactly that shift.
    """
    analyzer = build(tmp_path)
    answers = {65_000: Recognition(artist="Mr. Big", title="To Be with You",
                                   provider="shazamio", offset_seconds=10.0)}
    analyzer._external = lambda _w, at: answers.get(at)

    found = analyzer._identify(window(), 60_000)
    assert found is not None
    assert found.started_ms == 65_000 - 10_000


# --- turning several reports into one start --------------------------------


def test_the_start_is_the_median_of_what_the_probes_said(tmp_path):
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    run = Run(key="k", label=label("k", 40_000), first_ms=48_000, last_ms=96_000)
    run.starts = [40_100, 40_000, 39_900]

    assert analyzer._run_start(run, window()) == 40_000


def test_a_wild_start_is_not_trusted_over_the_probe_grid(tmp_path):
    """A start after the probe that heard it, or outside the window, is wrong."""
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    w = window()

    late = Run(key="k", label=label("k", 90_000), first_ms=48_000, last_ms=96_000)
    late.starts = [90_000]                       # after the probe that heard it
    assert analyzer._run_start(late, w) == 48_000

    before = Run(key="k", label=label("k", -5_000), first_ms=48_000, last_ms=96_000)
    before.starts = [-5_000]                     # before the window opens
    assert analyzer._run_start(before, w) == 48_000


def test_a_run_with_no_reported_start_falls_back_to_its_first_probe(tmp_path):
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    run = Run(key="k", label=label("k", None), first_ms=48_000, last_ms=96_000)
    assert analyzer._run_start(run, window()) == 48_000


def test_the_end_comes_from_the_release_length(tmp_path):
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    release(analyzer, "Mr. Big", "To Be with You", 208_000)
    run = Run(key="k", label=label("k", 10_000), first_ms=24_000, last_ms=48_000)

    assert analyzer._run_end(run, 10_000, window()) == 218_000


def test_a_song_heard_past_its_release_length_keeps_what_was_heard(tmp_path):
    """Radio edits and live versions run to their own length, not the sleeve's."""
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    release(analyzer, "Mr. Big", "To Be with You", 60_000)
    run = Run(key="k", label=label("k", 0), first_ms=0, last_ms=300_000)

    assert analyzer._run_end(run, 0, window()) == 300_000 + PROBE_MS


# --- laying the window out -------------------------------------------------


def plan_for(analyzer, spans):
    """spans: (key, artist, title, told_start_ms, first_probe, last_probe)."""
    probes: list[tuple[int, Label | None]] = []
    for key, artist, title, told, first, last in spans:
        position = first
        while position <= last:
            probes.append((position, label(key, told, artist, title)))
            position += STEP_MS
    return analyzer._plan(window(), probes)


def test_songs_are_placed_where_they_said_they_began(tmp_path):
    analyzer = build(tmp_path)
    release(analyzer, "A", "one", 200_000)
    release(analyzer, "B", "two", 200_000)

    plan = plan_for(analyzer, [
        ("a", "A", "one", 10_000, 24_000, 192_000),
        ("b", "B", "two", 215_000, 216_000, 384_000),
    ])
    placed = {run.key: (start, end) for run, start, end in plan if run.key}

    assert placed["a"][0] == 10_000
    assert placed["b"][0] == 215_000


def test_two_songs_never_overlap(tmp_path):
    """A release length that overruns is trimmed by the next song's own start.

    The start is measured; the length is metadata, and the station's copy may
    be shorter. Where they disagree, the measurement wins.
    """
    analyzer = build(tmp_path)
    release(analyzer, "A", "one", 400_000)      # far longer than it actually ran
    release(analyzer, "B", "two", 200_000)

    plan = plan_for(analyzer, [
        ("a", "A", "one", 0, 24_000, 96_000),
        ("b", "B", "two", 180_000, 192_000, 336_000),
    ])
    spans = [(start, end) for _run, start, end in plan]
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier[1] <= later[0], f"{earlier} overlaps {later}"


def test_the_space_between_songs_becomes_a_gap(tmp_path):
    analyzer = build(tmp_path)
    release(analyzer, "A", "one", 100_000)
    release(analyzer, "B", "two", 100_000)

    plan = plan_for(analyzer, [
        ("a", "A", "one", 0, 24_000, 72_000),
        ("b", "B", "two", 200_000, 216_000, 264_000),
    ])
    kinds = [run.key for run, _s, _e in plan]
    assert None in kinds, "the stretch between two songs was not marked as a gap"


def test_the_stretch_running_to_the_window_edge_is_held_back(tmp_path):
    """It is still growing; the next pass sees more audio and finishes it."""
    analyzer = build(tmp_path)
    release(analyzer, "A", "one", 100_000)

    plan = plan_for(analyzer, [("a", "A", "one", 0, 24_000, 72_000)])
    assert all(end < MIN_WINDOW_MS for _run, _start, end in plan)


# --- folding away what is too small to stand alone -------------------------


def test_absorb_slivers_folds_a_leading_tail_into_what_follows(tmp_path):
    from rockfm.analyzer import Run

    tail = Run(key="a", label=label("a", 0), first_ms=0, last_ms=0)
    real = Run(key="b", label=label("b", 0), first_ms=6_000, last_ms=200_000)
    cleaned = build(tmp_path)._absorb_slivers(
        [(tail, 0, 2_000), (real, 2_000, 200_000)], 10_000, 10_000
    )
    assert len(cleaned) == 1
    assert cleaned[0][1] == 0 and cleaned[0][2] == 200_000


def test_a_short_unnamed_stretch_between_songs_is_split_between_them(tmp_path):
    """A crossfade belongs to neither song, so let them meet in the middle."""
    from rockfm.analyzer import Run

    song_a = Run(key="a", label=label("a", 0), first_ms=0, last_ms=100_000)
    seam = Run(key=None, label=None, first_ms=100_000, last_ms=106_000)
    song_b = Run(key="b", label=label("b", 0), first_ms=106_000, last_ms=300_000)

    cleaned = build(tmp_path)._absorb_slivers(
        [(song_a, 0, 100_000), (seam, 100_000, 106_000), (song_b, 106_000, 300_000)],
        10_000, 10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", "b"]
    assert cleaned[0][2] == cleaned[1][1] == 103_000


def test_a_presenter_link_between_songs_survives(tmp_path):
    """Eighteen seconds of talk is a real thing that happened."""
    from rockfm.analyzer import Run

    song_a = Run(key="a", label=label("a", 0), first_ms=0, last_ms=100_000)
    link = Run(key=None, label=None, first_ms=100_000, last_ms=118_100)
    song_b = Run(key="b", label=label("b", 0), first_ms=118_100, last_ms=300_000)

    cleaned = build(tmp_path)._absorb_slivers(
        [(song_a, 0, 100_000), (link, 100_000, 118_100), (song_b, 118_100, 300_000)],
        10_000, 10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", None, "b"]
    assert cleaned[1][2] - cleaned[1][1] == pytest.approx(18_100)


# --- nothing may stand between a probe and the recogniser ------------------


def test_a_probe_always_reaches_the_recogniser(tmp_path):
    """A cheap speech check sat here once and silently dropped whole songs."""
    analyzer = build(tmp_path)
    asked: list[int] = []
    analyzer._external = lambda _w, at: asked.append(at)

    analyzer._identify(window(), 60_000)
    assert asked, "the probe never reached the recogniser"


def test_a_miss_is_asked_again_before_the_stretch_is_written_off(tmp_path):
    analyzer = build(tmp_path)
    asked: list[int] = []

    def external(_window, at_ms):
        asked.append(at_ms)
        return None

    analyzer._external = external
    assert analyzer._identify(window(), 60_000) is None
    assert len(asked) > 1, "one miss condemned the whole stretch"

    asked.clear()
    assert analyzer._identify(window(), 60_000, retry=False) is None
    assert len(asked) == 1


def test_nothing_imports_a_fingerprint_index() -> None:
    """Both modules are gone; an import means the old design is creeping back."""
    import importlib

    for name in ("rockfm.fingerprint", "rockfm.seed", "rockfm.classify.segmenter"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
