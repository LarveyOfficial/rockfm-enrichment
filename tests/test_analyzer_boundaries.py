"""Song boundaries must come from the audio, not from the scan geometry.

A live instance once emitted three consecutive songs at exactly 198.0s each --
Foreigner (5:00), Kaiser Chiefs (3:25) and AC/DC (3:29) -- perfectly abutting.
Nothing about that came from the music: 198 is 8 x the 24s scan step plus half a
probe. Two faults combined. Windows were short enough for one song to fill one,
so its edges fell on the window rather than on a change; and the only reference
a freshly identified song had was the single 12s probe that found it, so the
bisection that is supposed to locate the edge could not confirm the song
anywhere else and collapsed onto the last coarse probe.

This drives the real grouping and boundary code over a synthetic timeline, with
recognition stubbed, and checks the edges track the songs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm import db
from rockfm.analyzer import (
    BISECT_PROBE_MS,
    MIN_WINDOW_MS,
    PROBE_MS,
    Analyzer,
    Label,
    Window,
)
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer

RATE = 16_000
# Deliberately different lengths; the bug made them all identical.
SONGS = [("foreigner", 300), ("ruby", 205), ("tnt", 209), ("meatloaf", 302)]


def build(tmp_path):
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())


@pytest.fixture
def played():
    spans, at = [], 0
    for name, seconds in SONGS:
        spans.append((at, at + seconds * 1000, name))
        at += seconds * 1000
    return spans


def song_at(spans, ms):
    for start, end, name in spans:
        if start <= ms < end:
            return name
    return None


def run_analyzer(tmp_path, spans, passes=4):
    analyzer = build(tmp_path)

    def identify(_window, at_ms):
        name = song_at(spans, at_ms)
        if name is None:
            return None
        return Label(key=name, artist="x", title=name, source="stub",
                     confidence=1.0, track_id=1)

    # Once a run is learned, "is this still the same song?" is answerable
    # anywhere inside it: a probe is that song if most of it lies within.
    def same_track(_window, at_ms, key, min_score=0.0):
        return song_at(spans, at_ms + PROBE_MS // 2) == key

    # Boundary probes are much shorter, which is the whole point of them.
    def edge_match(_window, at_ms, key):
        return song_at(spans, at_ms + BISECT_PROBE_MS // 2) == key

    committed: list = []
    analyzer._identify = identify
    analyzer._same_track = same_track
    analyzer._edge_match = edge_match
    analyzer._enriched = lambda item: item
    analyzer._commit = committed.append
    analyzer._relearn = lambda *args: None

    cursor = 0
    for _ in range(passes):
        end = cursor + MIN_WINDOW_MS
        samples = np.zeros(int((end - cursor) / 1000 * RATE), dtype=np.float32)
        cursor = analyzer.process_window(Window(cursor, end, samples))
    return committed


def test_durations_are_not_locked_to_the_scan_geometry(tmp_path, played):
    songs = [i for i in run_analyzer(tmp_path, played) if i.title]
    durations = {round(i.duration_ms / 1000, 1) for i in songs}
    assert len(songs) >= 3
    assert len(durations) > 1, f"every song came out the same length: {durations}"


def test_each_song_lands_close_to_its_real_length(tmp_path, played):
    truth = dict(SONGS)
    for item in run_analyzer(tmp_path, played):
        if not item.title or item.title not in truth:
            continue
        # Short boundary probes should land these far tighter than the twelve
        # second ones did; five seconds is a deliberately loose ceiling.
        assert item.duration_ms / 1000 == pytest.approx(truth[item.title], abs=5)


def test_items_never_overlap_and_leave_no_holes(tmp_path, played):
    items = sorted(run_analyzer(tmp_path, played), key=lambda i: i.start_ms)
    for earlier, later in zip(items, items[1:], strict=False):
        assert later.start_ms == earlier.end_ms, "timeline must be contiguous"


def test_no_slivers_are_emitted_between_tracks(tmp_path, played):
    """A rewound window re-sees a couple of seconds of the song just committed."""
    items = run_analyzer(tmp_path, played)
    interior = [i for i in items if i.start_ms > 0 and i.end_ms < 1_000_000]
    assert all(i.duration_ms >= 20_000 for i in interior), [
        (i.title or i.kind, i.duration_ms / 1000) for i in interior
    ]


def test_absorb_slivers_folds_a_leading_tail_into_what_follows():
    from rockfm.analyzer import Run

    label_a = Label(key="a", artist="x", title="a", source="s", confidence=1, track_id=1)
    tail = Run(key="a", label=label_a, first_ms=0, last_ms=0)
    real = Run(key="b", label=label_a, first_ms=6_000, last_ms=200_000)
    cleaned = Analyzer._absorb_slivers(
        [(tail, 0, 2_000), (real, 2_000, 200_000)], seam_ms=10_000
    )
    assert len(cleaned) == 1
    assert cleaned[0][1] == 0 and cleaned[0][2] == 200_000


def test_a_short_unnamed_stretch_between_songs_is_split_between_them():
    """A crossfade belongs to neither song, so let them meet in the middle."""
    from rockfm.analyzer import Run

    label_a = Label(key="a", artist="x", title="a", source="s", confidence=1, track_id=1)
    label_b = Label(key="b", artist="x", title="b", source="s", confidence=1, track_id=2)
    song_a = Run(key="a", label=label_a, first_ms=0, last_ms=100_000)
    seam = Run(key=None, label=None, first_ms=100_000, last_ms=100_000)
    song_b = Run(key="b", label=label_b, first_ms=106_000, last_ms=300_000)

    cleaned = Analyzer._absorb_slivers(
        [(song_a, 0, 100_000), (seam, 100_000, 106_000), (song_b, 106_000, 300_000)],
        seam_ms=10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", "b"]
    assert cleaned[0][2] == cleaned[1][1] == 103_000


def test_a_presenter_link_between_songs_survives():
    """Eighteen seconds of talk fell inside a single probe and was discarded."""
    from rockfm.analyzer import Run

    label_a = Label(key="a", artist="x", title="a", source="s", confidence=1, track_id=1)
    label_b = Label(key="b", artist="x", title="b", source="s", confidence=1, track_id=2)
    song_a = Run(key="a", label=label_a, first_ms=0, last_ms=100_000)
    link = Run(key=None, label=None, first_ms=100_000, last_ms=100_000)
    song_b = Run(key="b", label=label_b, first_ms=118_100, last_ms=300_000)

    cleaned = Analyzer._absorb_slivers(
        [(song_a, 0, 100_000), (link, 100_000, 118_100), (song_b, 118_100, 300_000)],
        seam_ms=10_000,
    )
    assert [run.key for run, _s, _e in cleaned] == ["a", None, "b"]
    assert cleaned[1][2] - cleaned[1][1] == pytest.approx(18_100)


# --- match offsets, retries, confidence, duration sanity -------------------


def test_confidence_rewards_margin_over_raw_score():
    """A short probe of a long reference scores low even when unmistakable."""
    from rockfm.analyzer import _confidence
    from rockfm.fingerprint import Match

    def match(score, margin):
        return Match(track_id=1, key="k", kind="music", title="t", artist="a",
                     votes=100, offset_frames=0, score=score, margin=margin)

    decisive = _confidence(match(score=0.08, margin=40))   # few hashes, no rival
    ambiguous = _confidence(match(score=0.30, margin=1.1))  # good score, close call
    assert decisive > ambiguous


def test_offset_start_is_ignored_without_an_anchored_reference(tmp_path):
    """A song's first airing has nothing anchored to measure against."""
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    track_id = analyzer.index.add(
        kind="music", key="a|b", hashes=[(1, 0), (2, 1)], title="b", anchor_ms=1000
    )
    label = Label(key="a|b", artist="a", title="b", source="s",
                  confidence=1.0, track_id=track_id)
    run = Run(key="a|b", label=label, first_ms=0, last_ms=100_000)
    samples = np.zeros(200 * RATE, dtype=np.float32)
    assert analyzer._start_from_offsets(Window(0, 200_000, samples), run) is None


def test_replace_marks_a_reference_song_anchored(tmp_path):
    analyzer = build(tmp_path)
    track_id = analyzer.index.add(
        kind="music", key="a|b", hashes=[(1, 0)], title="b", anchor_ms=1000
    )
    assert analyzer.index.get(track_id)["song_anchored"] == 0
    analyzer.index.replace(track_id, [(1, 0), (2, 5)], anchor_ms=500, learned_ms=210_000)
    row = analyzer.index.get(track_id)
    assert row["song_anchored"] == 1
    assert row["learned_ms"] == 210_000
    assert row["anchor_ms"] == 500


def test_a_probe_that_comes_back_empty_is_retried_elsewhere(tmp_path):
    """The same song matches at one offset and misses at another."""
    from rockfm.analyzer import RETRY_OFFSETS_MS
    from rockfm.recognize.base import Recognition

    analyzer = build(tmp_path)
    asked: list[int] = []

    def external(_window, at_ms):
        asked.append(at_ms)
        if at_ms < RETRY_OFFSETS_MS[0]:
            return None
        return Recognition(artist="Blondie", title="Denis", provider="stub")

    analyzer._external = external
    analyzer._local = lambda *a, **k: None
    samples = np.zeros(200 * RATE, dtype=np.float32)
    found = analyzer._identify(Window(0, 200_000, samples), 0)

    assert found is not None and found.title == "Denis"
    assert len(asked) > 1, "gave up after a single miss"


def test_items_are_published_during_the_scan_not_only_at_the_end(tmp_path, played):
    """A cold window is minutes of throttled calls; waiting for all of them
    meant an empty dashboard and then everything at once."""
    analyzer = build(tmp_path)
    seen_at_probe: list[int] = []
    probes = {"n": 0}

    def identify(_window, at_ms):
        probes["n"] += 1
        name = song_at(played, at_ms)
        if name is None:
            return None
        return Label(key=name, artist="x", title=name, source="stub",
                     confidence=1.0, track_id=1)

    analyzer._identify = identify
    analyzer._same_track = lambda _w, at, key, s=0.0: song_at(played, at + PROBE_MS // 2) == key
    analyzer._edge_match = lambda _w, at, key: song_at(played, at + BISECT_PROBE_MS // 2) == key
    analyzer._enriched = lambda item: item
    analyzer._relearn = lambda *a: None
    analyzer._commit = lambda item: seen_at_probe.append(probes["n"])

    samples = np.zeros(int(MIN_WINDOW_MS / 1000 * RATE), dtype=np.float32)
    analyzer.process_window(Window(0, MIN_WINDOW_MS, samples))

    assert seen_at_probe, "nothing was committed at all"
    total = probes["n"]
    assert min(seen_at_probe) < total, (
        f"first item only appeared after every probe ({min(seen_at_probe)}/{total})"
    )


def test_progress_is_recorded_while_scanning(tmp_path, played):
    analyzer = build(tmp_path)
    analyzer._identify = lambda _w, at: None
    samples = np.zeros(int(MIN_WINDOW_MS / 1000 * RATE), dtype=np.float32)
    analyzer.process_window(Window(0, MIN_WINDOW_MS, samples))

    progress = db.get_meta(analyzer.conn, db.ANALYZER_PROGRESS_KEY)
    done, total = progress.split("/")
    assert int(done) == int(total) > 1


def test_a_run_is_only_relearned_when_its_extent_changes(tmp_path, played):
    """Planning runs after nearly every probe; re-fingerprinting minutes of
    audio each time would cost more than the scan itself."""
    analyzer = build(tmp_path)
    relearns: list[tuple] = []

    def identify(_window, at_ms):
        name = song_at(played, at_ms)
        if name is None:
            return None
        return Label(key=name, artist="x", title=name, source="stub",
                     confidence=1.0, track_id=1)

    analyzer._identify = identify
    analyzer._same_track = lambda _w, at, key, s=0.0: song_at(played, at + PROBE_MS // 2) == key
    analyzer._edge_match = lambda _w, at, key: song_at(played, at + BISECT_PROBE_MS // 2) == key
    analyzer._enriched = lambda item: item
    analyzer._commit = lambda item: None
    analyzer._relearn = lambda w, run, a, b: relearns.append((run.key, a, b))

    samples = np.zeros(int(MIN_WINDOW_MS / 1000 * RATE), dtype=np.float32)
    analyzer.process_window(Window(0, MIN_WINDOW_MS, samples))

    probes = analyzer._probe_count(Window(0, MIN_WINDOW_MS, samples))
    # Unmemoised this is one per run per probe -- well over a hundred.
    assert len(relearns) < probes / 2, f"{len(relearns)} relearns over {probes} probes"
    # A growing run legitimately refreshes 0-84s, then 0-180s, and so on. What
    # would be wasted work is relearning something an *earlier* pass already
    # covered.
    for index, (key, start, end) in enumerate(relearns):
        already = [
            (a, b) for k, a, b in relearns[:index]
            if k == key and a <= start and b >= end
        ]
        assert not already, f"{key} {start}-{end} was already covered by {already}"


def test_a_reference_is_never_replaced_with_a_shorter_one(tmp_path):
    """Re-analysing must not erode what it already knows.

    replace() clears a track's hashes before writing new ones, so learning a
    coarse probe extent over a boundary-refined span loses coverage the edge
    search depends on -- the song then measures shorter, and shorter again on
    the next pass. Observed live: Fleetwood Mac went 173.9s, then 155.9s, with
    the gap after it growing by exactly the difference.
    """
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    label = Label(key="a|b", artist="a", title="b", source="s",
                  confidence=1.0, track_id=0)
    track_id = analyzer.index.add(
        kind="music", key="a|b", hashes=[(1, 0)], title="b", anchor_ms=0
    )
    label = Label(key="a|b", artist="a", title="b", source="s",
                  confidence=1.0, track_id=track_id)
    run = Run(key="a|b", label=label, first_ms=0, last_ms=170_000)

    # A good, refined reference: the whole song.
    analyzer.index.replace(track_id, [(1, 0), (2, 10)], anchor_ms=0, learned_ms=174_000)
    assert analyzer.index.get(track_id)["learned_ms"] == 174_000

    # A later pass offers a shorter, coarser span. It must be refused.
    attempted: list = []
    analyzer._relearn = lambda w, r, a, b: attempted.append((a, b))
    analyzer._learn_once(None, run, 0, 156_000)

    assert attempted == [], "a shorter reference overwrote a longer one"
    assert analyzer.index.get(track_id)["learned_ms"] == 174_000


def test_a_longer_reference_still_wins(tmp_path):
    from rockfm.analyzer import Run

    analyzer = build(tmp_path)
    track_id = analyzer.index.add(
        kind="music", key="a|b", hashes=[(1, 0)], title="b", anchor_ms=0
    )
    analyzer.index.replace(track_id, [(1, 0)], anchor_ms=0, learned_ms=100_000)
    label = Label(key="a|b", artist="a", title="b", source="s",
                  confidence=1.0, track_id=track_id)
    run = Run(key="a|b", label=label, first_ms=0, last_ms=200_000)

    attempted: list = []
    analyzer._relearn = lambda w, r, a, b: attempted.append((a, b))
    analyzer._learn_once(None, run, 0, 210_000)
    assert attempted == [(0, 210_000)]
