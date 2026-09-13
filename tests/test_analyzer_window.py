"""The analyzer must not nibble at the live edge.

Once it has caught up, ingest only adds six seconds at a time. A window barely
longer than one probe gets exactly one probe, so a single unmatched probe
condemns the whole stretch to "unknown" and the cursor moves past it for good.
With hours of delay in hand there is no reason to work that way.

These tests use fake segment rows, so decoding yields nothing and no song is
ever found. What they check is whether the analyzer *decided to look*, which is
visible in the cursor: skipped windows leave it untouched.
"""

from __future__ import annotations

import time

import pytest

from rockfm import db
from rockfm.analyzer import CURSOR_KEY, MIN_WINDOW_MS, PROBE_MS, TAIL_GUARD_MS, Analyzer
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer


@pytest.fixture
def build(tmp_path):
    def _build(delay_seconds: int, recorded_ms: int) -> Analyzer:
        config = Config(data_dir=tmp_path, delay_seconds_override=delay_seconds)
        config.ensure_dirs()
        conn = db.connect(config.db_path)
        start = int(time.time() * 1000) - recorded_ms
        for index in range(recorded_ms // 6000):
            db.insert_segment(
                conn, pdt_ms=start + index * 6000, seq=index, duration_ms=6000,
                relpath=f"{index}.aac", size=1, fetched_ms=0,
            )
        return Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())

    return _build


def looked(analyzer: Analyzer) -> bool:
    analyzer.run_once()
    return db.get_meta(analyzer.conn, CURSOR_KEY) is not None


def test_waits_rather_than_analysing_a_sliver_at_the_live_edge(build):
    # Six hours of delay: two minutes of fresh audio is no reason to work yet.
    assert not looked(build(6 * 3600, 120_000 + TAIL_GUARD_MS))


def test_analyses_once_enough_audio_has_accumulated(build):
    assert looked(build(6 * 3600, MIN_WINDOW_MS + TAIL_GUARD_MS + 60_000))


def test_analyses_a_sliver_anyway_when_it_is_about_to_air(build):
    # A tiny delay means this audio airs almost immediately; waiting would miss it.
    assert looked(build(0, 120_000 + TAIL_GUARD_MS))


def test_does_nothing_with_less_than_two_probes_of_audio(build):
    assert not looked(build(0, PROBE_MS + TAIL_GUARD_MS))


def test_deadline_check_tracks_the_configured_delay(build):
    analyzer = build(6 * 3600, MIN_WINDOW_MS + TAIL_GUARD_MS)
    now_ms = int(time.time() * 1000)
    # Audio recorded a moment ago airs in six hours: no rush.
    assert not analyzer._must_analyze_now(now_ms)
    # Audio recorded six hours ago is airing about now.
    assert analyzer._must_analyze_now(now_ms - 6 * 3600 * 1000)
