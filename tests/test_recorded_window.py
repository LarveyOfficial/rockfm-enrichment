"""Twenty real minutes of RockFM, replayed through the real analyzer.

Shazam's answers were recorded once, for every position the first window asks
about, so this runs offline and in milliseconds. It is the window that exposed
three faults at once: straddled starts thrown away, told starts that drift as
probes arrive, and moved boundaries left unwritten -- 181s and 252s of the
timeline simply missing.

The expectations are what Shazam itself said, not what the analyzer produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rockfm import db
from rockfm.analyzer import Analyzer, Window
from rockfm.config import Config
from rockfm.recognize.base import NullRecognizer, Recognition

FIXTURE = Path(__file__).parent / "fixtures" / "shazam_window_20260915.json"
WINDOW_MS = 1_200_000


@pytest.fixture
def timeline(tmp_path):
    answers = {int(k): v for k, v in json.loads(FIXTURE.read_text())["answers"].items()}
    config = Config(data_dir=tmp_path, delay_seconds_override=6 * 3600)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    analyzer = Analyzer(config, conn, recognizer=NullRecognizer(), enricher=object())
    analyzer._enriched = lambda item: item
    analyzer._programme = lambda _at: None

    def external(_window, at_ms):
        heard = answers.get(at_ms)
        if not heard:
            return None
        return Recognition(artist=heard["artist"], title=heard["title"], provider="shazamio",
                           offset_seconds=heard["offset"], track_id=str(heard["key"]))

    analyzer._external = external
    analyzer.process_window(Window(0, WINDOW_MS, np.zeros(16, dtype=np.float32)))
    return [dict(row) for row in conn.execute("SELECT * FROM timeline ORDER BY start_ms")]


def start_of(rows, title):
    matches = [row for row in rows if row["title"] == title]
    assert matches, f"{title} is not on the timeline at all"
    return matches[0]["start_ms"]


def test_nothing_is_missing_from_the_timeline(timeline):
    holes = [
        (earlier["end_ms"], later["start_ms"])
        for earlier, later in zip(timeline, timeline[1:], strict=False)
        if later["start_ms"] != earlier["end_ms"]
    ]
    assert not holes, f"stretches with no row at all: {holes}"


def test_a_song_starting_inside_its_first_probe_keeps_that_start(timeline):
    """You Really Got Me began 5.2s into the probe that first named it."""
    assert start_of(timeline, "You Really Got Me") == pytest.approx(418_700, abs=600)


def test_a_start_follows_the_probes_that_agree(timeline):
    """Ten probes said 851.3s; six others matched a different release."""
    assert start_of(timeline, "Nothing Else Matters") == pytest.approx(851_300, abs=600)


def test_an_edited_song_keeps_the_start_its_opening_probes_gave(timeline):
    """Four probes said 214.5s before the station's edit threw later offsets off."""
    assert start_of(timeline, "Gimme All Your Lovin'") == pytest.approx(214_500, abs=600)


def test_the_first_probes_cannot_be_overtaken_by_later_disagreement():
    """The cluster rule itself, away from the fixture."""
    from rockfm.analyzer import Label, Run

    config_dir = Path(__file__).parent  # unused by _run_start
    analyzer = object.__new__(Analyzer)
    run = Run(key="k", label=Label(key="k", artist="a", title="t", source="s",
                                   confidence=1.0), first_ms=840_000, last_ms=1_200_000)
    run.starts = [851_300] * 10 + [889_400] * 3 + [909_600] * 3
    window = Window(0, WINDOW_MS, np.zeros(16, dtype=np.float32))
    assert analyzer._run_start(run, window) == 851_300
    assert config_dir.exists()
