from datetime import datetime

import pytest

from rockfm import db
from rockfm.classify.decide import Classifier
from rockfm.classify.repetition import Cluster
from rockfm.classify.segmenter import MUSIC, SPEECH, UNKNOWN
from rockfm.config import Config
from rockfm.rockfm_api import Programme

NOON = int(datetime(2026, 9, 14, 10, 30).timestamp() * 1000)  # inside a show


class FakeSchedule:
    def __init__(self, no_break=False):
        self._no_break = no_break

    def at(self, moment):
        return Programme(
            title="El Pirata y su banda", lead="El Pirata", horario="",
            image_url="/img.jpg", description="", start_minute=360,
            end_minute=600, category_id=42,
        )

    def is_no_break_block(self, moment):
        return self._no_break


@pytest.fixture
def classifier(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    return Classifier(config, conn, schedule=FakeSchedule(), segmenter=object())


def repeat(n=2):
    return Cluster(track_id=1, key="cluster:1", occurrences=n)


def test_repeated_audio_is_an_advert(classifier):
    kind, _ = classifier._decide(NOON, SPEECH, repeat())
    assert kind == "publicidad"


def test_first_airing_is_not_guessed_as_an_advert(classifier):
    kind, _ = classifier._decide(NOON, SPEECH, repeat(n=1))
    assert kind == "programa"


def test_unrecognised_music_is_never_called_talk(classifier):
    kind, _ = classifier._decide(NOON, MUSIC, None)
    assert kind == "desconocido"


def test_repeats_during_a_no_break_block_name_the_show_instead(classifier):
    """The station bills that block as advert-free, so a repeat is a promo."""
    classifier.schedule = FakeSchedule(no_break=True)
    kind, _ = classifier._decide(NOON, SPEECH, repeat())
    assert kind == "programa"


def test_an_unreadable_stretch_falls_back_to_naming_the_show(classifier):
    kind, confidence = classifier._decide(NOON, UNKNOWN, None)
    assert kind == "programa"
    assert confidence < 0.5


def test_merge_joins_adjacent_chunks_of_the_same_kind(classifier):
    from rockfm.classify.decide import Chunk

    chunks = [
        Chunk(0, 15_000, "publicidad", 1, 0.7),
        Chunk(15_000, 30_000, "publicidad", None, 0.8),
        Chunk(30_000, 45_000, "programa", None, 0.6),
    ]
    merged = Classifier._merge(chunks)
    assert len(merged) == 2
    assert merged[0].end_ms == 30_000
    assert merged[0].confidence == 0.8
    assert merged[0].cluster_id == 1


def test_a_sliver_between_tracks_is_retired_without_a_verdict(classifier):
    """Station jingles run a couple of seconds; no useful label fits them."""
    now = 1_700_000_000_000
    db.upsert_timeline(
        classifier.conn,
        {"start_ms": now, "end_ms": now + 2_500, "kind": "desconocido", "source": "unresolved"},
        0,
    )
    assert classifier.run_once() == 1

    row = db.timeline_at(classifier.conn, now + 1_000)
    assert row["kind"] == "desconocido"
    assert row["source"] == "too-short"
    # And it is not rescanned on the next pass.
    assert classifier.run_once() == 0


def test_a_real_break_is_still_classified(classifier, monkeypatch):
    """Anything past the floor goes through the normal path."""
    now = 1_700_000_000_000
    db.upsert_timeline(
        classifier.conn,
        {"start_ms": now, "end_ms": now + 90_000, "kind": "desconocido", "source": "unresolved"},
        0,
    )
    seen = []
    monkeypatch.setattr(classifier, "_chunks", lambda a, b: seen.append((a, b)) or [])
    classifier.run_once()
    assert seen == [(now, now + 90_000)]


def test_the_floor_is_configurable_at_runtime(tmp_path):
    """Changing it on the dashboard must take effect without a restart."""
    from rockfm import db as database
    from rockfm import settings
    from rockfm.config import Config

    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = database.connect(config.db_path)
    classifier = Classifier(config, conn, schedule=FakeSchedule(), segmenter=object())
    assert classifier.min_nonmusic_ms == 5_000

    settings.save(conn, {"min_nonmusic_seconds": 20})
    assert classifier.min_nonmusic_ms == 20_000
