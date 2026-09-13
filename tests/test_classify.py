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
    def __init__(self, no_break=False, news=False):
        self._no_break = no_break
        self._news = news

    def at(self, moment):
        return Programme(
            title="El Pirata y su banda", lead="El Pirata", horario="",
            image_url="/img.jpg", description="", start_minute=360,
            end_minute=600, category_id=42,
        )

    def is_no_break_block(self, moment):
        return self._no_break

    def is_news_window(self, moment):
        return self._news


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


def test_speech_in_the_news_window_is_news(classifier):
    classifier.schedule = FakeSchedule(news=True)
    kind, _ = classifier._decide(NOON, SPEECH, repeat(n=1))
    assert kind == "noticias"


def test_repeats_during_a_no_break_block_are_idents_not_adverts(classifier):
    classifier.schedule = FakeSchedule(no_break=True)
    kind, _ = classifier._decide(NOON, SPEECH, repeat())
    assert kind == "sintonia"


def test_an_unreadable_stretch_falls_back_to_naming_the_show(classifier):
    kind, confidence = classifier._decide(NOON, UNKNOWN, None)
    assert kind == "programa"
    assert confidence < 0.5


def test_short_adverts_are_reclassified_as_idents(classifier):
    from rockfm.classify.decide import Chunk

    chunk = Chunk(start_ms=0, end_ms=10_000, kind="publicidad", cluster_id=1, confidence=0.8)
    assert classifier._refine(chunk).kind == "sintonia"


def test_a_very_long_news_run_is_really_the_programme(classifier):
    from rockfm.classify.decide import Chunk

    chunk = Chunk(start_ms=0, end_ms=600_000, kind="noticias", cluster_id=None, confidence=0.6)
    assert classifier._refine(chunk).kind == "programa"


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
