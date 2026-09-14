"""Repetition must count airings, not how many times we looked at them.

Re-analysing already-scanned audio finds each cluster again at the same moment
it was first heard. Counting that as a second airing promotes presenter talk to
an advert simply by re-running the analyzer over it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfm import db
from rockfm.classify.repetition import SAME_AIRING_MS, RepetitionIndex
from rockfm.config import Config


@pytest.fixture
def repetition(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    return RepetitionIndex(db.connect(config.db_path))


def talk(seed: int, seconds: float = 15.0) -> np.ndarray:
    """Speech-like audio: distinct enough to fingerprint, deterministic."""
    rng = np.random.default_rng(seed)
    samples = np.zeros(int(seconds * 8000), dtype=np.float64)
    for index in range(int(seconds / 0.2)):
        start = int(index * 0.2 * 8000)
        end = min(len(samples), start + int(0.2 * 8000))
        local = np.arange(end - start) / 8000
        f0 = float(rng.uniform(90, 300))
        for harmonic in range(1, 8):
            samples[start:end] += np.sin(2 * np.pi * f0 * harmonic * local) / harmonic
    samples += rng.normal(0, 0.01, samples.shape)
    return (samples / np.abs(samples).max()).astype(np.float32)


def test_a_first_airing_counts_once(repetition):
    assert repetition.observe(talk(1), 1_000_000).occurrences == 1


def test_rescanning_the_same_moment_is_not_a_repeat(repetition):
    """This is what the re-analyze button does to every cluster."""
    audio = talk(2)
    first = repetition.observe(audio, 5_000_000)
    again = repetition.observe(audio, 5_000_000)

    assert first.occurrences == 1
    assert again.occurrences == 1, "re-analysing counted as a second airing"
    assert not again.is_repeat


def test_a_nearby_sighting_is_the_same_airing(repetition):
    audio = talk(3)
    repetition.observe(audio, 5_000_000)
    nearby = repetition.observe(audio, 5_000_000 + SAME_AIRING_MS - 1_000)
    assert nearby.occurrences == 1


def test_a_genuine_second_airing_counts(repetition):
    audio = talk(4)
    repetition.observe(audio, 5_000_000)
    later = repetition.observe(audio, 5_000_000 + 3 * 3600 * 1000)

    assert later.occurrences == 2
    assert later.is_repeat, "a real repeat should be recognised"


def test_repeated_rescans_never_accumulate(repetition):
    audio = talk(5)
    for _ in range(6):
        cluster = repetition.observe(audio, 9_000_000)
    assert cluster.occurrences == 1
    assert not cluster.is_repeat


def test_different_audio_makes_a_different_cluster(repetition):
    first = repetition.observe(talk(6), 1_000_000)
    second = repetition.observe(talk(7), 1_000_000)
    assert first.track_id != second.track_id
