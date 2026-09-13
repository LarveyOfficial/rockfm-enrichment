"""Fingerprinter unit tests on deterministic synthetic audio.

Real-audio validation (iTunes previews put through a broadcast-like AAC
round-trip) lives in scripts/validate_fingerprint.py, which downloads on demand
and so is not suitable as a unit test.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from rockfm import db, fingerprint

SR = 8000


def synth(seed: int, seconds: float = 20.0) -> np.ndarray:
    """A deterministic 'music-like' signal: a run of harmonic notes."""
    rng = np.random.default_rng(seed)
    samples = np.zeros(int(seconds * SR), dtype=np.float64)
    note = 0.35
    for index in range(int(seconds / note)):
        f0 = float(rng.uniform(120, 900))
        start = int(index * note * SR)
        end = min(len(samples), start + int(note * SR))
        local = np.arange(end - start) / SR
        envelope = np.exp(-3 * local)
        for harmonic in range(1, 6):
            samples[start:end] += envelope * np.sin(2 * np.pi * f0 * harmonic * local) / harmonic
    samples += rng.normal(0, 0.01, samples.shape)
    return (samples / np.abs(samples).max()).astype(np.float32)


@pytest.fixture
def index(tmp_path) -> fingerprint.FingerprintIndex:
    conn: sqlite3.Connection = db.connect(tmp_path / "fp.db")
    return fingerprint.FingerprintIndex(conn)


def excerpt(samples: np.ndarray, start: float, duration: float = 8.0) -> np.ndarray:
    return samples[int(start * SR) : int((start + duration) * SR)]


def test_compute_is_deterministic():
    audio = synth(1, 5)
    assert fingerprint.compute(audio) == fingerprint.compute(audio)


def test_compute_returns_hashes_at_a_sane_density():
    hashes = fingerprint.compute(synth(1, 10))
    per_second = len(hashes) / 10
    assert 100 < per_second < 600


def test_short_audio_yields_no_hashes():
    assert fingerprint.compute(np.zeros(100, dtype=np.float32)) == []


def test_matches_itself(index):
    audio = synth(2, 20)
    index.add(kind="music", key="track-a", hashes=fingerprint.compute(audio), title="A")
    match = index.match(fingerprint.compute(excerpt(audio, 5)), kind="music")
    assert match is not None
    assert match.key == "track-a"
    assert match.votes >= fingerprint.DEFAULT_MIN_VOTES


@pytest.mark.parametrize("start", [0.0, 4.0, 9.0, 12.0])
def test_recovers_the_time_offset(index, start):
    audio = synth(3, 20)
    index.add(kind="music", key="track-b", hashes=fingerprint.compute(audio))
    match = index.match(fingerprint.compute(excerpt(audio, start)), kind="music")
    assert match is not None
    assert match.offset_seconds == pytest.approx(start, abs=0.1)


def test_does_not_match_a_different_track(index):
    index.add(kind="music", key="track-c", hashes=fingerprint.compute(synth(4, 20)))
    assert index.match(fingerprint.compute(excerpt(synth(5, 20), 5)), kind="music") is None


def test_does_not_match_noise(index):
    index.add(kind="music", key="track-d", hashes=fingerprint.compute(synth(6, 20)))
    noise = np.random.default_rng(7).normal(0, 0.1, 8 * SR).astype(np.float32)
    assert index.match(fingerprint.compute(noise), kind="music") is None


def test_kind_filter_isolates_the_two_indexes(index):
    audio = synth(8, 20)
    index.add(kind="nonmusic", key="jingle-1", hashes=fingerprint.compute(audio))
    probe = fingerprint.compute(excerpt(audio, 5))
    assert index.match(probe, kind="music") is None
    assert index.match(probe, kind="nonmusic") is not None


def test_repeat_airing_counts_without_duplicating_hashes(index):
    audio = synth(9, 10)
    hashes = fingerprint.compute(audio)
    track_id = index.add(kind="nonmusic", key="advert-1", hashes=hashes)
    assert index.occurrences(track_id) == 1
    stored = index._hash_count(track_id)

    again = index.add(kind="nonmusic", key="advert-1", hashes=hashes)
    assert again == track_id
    assert index.occurrences(track_id) == 2
    assert index._hash_count(track_id) == stored
