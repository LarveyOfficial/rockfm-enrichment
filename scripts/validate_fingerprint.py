"""Measure the fingerprinter's recall and precision against real music.

Indexes 30-second previews of known songs, then queries with excerpts -- clean,
and again after a round-trip through 48 kbps AAC with noise added, which is
roughly what the broadcast chain does to audio. Also checks that a song which
was never indexed, and pure noise, are correctly left unmatched.

Downloads previews from iTunes on first run and caches them.

    python scripts/validate_fingerprint.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from rockfm import audio, db, fingerprint  # noqa: E402

SONGS = [
    "blondie denis",
    "acdc highway to hell",
    "queen bohemian rhapsody",
    "nirvana smells like teen spirit",
    "the police roxanne",
    "dire straits sultans of swing",  # held out: must never match
]
OFFSETS = (2, 8, 15)
PROBE_SECONDS = 10.0
CACHE = Path(os.environ.get("FP_CACHE", "/tmp/rockfm-fp-previews"))
DATABASE = Path(os.environ.get("FP_DB", "/tmp/rockfm-fp.db"))


def preview(term: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / (term.replace(" ", "_") + ".m4a")
    if destination.exists():
        return destination
    query = urllib.parse.urlencode({"term": term, "entity": "song", "limit": 1})
    with urllib.request.urlopen(f"https://itunes.apple.com/search?{query}", timeout=20) as response:
        result = json.load(response)["results"][0]
    urllib.request.urlretrieve(result["previewUrl"], destination)
    print(f"  fetched {result['artistName']} - {result['trackName']}")
    return destination


def broadcastify(samples: np.ndarray) -> np.ndarray:
    """Approximate what the broadcast chain does: lossy encode plus noise."""
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "in.raw"
        encoded = Path(directory) / "out.aac"
        (samples * 32767).astype("<i2").tofile(raw)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le",
             "-ar", "8000", "-ac", "1", "-i", str(raw), "-c:a", "aac",
             "-b:a", "48k", str(encoded)],
            check=True,
        )
        degraded = audio.decode_path(encoded)
    noise = np.random.default_rng(0).normal(0, 0.002, degraded.shape).astype(np.float32)
    return degraded + noise


def excerpt(samples: np.ndarray, start: float) -> np.ndarray:
    begin = int(start * audio.ANALYSIS_RATE)
    return samples[begin : begin + int(PROBE_SECONDS * audio.ANALYSIS_RATE)]


def main() -> None:
    print("fetching previews ...")
    samples = {song: audio.decode_path(preview(song)) for song in SONGS}

    DATABASE.unlink(missing_ok=True)
    index = fingerprint.FingerprintIndex(db.connect(DATABASE))
    indexed, held_out = SONGS[:-1], SONGS[-1]

    print("\nindexing references:")
    for song in indexed:
        hashes = fingerprint.compute(samples[song])
        index.add(kind="music", key=song, hashes=hashes, title=song, source="preview")
        print(f"  {song:<32} {len(samples[song]) / audio.ANALYSIS_RATE:5.1f}s  {len(hashes):6d} hashes")

    def sweep(label: str, transform) -> tuple[int, int]:
        print(f"\n--- {label}")
        hits = total = 0
        for song in indexed:
            audio_samples = transform(samples[song])
            for start in OFFSETS:
                match = index.match(fingerprint.compute(excerpt(audio_samples, start)), kind="music")
                correct = match is not None and match.key == song
                hits += correct
                total += 1
                detail = (
                    f"votes={match.votes:4d} score={match.score:.3f} offset={match.offset_seconds:+.2f}s"
                    if match
                    else ""
                )
                print(f"  {song:<32} @{start:>2}s  {'HIT ' if correct else 'MISS'} {detail}")
        print(f"  => {hits}/{total} correct")
        return hits, total

    sweep("clean excerpts", lambda x: x)
    sweep("through a 48 kbps AAC round-trip with noise", broadcastify)

    print(f"\n--- a song that was never indexed ({held_out})")
    false_positives = 0
    for start in OFFSETS:
        match = index.match(fingerprint.compute(excerpt(samples[held_out], start)), kind="music")
        false_positives += match is not None
        print(f"  @{start:>2}s  {'FALSE POSITIVE: ' + match.key if match else 'correctly unmatched'}")

    noise = np.random.default_rng(1).normal(0, 0.1, 10 * audio.ANALYSIS_RATE).astype(np.float32)
    noise_match = index.match(fingerprint.compute(noise), kind="music")
    print(f"\n--- pure noise\n  {'FALSE POSITIVE' if noise_match else 'correctly unmatched'}")
    print(f"\nfalse positives: {false_positives + bool(noise_match)}")


if __name__ == "__main__":
    main()
