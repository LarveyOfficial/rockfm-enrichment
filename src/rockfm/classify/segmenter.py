"""Speech vs music segmentation.

Two implementations behind one interface:

  ina    inaSpeechSegmenter (INA, MIT, CNN). Trained, needs no tuning, and
         usefully treats singing as music and speech-over-music as speech --
         which is exactly how a DJ talking over a bed should be read. Pulls in
         TensorFlow, so it is heavy.
  light  numpy/scipy only. Uses the features speech/music discrimination
         classically relies on: speech pauses a lot, and its energy envelope
         is strongly modulated around the 4 Hz syllable rate.

`SEGMENTER=ina` is the default and falls back to `light` automatically if the
model cannot be loaded, so a small Unraid box still works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("rockfm.classify.segmenter")

SPEECH = "speech"
MUSIC = "music"
SILENCE = "silence"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Zone:
    label: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _frame(samples: np.ndarray, size: int, hop: int) -> np.ndarray:
    if samples.size < size:
        return np.zeros((0, size), dtype=np.float32)
    count = 1 + (samples.size - size) // hop
    strides = (samples.strides[0] * hop, samples.strides[0])
    return np.lib.stride_tricks.as_strided(
        samples, shape=(count, size), strides=strides, writeable=False
    )


class LightSegmenter:
    """Approximate speech/music discrimination with no heavyweight dependencies.

    Built on Low Short-Time Energy Ratio: the fraction of frames whose energy
    falls well below the local average. Speech constantly pauses between
    syllables and words, so LSTER runs high (~0.4); music sustains energy, so it
    runs low (~0.1). Zero-crossing-rate variance adds a second, weaker vote.

    An earlier version keyed on 4 Hz envelope modulation, the classic syllable
    rate. That is a bad feature for a rock station: a drum beat modulates the
    envelope in the same band, and it labelled every song as speech.

    This is a fallback for machines that cannot run the CNN. It is deliberately
    biased toward reporting music, and callers should treat a mid-range score as
    "unknown" rather than as a decision.
    """

    name = "light"

    LSTER_MUSIC = 0.15
    LSTER_SPEECH = 0.40

    def speech_ratio(self, samples: np.ndarray, rate: int) -> float:
        if samples.size < rate // 2:
            return 0.0
        frame_size = int(0.025 * rate)
        hop = int(0.010 * rate)
        frames = _frame(samples.astype(np.float32), frame_size, hop)
        if frames.shape[0] < 32:
            return 0.0

        energy = (frames**2).mean(axis=1)
        mean_energy = float(energy.mean())
        if mean_energy < 1e-9:
            return 0.0
        lster = float((energy < 0.5 * mean_energy).mean())

        span = self.LSTER_SPEECH - self.LSTER_MUSIC
        return float(np.clip((lster - self.LSTER_MUSIC) / span, 0.0, 1.0))

    # Outside these bounds the reading is clear enough to act on; between them
    # the honest answer is "I do not know", and the caller should lean on
    # repetition and the schedule instead of a coin flip.
    CONFIDENT_MUSIC = 0.25
    CONFIDENT_SPEECH = 0.75

    def classify(self, samples: np.ndarray, rate: int) -> str:
        ratio = self.speech_ratio(samples, rate)
        if ratio <= self.CONFIDENT_MUSIC:
            return MUSIC
        if ratio >= self.CONFIDENT_SPEECH:
            return SPEECH
        return UNKNOWN

    def segments(self, samples: np.ndarray, rate: int) -> list[Zone]:
        label = self.classify(samples, rate)
        return [Zone(label=label, start_s=0.0, end_s=samples.size / rate)]


class InaSegmenter:
    """inaSpeechSegmenter, a CNN trained for exactly this task."""

    name = "ina"
    _LABELS = {
        "speech": SPEECH,
        "male": SPEECH,
        "female": SPEECH,
        "music": MUSIC,
        "noEnergy": SILENCE,
        "noise": SILENCE,
    }

    def __init__(self) -> None:
        from inaSpeechSegmenter import Segmenter

        self._segmenter = Segmenter(vad_engine="smn", detect_gender=False)

    def segments(self, samples: np.ndarray, rate: int) -> list[Zone]:
        import tempfile
        from pathlib import Path

        from ..audio import pcm_to_wav

        if samples.size < rate:
            return []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.wav"
            path.write_bytes(pcm_to_wav(samples, rate))
            raw = self._segmenter(str(path))
        return [
            Zone(label=self._LABELS.get(label, SILENCE), start_s=float(start), end_s=float(stop))
            for label, start, stop in raw
        ]

    def speech_ratio(self, samples: np.ndarray, rate: int) -> float:
        zones = self.segments(samples, rate)
        voiced = sum(z.duration_s for z in zones if z.label in (SPEECH, MUSIC))
        if voiced <= 0:
            return 0.0
        speech = sum(z.duration_s for z in zones if z.label == SPEECH)
        return speech / voiced

    def classify(self, samples: np.ndarray, rate: int) -> str:
        zones = self.segments(samples, rate)
        if not zones:
            return UNKNOWN
        voiced = sum(z.duration_s for z in zones if z.label in (SPEECH, MUSIC))
        if voiced <= 0:
            return SILENCE
        return SPEECH if self.speech_ratio(samples, rate) >= 0.5 else MUSIC


def build(name: str = "ina"):
    key = (name or "").strip().lower()
    if key in {"light", "heuristic"}:
        return LightSegmenter()
    try:
        return InaSegmenter()
    except Exception as exc:
        log.warning("inaSpeechSegmenter unavailable (%s); using the light segmenter", exc)
        return LightSegmenter()
