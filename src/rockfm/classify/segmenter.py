"""Is this audio speech, or music?

Only one implementation now, and deliberately the weaker one. This used to
decide whether a non-song stretch was an advert or a presenter -- a judgement
the system no longer makes, since stretches between songs are named from the
station's own schedule. A trained CNN (inaSpeechSegmenter) did that job and
brought TensorFlow with it.

What is left is a cheaper question: is this probe worth sending to a *music*
recogniser? Asking costs 0.2 ms; getting it wrong costs one missed
identification of a song the index learns on its next airing. Answering it with
a neural network would be spending a great deal to avoid very little.

Built on Low Short-Time Energy Ratio, and biased against reporting speech, so
anything ambiguous still goes out to be identified.
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


def build(_name: str = "light") -> LightSegmenter:
    """The only segmenter left.

    A CNN used to decide whether a non-song stretch was an advert or a
    presenter, which is a judgement the system no longer makes: stretches
    between songs are named from the station's own schedule. All that remains
    is deciding whether a probe is worth sending to a music recogniser, and
    that wants the cheap, abstaining answer rather than the accurate one.
    """
    return LightSegmenter()
