"""Audio decoding helpers.

Everything analysis-side runs on 8 kHz mono PCM, which is plenty for landmark
fingerprinting and keeps the spectrogram cheap. Playout never touches this path
-- it serves the original AAC bytes untouched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ANALYSIS_RATE = 8000


class DecodeError(RuntimeError):
    pass


def _run_ffmpeg(args: list[str], stdin: bytes | None = None, timeout: float = 120) -> bytes:
    process = subprocess.run(
        args, input=stdin, capture_output=True, timeout=timeout, check=False
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace").strip().splitlines()
        raise DecodeError(detail[-1] if detail else f"ffmpeg exited {process.returncode}")
    return process.stdout


def _to_float(raw: bytes) -> np.ndarray:
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _base_args(source: str, rate: int, start: float | None, duration: float | None) -> list[str]:
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", source]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(rate), "pipe:1"]
    return args


def decode_bytes(
    data: bytes,
    rate: int = ANALYSIS_RATE,
    start: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    return _to_float(_run_ffmpeg(_base_args("pipe:0", rate, start, duration), stdin=data))


def decode_path(
    path: str | Path,
    rate: int = ANALYSIS_RATE,
    start: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    return _to_float(_run_ffmpeg(_base_args(str(path), rate, start, duration)))


def decode_url(
    url: str,
    rate: int = ANALYSIS_RATE,
    start: float | None = None,
    duration: float | None = None,
    timeout: float = 60,
) -> np.ndarray:
    return _to_float(
        _run_ffmpeg(_base_args(url, rate, start, duration), timeout=timeout)
    )


def decode_segment_files(
    paths: list[Path],
    rate: int = ANALYSIS_RATE,
    start: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    """Decode consecutive packed-AAC segments as one continuous stream.

    ADTS frames concatenate cleanly, so the segments can simply be joined and
    handed to ffmpeg in one go.
    """
    blob = b"".join(path.read_bytes() for path in paths if path.exists())
    if not blob:
        return np.zeros(0, dtype=np.float32)
    return decode_bytes(blob, rate=rate, start=start, duration=duration)


def pcm_to_wav(samples: np.ndarray, rate: int = ANALYSIS_RATE) -> bytes:
    """Wrap mono float samples in a WAV container (recognisers want a file)."""
    import io
    import wave

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
