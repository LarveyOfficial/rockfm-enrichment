"""ShazamIO backend.

ShazamIO is a free, reverse-engineered client for Shazam's internal API: no key,
no quota. It is unofficial, so it may break without warning and its use may not
sit within Shazam's terms -- which is exactly why the local fingerprint index
does the heavy lifting and this is only consulted for audio we have not learned
yet.

IMPORTANT: `SearchParams(segment_duration_seconds=...)` must be set. The library
default produces signatures Shazam does not match, so leaving it off means every
lookup silently returns nothing.

There is a ceiling, not a magic number. Measured against three well-known
tracks, every length from 4 s to 14 s matched and 16 s and 20 s did not, so
Shazam accepts up to somewhere between 14 and 16 seconds. Shorter signatures
carry less evidence and fail sooner on obscure material -- an earlier reading of
this, taken from a single hard-to-match garage rock track where only 12 s
worked, wrongly concluded that 12 was the only accepted value.

12 s is used because it is the most evidence Shazam will take, which matters for
exactly the deep cuts this station plays.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ..audio import pcm_to_wav
from .base import Recognition

log = logging.getLogger("rockfm.recognize.shazam")

# Shazam's signature format assumes a 12 s sample. See the module docstring.
SEGMENT_SECONDS = 12


class ShazamRecognizer:
    name = "shazamio"

    def __init__(self) -> None:
        from shazamio import SearchParams, Shazam  # lazy so the dep stays optional

        self._shazam = Shazam()
        self._options = SearchParams(segment_duration_seconds=SEGMENT_SECONDS)

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        if samples.size == 0:
            return None
        try:
            payload = asyncio.run(self._recognize(pcm_to_wav(samples, rate)))
        except Exception as exc:
            log.warning("shazam lookup failed: %s", exc)
            return None
        return _parse(payload)

    async def _recognize(self, wav: bytes) -> dict[str, Any]:
        return await self._shazam.recognize(wav, options=self._options)

    def close(self) -> None:
        return None


def _parse(payload: dict[str, Any] | None) -> Recognition | None:
    if not payload:
        return None
    track = payload.get("track") or {}
    title = (track.get("title") or "").strip()
    artist = (track.get("subtitle") or "").strip()
    if not title or not artist:
        return None

    album = None
    for section in track.get("sections") or []:
        for item in section.get("metadata") or []:
            if (item.get("title") or "").strip().lower() in {"album", "álbum"}:
                album = (item.get("text") or "").strip() or None
    art = (track.get("images") or {}).get("coverarthq") or (
        track.get("images") or {}
    ).get("coverart")

    return Recognition(
        artist=artist, title=title, album=album, art_url=art, provider="shazamio"
    )
