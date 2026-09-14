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

Every lookup is bounded by a hard wall-clock timeout. ShazamIO's stock client is
built with `ExponentialRetry(attempts=20, max_timeout=60)` and no
`ClientTimeout`, so a single unanswered lookup can block for well over an hour
-- and because it blocks rather than raising, no amount of error handling
upstream will notice. That is not hypothetical: it stalled a live deployment at
roughly fifteen minutes per probe, which over seven hours identified three songs,
all of them from the local index and none from Shazam.
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

# A lookup that has not answered in this long is not going to. The analyser has
# hours of buffer in hand, so giving up early and falling back to the local
# index costs far less than waiting.
TIMEOUT_SECONDS = 25.0

# Two attempts, briefly spaced. Rate limiting (429) is deliberately *not* in the
# retry set: the caller's own throttle backs off on failure, and retrying a
# refusal twenty times in a row is what earns a longer refusal.
RETRY_ATTEMPTS = 2
RETRY_MAX_BACKOFF = 4.0
RETRY_STATUSES = {500, 502, 503, 504}


class ShazamRecognizer:
    name = "shazamio"

    def __init__(self, timeout: float = TIMEOUT_SECONDS) -> None:
        from shazamio import SearchParams, Shazam  # lazy so the dep stays optional

        self.timeout = timeout
        self._shazam = Shazam(http_client=_bounded_client())
        self._options = SearchParams(segment_duration_seconds=SEGMENT_SECONDS)

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        if samples.size == 0:
            return None
        payload = asyncio.run(self._recognize(pcm_to_wav(samples, rate)))
        return _parse(payload)

    async def _recognize(self, wav: bytes) -> dict[str, Any]:
        """Ask Shazam, or give up. Failure raises so the throttle backs off.

        Returning None on a transport failure would be indistinguishable from
        "Shazam answered, and does not know this track" -- which resets the
        backoff and sends us straight back into the same dead call.
        """
        try:
            return await asyncio.wait_for(
                self._shazam.recognize(wav, options=self._options), self.timeout
            )
        except TimeoutError:
            raise TimeoutError(f"no answer within {self.timeout:.0f}s") from None

    def close(self) -> None:
        return None


def _bounded_client() -> Any:
    """ShazamIO's HTTP client, with a retry policy that cannot run away.

    Built defensively: this reaches into ShazamIO's internals, and the library
    is a reverse-engineered client that may rearrange them. If that happens we
    fall back to the stock client, where `recognize` still holds the wall-clock
    timeout that actually bounds the call.
    """
    try:
        from aiohttp_retry import ExponentialRetry
        from shazamio.client import HTTPClient

        return HTTPClient(
            retry_options=ExponentialRetry(
                attempts=RETRY_ATTEMPTS,
                max_timeout=RETRY_MAX_BACKOFF,
                statuses=RETRY_STATUSES,
            )
        )
    except Exception as exc:  # pragma: no cover - depends on ShazamIO internals
        log.warning("using ShazamIO's stock HTTP client: %s", exc)
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
