"""AudD backend -- official, supported, metered.

Set RECOGNIZER=audd and AUDD_API_TOKEN to use this instead of ShazamIO.
"""

from __future__ import annotations

import logging
import os

import httpx
import numpy as np

from ..audio import pcm_to_wav
from .base import Recognition

log = logging.getLogger("rockfm.recognize.audd")

ENDPOINT = "https://api.audd.io/"


class AudDRecognizer:
    name = "audd"

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        self.token = token or os.environ.get("AUDD_API_TOKEN", "")
        if not self.token:
            raise RuntimeError("AUDD_API_TOKEN is not set")
        self.client = httpx.Client(timeout=timeout)

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        if samples.size == 0:
            return None
        response = self.client.post(
            ENDPOINT,
            data={"api_token": self.token, "return": "apple_music,deezer"},
            files={"file": ("probe.wav", pcm_to_wav(samples, rate), "audio/wav")},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success" or not payload.get("result"):
            return None

        result = payload["result"]
        apple = result.get("apple_music") or {}
        artwork = (apple.get("artwork") or {}).get("url", "")
        art = artwork.replace("{w}", "600").replace("{h}", "600") if artwork else None
        return Recognition(
            artist=(result.get("artist") or "").strip(),
            title=(result.get("title") or "").strip(),
            album=(result.get("album") or "").strip() or None,
            art_url=art,
            provider="audd",
        )

    def close(self) -> None:
        self.client.close()
