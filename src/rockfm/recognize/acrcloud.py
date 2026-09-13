"""ACRCloud backend -- official, purpose-built for broadcast monitoring.

Set RECOGNIZER=acrcloud plus ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY and
ACRCLOUD_ACCESS_SECRET.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time

import httpx
import numpy as np

from ..audio import pcm_to_wav
from .base import Recognition

log = logging.getLogger("rockfm.recognize.acrcloud")


class AcrCloudRecognizer:
    name = "acrcloud"

    def __init__(
        self,
        host: str | None = None,
        access_key: str | None = None,
        access_secret: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.host = host or os.environ.get("ACRCLOUD_HOST", "")
        self.access_key = access_key or os.environ.get("ACRCLOUD_ACCESS_KEY", "")
        self.access_secret = access_secret or os.environ.get("ACRCLOUD_ACCESS_SECRET", "")
        if not (self.host and self.access_key and self.access_secret):
            raise RuntimeError("ACRCLOUD_HOST/ACCESS_KEY/ACCESS_SECRET are not all set")
        self.client = httpx.Client(timeout=timeout)

    def _signature(self, timestamp: str) -> str:
        payload = "\n".join(
            ["POST", "/v1/identify", self.access_key, "audio", "1", timestamp]
        )
        digest = hmac.new(
            self.access_secret.encode(), payload.encode(), hashlib.sha1
        ).digest()
        return base64.b64encode(digest).decode()

    def recognize(self, samples: np.ndarray, rate: int) -> Recognition | None:
        if samples.size == 0:
            return None
        wav = pcm_to_wav(samples, rate)
        timestamp = str(int(time.time()))
        response = self.client.post(
            f"https://{self.host}/v1/identify",
            data={
                "access_key": self.access_key,
                "sample_bytes": str(len(wav)),
                "timestamp": timestamp,
                "signature": self._signature(timestamp),
                "data_type": "audio",
                "signature_version": "1",
            },
            files={"sample": ("probe.wav", wav, "audio/wav")},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status", {}).get("code") != 0:
            return None

        music = (payload.get("metadata") or {}).get("music") or []
        if not music:
            return None
        best = music[0]
        artists = ", ".join(a.get("name", "") for a in best.get("artists") or [])
        return Recognition(
            artist=artists.strip(),
            title=(best.get("title") or "").strip(),
            album=((best.get("album") or {}).get("name") or "").strip() or None,
            art_url=None,
            provider="acrcloud",
            confidence=float(best.get("score", 100)) / 100.0,
        )

    def close(self) -> None:
        self.client.close()
