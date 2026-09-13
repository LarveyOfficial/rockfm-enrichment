"""Clients for RockFM's own (undocumented) player endpoints.

  /ply/emds   station list, including the national HLS URL
  /ply/tracks rotation catalog: [TITLE, ARTIST, cover path] -- official artwork
  /ply/prg    weekly schedule: programmes, presenters, artwork

The catalog is *not* exhaustive, so it seeds the fingerprint index and supplies
artwork; it is never treated as a closed set of what the station can play.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import httpx

log = logging.getLogger("rockfm.api")

BASE_URL = "https://www.rockfm.fm"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": f"{BASE_URL}/"}


@dataclass(frozen=True)
class CatalogEntry:
    title: str
    artist: str
    cover_url: str | None

    @property
    def key(self) -> str:
        return catalog_key(self.artist, self.title)


@dataclass(frozen=True)
class Programme:
    title: str
    lead: str
    horario: str
    image_url: str | None
    description: str
    start_minute: int
    end_minute: int
    category_id: int | None


def catalog_key(artist: str | None, title: str | None) -> str:
    """Normalised key used to join catalog, recogniser output and fingerprints."""
    return f"{(artist or '').strip().casefold()}|{(title or '').strip().casefold()}"


def absolute(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith(("http://", "https://")):
        return path
    return f"{BASE_URL}{path if path.startswith('/') else '/' + path}"


class RockFmApi:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(headers=HEADERS, timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> RockFmApi:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _get_json(self, path: str):
        # The player appends a random query string; these endpoints are cached hard.
        response = self.client.get(f"{self.base_url}{path}?{random.random()}")
        response.raise_for_status()
        return response.json()

    def tracks(self) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        for row in self._get_json("/ply/tracks"):
            if not isinstance(row, list) or len(row) < 2:
                continue
            title, artist = (row[0] or "").strip(), (row[1] or "").strip()
            if not title or not artist:
                continue
            cover = absolute(row[2] if len(row) > 2 else None)
            entries.append(CatalogEntry(title=title, artist=artist, cover_url=cover))
        log.info("catalog: %d entries", len(entries))
        return entries

    def schedule(self) -> dict[int, list[Programme]]:
        """Programmes per Python weekday (Monday=0 .. Sunday=6)."""
        payload = self._get_json("/ply/prg")
        days: dict[int, list[Programme]] = {}
        for weekday in range(7):
            raw = payload.get("prg", {}).get(f"d{weekday}", {}).get("es", [])
            days[weekday] = [_programme(entry) for entry in raw]
        return days

    def fallback_programme(self) -> Programme | None:
        payload = self._get_json("/ply/prg")
        entry = payload.get("cfg", {}).get("failBackEmision")
        return _programme(entry) if entry else None

    def stream_url(self) -> str | None:
        payload = self._get_json("/ply/prg")
        track = payload.get("cfg", {}).get("failBackEmision", {}).get("track", {})
        return track.get("track")


def _programme(entry: dict) -> Programme:
    return Programme(
        title=(entry.get("title") or "").strip(),
        lead=(entry.get("lead") or "").strip(),
        horario=(entry.get("horario") or "").strip(),
        image_url=absolute(entry.get("image")),
        description=(entry.get("description") or "").strip(),
        start_minute=int(entry.get("from") or 0),
        end_minute=int(entry.get("to") or 1440),
        category_id=entry.get("categoryId"),
    )
