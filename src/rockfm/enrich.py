"""Fill in album, year, artwork and preview audio for an identified song.

Artwork order is deliberate: RockFM's own catalog art first (it is the station's
own curated cover for its rotation), then iTunes, then Deezer. Both external
APIs are free and need no key.

Note that a song's *broadcast* duration comes from the analyzer's boundaries,
not from here -- radio edits and crossfades differ from the release.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from .rockfm_api import CatalogEntry, catalog_key

log = logging.getLogger("rockfm.enrich")

ITUNES_SEARCH = "https://itunes.apple.com/search"
DEEZER_SEARCH = "https://api.deezer.com/search"
MIN_SIMILARITY = 0.62
MIN_ARTIST_SIMILARITY = 0.72
MIN_TITLE_SIMILARITY = 0.70
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)

# Search APIs happily return tribute, karaoke and lullaby records whose artist
# and title match perfectly. They are never what a rock station just played.
JUNK_MARKERS = (
    "lullaby", "rendition", "tribute", "karaoke", "made famous", "made popular",
    "cover version", "covers of", "instrumental version", "string quartet",
    "8-bit", "8 bit", "workout mix", "in the style of", "originally performed",
)


@dataclass(frozen=True)
class Enrichment:
    title: str
    artist: str
    album: str | None = None
    year: int | None = None
    duration_ms: int | None = None
    art_url: str | None = None
    preview_url: str | None = None
    source: str = "none"


def normalise(text: str | None) -> str:
    cleaned = _PUNCT.sub(" ", (text or "").casefold())
    return " ".join(cleaned.split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def _is_junk(*fields: str | None) -> bool:
    haystack = " ".join(normalise(field) for field in fields)
    return any(marker in haystack for marker in JUNK_MARKERS)


@dataclass(frozen=True)
class _Candidate:
    score: float
    junk: bool
    year: int | None
    payload: dict


def _rank(
    artist: str,
    title: str,
    items: list[dict],
    read: "Callable[[dict], tuple[str, str, str | None, int | None]]",
) -> dict | None:
    """Pick the best release for artist/title.

    Artist and title must each clear their own threshold -- a great title match
    with the wrong artist is a different recording. Among survivors, real
    releases beat tribute/karaoke records, and the earliest release wins so we
    land on the original album rather than a later compilation.
    """
    candidates: list[_Candidate] = []
    for item in items:
        item_artist, item_title, album, year = read(item)
        artist_similarity = similarity(artist, item_artist)
        title_similarity = similarity(title, item_title)
        if artist_similarity < MIN_ARTIST_SIMILARITY or title_similarity < MIN_TITLE_SIMILARITY:
            continue
        score = 0.5 * artist_similarity + 0.5 * title_similarity
        if score < MIN_SIMILARITY:
            continue
        candidates.append(
            _Candidate(
                score=score,
                junk=_is_junk(item_artist, item_title, album),
                year=year,
                payload=item,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.junk, -round(c.score, 3), c.year or 9999))
    return candidates[0].payload


class Enricher:
    def __init__(
        self,
        art_dir: Path,
        catalog: list[CatalogEntry] | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.art_dir = art_dir
        self.art_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: dict[str, CatalogEntry] = {
            entry.key: entry for entry in (catalog or [])
        }
        self.client = httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": "rockfm-enrichment/0.1"}
        )
        self._cache: dict[str, Enrichment] = {}

    def close(self) -> None:
        self.client.close()

    # --- providers ---

    def _itunes(self, artist: str, title: str) -> Enrichment | None:
        try:
            response = self.client.get(
                ITUNES_SEARCH,
                params={"term": f"{artist} {title}", "entity": "song", "limit": 25},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:
            log.debug("iTunes lookup failed for %s - %s: %s", artist, title, exc)
            return None

        best = _rank(
            artist,
            title,
            results,
            lambda item: (
                item.get("artistName", ""),
                item.get("trackName", ""),
                item.get("collectionName"),
                int(item["releaseDate"][:4]) if item.get("releaseDate") else None,
            ),
        )
        if best is None:
            return None

        art = best.get("artworkUrl100") or ""
        return Enrichment(
            title=best.get("trackName") or title,
            artist=best.get("artistName") or artist,
            album=best.get("collectionName"),
            year=int(best["releaseDate"][:4]) if best.get("releaseDate") else None,
            duration_ms=best.get("trackTimeMillis"),
            art_url=art.replace("100x100", "600x600") or None,
            preview_url=best.get("previewUrl"),
            source="itunes",
        )

    def _deezer(self, artist: str, title: str) -> Enrichment | None:
        try:
            response = self.client.get(
                DEEZER_SEARCH, params={"q": f"{artist} {title}", "limit": 25}
            )
            response.raise_for_status()
            results = response.json().get("data", [])
        except Exception as exc:
            log.debug("Deezer lookup failed for %s - %s: %s", artist, title, exc)
            return None

        best = _rank(
            artist,
            title,
            results,
            lambda item: (
                item.get("artist", {}).get("name", ""),
                item.get("title", ""),
                item.get("album", {}).get("title"),
                None,
            ),
        )
        if best is None:
            return None

        album = best.get("album", {})
        return Enrichment(
            title=best.get("title") or title,
            artist=best.get("artist", {}).get("name") or artist,
            album=album.get("title"),
            year=None,
            duration_ms=int(best["duration"]) * 1000 if best.get("duration") else None,
            art_url=album.get("cover_xl") or album.get("cover_big"),
            preview_url=best.get("preview"),
            source="deezer",
        )

    # --- public ---

    def lookup(self, artist: str, title: str) -> Enrichment:
        key = catalog_key(artist, title)
        if key in self._cache:
            return self._cache[key]

        result = self._itunes(artist, title) or self._deezer(artist, title)
        if result is None:
            result = Enrichment(title=title, artist=artist, source="none")

        # The station's own cover wins when it has one for this track.
        entry = self.catalog.get(key)
        if entry and entry.cover_url:
            result = Enrichment(**{**result.__dict__, "art_url": entry.cover_url,
                                   "source": f"{result.source}+catalog"})

        self._cache[key] = result
        return result

    def cache_art(self, url: str | None) -> str | None:
        """Download artwork once; return the path playout serves it under."""
        if not url:
            return None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        target = self.art_dir / f"{digest}.jpg"
        if not target.exists():
            try:
                response = self.client.get(url)
                response.raise_for_status()
                target.write_bytes(response.content)
            except Exception as exc:
                log.debug("art download failed for %s: %s", url, exc)
                return None
        return f"/art/{digest}.jpg"
