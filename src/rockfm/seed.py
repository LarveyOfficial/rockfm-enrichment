"""Bootstrap the fingerprint index from RockFM's own rotation catalog.

For each catalog entry we resolve a 30 s preview (iTunes, then Deezer) and
fingerprint it. That is enough to recognise much of the core rotation on day
one without a single external recognition call.

Previews only cover 30 s of each track, so a broadcast probe has to overlap that
window to match. The analyzer therefore probes at several offsets, and every
song the external recogniser identifies is re-fingerprinted from the real
broadcast audio -- that learned copy, not the preview, is what makes the index
good over time.

Safe to re-run: entries already present are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time

from . import db
from .audio import decode_url
from .config import Config
from .enrich import Enricher
from .fingerprint import FingerprintIndex, compute
from .rockfm_api import RockFmApi

log = logging.getLogger("rockfm.seed")

MIN_PREVIEW_SECONDS = 5.0


def seed(
    conn: sqlite3.Connection,
    config: Config,
    *,
    limit: int | None = None,
    pause: float = 0.3,
) -> dict[str, int]:
    api = RockFmApi()
    try:
        catalog = api.tracks()
    finally:
        api.close()

    enricher = Enricher(config.art_dir, catalog)
    index = FingerprintIndex(conn)
    stats = {"total": 0, "skipped": 0, "seeded": 0, "no_preview": 0, "failed": 0}

    entries = catalog[:limit] if limit else catalog
    for position, entry in enumerate(entries, start=1):
        stats["total"] += 1
        existing = conn.execute(
            "SELECT id FROM fp_tracks WHERE key = ?", (entry.key,)
        ).fetchone()
        if existing is not None:
            stats["skipped"] += 1
            continue

        try:
            enrichment = enricher.lookup(entry.artist, entry.title)
            art_path = enricher.cache_art(enrichment.art_url or entry.cover_url)
            db.upsert_song_meta(
                conn,
                {
                    "key": entry.key,
                    "title": entry.title,
                    "artist": entry.artist,
                    "album": enrichment.album,
                    "year": enrichment.year,
                    "duration_ms": enrichment.duration_ms,
                    "art_url": enrichment.art_url or entry.cover_url,
                    "art_path": art_path,
                    "preview_url": enrichment.preview_url,
                    "source": enrichment.source,
                },
                int(time.time() * 1000),
            )

            if not enrichment.preview_url:
                stats["no_preview"] += 1
                continue

            samples = decode_url(enrichment.preview_url)
            if samples.size < MIN_PREVIEW_SECONDS * 8000:
                stats["no_preview"] += 1
                continue

            index.add(
                kind="music",
                key=entry.key,
                hashes=compute(samples),
                title=entry.title,
                artist=entry.artist,
                source="preview",
                duration_ms=enrichment.duration_ms,
            )
            stats["seeded"] += 1
        except Exception as exc:
            stats["failed"] += 1
            log.warning("seed failed for %s - %s: %s", entry.artist, entry.title, exc)

        if position % 25 == 0:
            log.info(
                "seeded %d/%d (indexed=%d skipped=%d no-preview=%d failed=%d)",
                position, len(entries), stats["seeded"], stats["skipped"],
                stats["no_preview"], stats["failed"],
            )
        time.sleep(pause)

    enricher.close()
    log.info("seeding complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the fingerprint index from the RockFM catalog")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N entries")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between lookups")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = Config()
    config.ensure_dirs()
    seed(db.connect(config.db_path), config, limit=args.limit, pause=args.pause)


if __name__ == "__main__":
    main()
