"""Continuously record the upstream HLS stream to disk.

The upstream live window is only ~4 segments (~24 s), so a stall of half a
minute loses audio permanently. The loop therefore polls fast, retries hard, and
records any gap it could not avoid so playout can emit a discontinuity instead
of silently skipping time.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading
import time
from datetime import UTC, datetime

import httpx

from . import db, hlsutil
from .config import Config

log = logging.getLogger("rockfm.ingest")

GAP_TOLERANCE_MS = 1500
MASTER_REFRESH_SECONDS = 300


def segment_relpath(pdt_ms: int) -> str:
    """Shard segments by UTC day/hour so directories stay small and prune cheaply."""
    moment = datetime.fromtimestamp(pdt_ms / 1000, tz=UTC)
    return f"{moment:%Y%m%d}/{moment:%H}/{pdt_ms}.aac"


class Ingestor:
    def __init__(self, config: Config, database: db.ThreadLocalDB) -> None:
        self.config = config
        self._db = database
        self.client = httpx.Client(
            headers=config.http_headers,
            timeout=config.http_timeout,
            follow_redirects=True,
        )
        self._media_url: str | None = None
        self._media_url_at = 0.0
        self._last_prune = 0.0
        self.stop_event = threading.Event()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.conn

    # --- playlist resolution ---

    def media_url(self) -> str:
        now = time.monotonic()
        if self._media_url and now - self._media_url_at < MASTER_REFRESH_SECONDS:
            return self._media_url
        response = self.client.get(self.config.source_url)
        response.raise_for_status()
        text = response.text
        if hlsutil.is_master_playlist(text):
            variants = hlsutil.parse_master(text, str(response.url))
            if not variants:
                raise RuntimeError("master playlist contained no variants")
            self._media_url = variants[0]
        else:
            self._media_url = str(response.url)
        self._media_url_at = now
        log.info("resolved media playlist: %s", self._media_url)
        return self._media_url

    # --- one poll cycle ---

    def poll_once(self) -> int:
        playlist = hlsutil.parse_media(
            self._fetch_text(self.media_url()), self.media_url()
        )
        stored = 0
        for segment in playlist.segments:
            if self.stop_event.is_set():
                break
            if segment.pdt_ms is None:
                log.warning("segment %s has no PROGRAM-DATE-TIME; skipping", segment.uri)
                continue
            if db.has_segment(self.conn, segment.pdt_ms):
                continue
            if self._store(segment):
                stored += 1
        return stored

    def _fetch_text(self, url: str) -> str:
        response = self.client.get(url)
        response.raise_for_status()
        return response.text

    def _store(self, segment: hlsutil.MediaSegment) -> bool:
        assert segment.pdt_ms is not None
        try:
            response = self.client.get(segment.uri)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("failed to fetch %s: %s", segment.uri, exc)
            return False

        data = response.content
        if not data:
            log.warning("empty segment body for %s", segment.uri)
            return False

        relpath = segment_relpath(segment.pdt_ms)
        target = self.config.segments_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)

        self._note_gap(segment.pdt_ms)
        db.insert_segment(
            self.conn,
            pdt_ms=segment.pdt_ms,
            seq=segment.seq,
            duration_ms=segment.duration_ms,
            relpath=relpath,
            size=len(data),
            fetched_ms=int(time.time() * 1000),
        )
        return True

    def _note_gap(self, pdt_ms: int) -> None:
        latest = db.latest_segment(self.conn)
        if latest is None:
            return
        expected = latest["pdt_ms"] + latest["duration_ms"]
        missing = pdt_ms - expected
        if missing > GAP_TOLERANCE_MS:
            log.warning(
                "gap of %.1fs between %s and %s", missing / 1000, expected, pdt_ms
            )
            db.record_gap(
                self.conn,
                after_pdt_ms=latest["pdt_ms"],
                before_pdt_ms=pdt_ms,
                missing_ms=missing,
                reason="missed_segments",
                detected_ms=int(time.time() * 1000),
            )

    # --- retention ---

    def prune(self) -> int:
        cutoff = int(time.time() * 1000) - self.config.buffer_hours * 3600 * 1000
        relpaths = db.delete_segments_before(self.conn, cutoff)
        for relpath in relpaths:
            (self.config.segments_dir / relpath).unlink(missing_ok=True)
        # Timeline rows describe audio that no longer exists, so they go with it.
        # Fingerprints deliberately do not: the whole point of learning a song is
        # that it stays known after the recording is gone.
        db.delete_timeline_before(self.conn, cutoff)
        self._remove_empty_dirs()
        if relpaths:
            log.info("pruned %d segments older than %d h", len(relpaths), self.config.buffer_hours)
        return len(relpaths)

    def _remove_empty_dirs(self) -> None:
        root = self.config.segments_dir
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    # --- main loop ---

    def run(self) -> None:
        self.config.ensure_dirs()
        log.info("ingest starting: %s", self.config.source_url)
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                stored = self.poll_once()
                backoff = 1.0
                if stored:
                    log.debug("stored %d segment(s)", stored)
            except Exception as exc:  # keep the recorder alive no matter what
                log.warning("poll failed: %s", exc)
                self._media_url = None
                self.stop_event.wait(min(backoff, 15.0))
                backoff = min(backoff * 2, 15.0)
                continue

            now = time.monotonic()
            if now - self._last_prune > 300:
                self._last_prune = now
                try:
                    self.prune()
                except Exception as exc:
                    log.warning("prune failed: %s", exc)

            self.stop_event.wait(self.config.poll_interval)
        self.client.close()
        log.info("ingest stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config = Config()
    config.ensure_dirs()
    ingestor = Ingestor(config, db.ThreadLocalDB(config.db_path))

    def handle(signum, _frame):
        log.info("signal %s received; shutting down", signum)
        ingestor.stop_event.set()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    ingestor.run()


if __name__ == "__main__":
    main()
