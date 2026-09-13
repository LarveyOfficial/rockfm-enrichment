"""Feed the delayed stream and its metadata into AzuraCast.

Two independent jobs:

  audio     ffmpeg reads our local delayed HLS and connects to AzuraCast's
            DJ/streamer port as an Icecast source.
  metadata  at each boundary, POST /api/station/{id}/nowplaying/update.

That endpoint routes to Liquidsoap's `custom_metadata.insert`, so the station
must have a Liquidsoap backend -- a relay-only station has no backend adapter
and the call fails. The API key needs the station's "Broadcasting" permission.

On artwork: AzuraCast passes every parameter of that request through to a
Liquidsoap annotation, so we also send `art`. Whether it survives into
AzuraCast's now-playing feedback depends on the version, and it is harmless if
it does not -- AzuraCast then resolves art through its own Last.fm/MusicBrainz
services, and our own now-playing API carries the official RockFM artwork
regardless. `probe_art_support()` reports which of the two is happening.
"""

from __future__ import annotations

import logging
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass

import httpx

from . import db, labels
from . import strings_es as S
from .config import Config
from .timeshift import DelayController, playout_position

log = logging.getLogger("rockfm.azuracast")

POLL_SECONDS = 2.0
RESTART_DELAY = 5.0


@dataclass(frozen=True)
class Metadata:
    title: str
    artist: str
    album: str | None = None
    art: str | None = None

    def as_params(self) -> dict[str, str]:
        params = {"title": self.title, "artist": self.artist}
        if self.album:
            params["album"] = self.album
        if self.art:
            params["art"] = self.art
        return params


def metadata_for(row: sqlite3.Row | dict, language: str, public_url: str = "") -> Metadata:
    """Map a timeline row onto the two fields AzuraCast shows.

    Songs map naturally. Everything else borrows the pair: the Spanish label
    goes in the title, and the presenters (or the station name) in the artist,
    so an AzuraCast player shows "Publicidad" or "El Pirata y su banda" rather
    than a blank.
    """
    item = dict(row)
    primary, secondary = labels.render(item, language)
    art = item.get("art_url")
    if art and art.startswith("/") and public_url:
        art = public_url.rstrip("/") + art

    if item.get("kind") == S.KIND_CANCION:
        return Metadata(
            title=item.get("title") or primary,
            artist=item.get("artist") or "",
            album=item.get("album"),
            art=art,
        )
    return Metadata(title=primary, artist=secondary or S.STATION_NAME, art=art)


class MetadataClient:
    def __init__(self, config: Config, timeout: float = 15.0) -> None:
        self.config = config
        self.client = httpx.Client(
            timeout=timeout,
            headers={"X-API-Key": config.azuracast_api_key},
            follow_redirects=True,
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.config.azuracast_base_url
            and self.config.azuracast_station_id
            and self.config.azuracast_api_key
        )

    def _url(self, path: str) -> str:
        return f"{self.config.azuracast_base_url.rstrip('/')}{path}"

    def push(self, metadata: Metadata) -> bool:
        if not self.configured:
            return False
        path = f"/api/station/{self.config.azuracast_station_id}/nowplaying/update"
        try:
            response = self.client.post(self._url(path), params=metadata.as_params())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("metadata push failed: %s", exc)
            return False
        log.info("pushed: %s - %s", metadata.artist, metadata.title)
        return True

    def now_playing(self) -> dict | None:
        if not self.configured:
            return None
        path = f"/api/nowplaying/{self.config.azuracast_station_id}"
        try:
            response = self.client.get(self._url(path))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            log.warning("could not read AzuraCast now-playing: %s", exc)
            return None

    def probe_art_support(self, metadata: Metadata) -> bool | None:
        """Push metadata with art, then check whether AzuraCast kept our URL."""
        if not (metadata.art and self.push(metadata)):
            return None
        time.sleep(3)
        payload = self.now_playing()
        if not payload:
            return None
        art = ((payload.get("now_playing") or {}).get("song") or {}).get("art")
        kept = bool(art and metadata.art.split("/")[-1] in str(art))
        log.info(
            "AzuraCast %s our artwork (reported art: %s)",
            "kept" if kept else "replaced",
            art,
        )
        return kept

    def close(self) -> None:
        self.client.close()


class SourcePusher:
    """Keeps an ffmpeg process shipping our delayed HLS to AzuraCast."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self._process: subprocess.Popen | None = None

    @property
    def configured(self) -> bool:
        return bool(self.config.azuracast_dj_url and self.config.azuracast_dj_password)

    def _source_url(self) -> str:
        # icecast://source:password@host:port/mount
        target = self.config.azuracast_dj_url
        scheme, _, rest = target.partition("://")
        if not rest:
            rest, scheme = scheme, "icecast"
        return f"icecast://source:{self.config.azuracast_dj_password}@{rest}"

    def command(self) -> list[str]:
        local = f"http://127.0.0.1:{self.config.http_port}/hls/playlist.m3u8"
        codec = (self.config.azuracast_dj_codec or "mp3").lower()
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-re", "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10", "-i", local,
        ]
        if codec == "copy":
            args += ["-c:a", "copy", "-content_type", "audio/aac", "-f", "adts"]
        else:
            args += [
                "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-content_type", "audio/mpeg", "-f", "mp3",
            ]
        return args + [self._source_url()]

    def run(self) -> None:
        if not self.configured:
            log.info("no AzuraCast DJ target configured; not pushing audio")
            return
        while not self.stop_event.is_set():
            log.info("connecting to AzuraCast as a live source")
            try:
                self._process = subprocess.Popen(self.command())
                self._process.wait()
            except Exception as exc:
                log.warning("source push failed: %s", exc)
            if self.stop_event.is_set():
                break
            log.warning("source disconnected; retrying in %.0fs", RESTART_DELAY)
            self.stop_event.wait(RESTART_DELAY)

    def stop(self) -> None:
        self.stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()


class MetadataBridge:
    """Watches what is airing and pushes each change to AzuraCast."""

    def __init__(self, config: Config, database: db.ThreadLocalDB) -> None:
        self.config = config
        self._db = database
        self.client = MetadataClient(config)
        self.delay = DelayController(config)
        self.stop_event = threading.Event()
        self._last: tuple | None = None

    def current(self) -> sqlite3.Row | None:
        position = playout_position(self.config, self.delay.current)
        return db.timeline_at(self._db.conn, position)

    def tick(self) -> bool:
        row = self.current()
        if row is None:
            return False
        key = (row["kind"], row["title"], row["artist"], row["start_ms"])
        if key == self._last:
            return False
        metadata = metadata_for(row, self.config.display_language, self.config.public_url)
        if self.client.push(metadata):
            self._last = key
            return True
        return False

    def run(self) -> None:
        if not self.client.configured:
            log.info("no AzuraCast API target configured; not pushing metadata")
            return
        log.info("metadata bridge started")
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                log.warning("metadata tick failed: %s", exc)
            self.stop_event.wait(POLL_SECONDS)
        self.client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = Config()
    config.ensure_dirs()
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg not found on PATH")

    bridge = MetadataBridge(config, db.ThreadLocalDB(config.db_path))
    pusher = SourcePusher(config)

    def handle(signum, _frame):
        log.info("signal %s received; shutting down", signum)
        bridge.stop_event.set()
        pusher.stop()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    threading.Thread(target=bridge.run, daemon=True).start()
    pusher.run()
    while not bridge.stop_event.is_set():
        time.sleep(1)


if __name__ == "__main__":
    main()
