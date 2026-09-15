"""Feed the delayed stream and its metadata into AzuraCast.

One job: at each boundary, POST /api/station/{id}/nowplaying/update.

The audio reaches AzuraCast on its own -- point the station at our stream as a
remote source. We used to push it as a DJ/streamer connection too, which made
AzuraCast treat the station as live and ignore what we said about it.

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
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass

import httpx

from . import appearance, db, labels, settings
from . import strings_es as S
from .config import Config
from .rockfm_api import STATION_ART
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


def metadata_for(
    row: sqlite3.Row | dict,
    language: str,
    public_url: str = "",
    appearance: dict[str, dict[str, str]] | None = None,
) -> Metadata:
    """Map a timeline row onto the two fields AzuraCast shows.

    Songs map naturally. Everything else borrows the pair: the Spanish label
    goes in the title, and the presenters (or the station name) in the artist,
    so an AzuraCast player shows "Publicidad" or "El Pirata y su banda" rather
    than a blank.
    """
    item = dict(row)
    primary, secondary, art = labels.present(item, language, appearance)
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
    """Talks to AzuraCast using whatever the settings currently say.

    Settings are read per call rather than captured at construction, so turning
    the integration on, off, or repointing it at another station takes effect
    without a restart -- which would otherwise cost a hole in the recording.
    """

    def __init__(self, config: Config, database: db.ThreadLocalDB, timeout: float = 15.0) -> None:
        self.config = config
        self._db = database
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @property
    def settings(self) -> dict:
        return settings.load(self._db.conn)

    @property
    def configured(self) -> bool:
        current = self.settings
        return bool(
            current["azuracast_enabled"]
            and current["azuracast_base_url"]
            and current["azuracast_station_id"]
            and current["azuracast_api_key"]
        )

    def _url(self, current: dict, path: str) -> str:
        return f"{current['azuracast_base_url'].rstrip('/')}{path}"

    def push(self, metadata: Metadata) -> bool:
        current = self.settings
        if not self.configured:
            return False
        path = f"/api/station/{current['azuracast_station_id']}/nowplaying/update"
        try:
            response = self.client.post(
                self._url(current, path),
                params=metadata.as_params(),
                headers={"X-API-Key": current["azuracast_api_key"]},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("metadata push failed: %s", exc)
            return False
        log.info("pushed: %s - %s", metadata.artist, metadata.title)
        return True

    def push_regardless(self, metadata: Metadata) -> bool:
        """Push even though the integration is now off.

        Used once, on the way out, so the station is not left displaying a song
        that stopped being true the moment we stopped updating it.
        """
        current = self.settings
        if not (current["azuracast_base_url"] and current["azuracast_station_id"]
                and current["azuracast_api_key"]):
            return False
        path = f"/api/station/{current['azuracast_station_id']}/nowplaying/update"
        try:
            response = self.client.post(
                self._url(current, path),
                params=metadata.as_params(),
                headers={"X-API-Key": current["azuracast_api_key"]},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("could not restore the station identity: %s", exc)
            return False
        return True

    def now_playing(self) -> dict | None:
        current = self.settings
        if not self.configured:
            return None
        path = f"/api/nowplaying/{current['azuracast_station_id']}"
        try:
            response = self.client.get(
                self._url(current, path), headers={"X-API-Key": current["azuracast_api_key"]}
            )
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


def station_metadata() -> Metadata:
    """What to leave on the station when we stop driving its metadata.

    Switching the integration off should not freeze AzuraCast on whatever song
    happened to be playing at the time. Hand it back the station's own identity
    and let it be.
    """
    return Metadata(title=S.STATION_NAME, artist="", art=STATION_ART)


class MetadataBridge:
    """Watches what is airing and pushes each change to AzuraCast."""

    def __init__(self, config: Config, database: db.ThreadLocalDB) -> None:
        self.config = config
        self._db = database
        self.client = MetadataClient(config, database)
        self.delay = DelayController(config)
        self.stop_event = threading.Event()
        self._last: tuple | None = None
        self._was_enabled = False

    def current(self) -> sqlite3.Row | None:
        position = playout_position(self.config, self.delay.current)
        return db.timeline_at(self._db.conn, position)

    def tick(self) -> bool:
        enabled = self.client.configured
        if not enabled:
            if self._was_enabled:
                # Turned off just now: hand the station back its own identity
                # rather than leaving it frozen on the last song we pushed.
                log.info("AzuraCast metadata push disabled; restoring the station identity")
                self._was_enabled = False
                self._last = None
                self.client.push_regardless(station_metadata())
            return False

        self._was_enabled = True
        row = self.current()
        if row is None:
            return False
        key = (row["kind"], row["title"], row["artist"], row["start_ms"])
        if key == self._last:
            return False
        current = settings.load(self._db.conn)
        metadata = metadata_for(
            row,
            current["display_language"],
            current["public_url"],
            appearance.load(self._db.conn, current["display_language"]),
        )
        if self.client.push(metadata):
            self._last = key
            return True
        return False

    def run(self) -> None:
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
    database = db.ThreadLocalDB(config.db_path)
    bridge = MetadataBridge(config, database)

    def handle(signum, _frame):
        log.info("signal %s received; shutting down", signum)
        bridge.stop_event.set()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    bridge.run()


if __name__ == "__main__":
    main()
