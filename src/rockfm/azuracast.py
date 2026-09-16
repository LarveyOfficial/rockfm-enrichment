"""Feed the delayed stream and its metadata into AzuraCast.

One job: at each boundary, POST /api/station/{id}/nowplaying/update.

The audio reaches AzuraCast on its own -- point the station at our stream as a
remote source. We used to push it as a DJ/streamer connection too, which made
AzuraCast treat the station as live and ignore what we said about it.

That endpoint routes to Liquidsoap's `custom_metadata.insert`, so the station
must have a Liquidsoap backend -- a relay-only station has no backend adapter
and the call fails. The API key needs the station's "Broadcasting" permission.

On artwork: the `art` parameter itself never arrives. AzuraCast filters the
request's parameters against `AnnotateNextSong::ALLOWED_ANNOTATIONS` before
building the annotation, and that list holds `title`, `artist`, `duration`,
various ids and cue points -- no `art`, no `album`. Both are dropped without a
word, which is why the album field stays empty however we send it.

`media_id` is on that list, though, so artwork reaches AzuraCast a different
way: MediaCarrier keeps one never-played media record in the station library,
writes the current cover onto it, and we name it in the push. AzuraCast then
resolves the cover from that record -- its own library -- rather than from a
default or an external lookup. See docs/azuracast-artwork.md.
"""

from __future__ import annotations

import base64
import logging
import shutil
import signal
import sqlite3
import subprocess
import threading
from dataclasses import dataclass

import httpx

from . import appearance, db, labels, settings
from . import strings_es as S
from .config import Config
from .rockfm_api import STATION_ART
from .timeshift import DelayController, playout_position

log = logging.getLogger("rockfm.azuracast")

POLL_SECONDS = 2.0
# Where the artwork carrier lives in the station's library, and where its id is
# remembered so it is only ever created once.
CARRIER_PATH = "rockfm/nowplaying-artwork.mp3"
MEDIA_ID_KEY = "azuracast_media_id"
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

    def push(self, metadata: Metadata, media_id: int | str | None = None) -> bool:
        current = self.settings
        if not self.configured:
            return False
        path = f"/api/station/{current['azuracast_station_id']}/nowplaying/update"
        params = metadata.as_params()
        if media_id is not None:
            # The one way artwork reaches AzuraCast. `art` is filtered out of the
            # annotation before it is built; `media_id` is not, and a now-playing
            # row carrying one is a StationMedia, whose art we control.
            params["media_id"] = str(media_id)
        try:
            response = self.client.post(
                self._url(current, path),
                params=params,
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

    def close(self) -> None:
        self.client.close()


class MediaCarrier:
    """A media record in AzuraCast's library that carries our artwork.

    AzuraCast will not accept artwork for now-playing: `art` is dropped from the
    Liquidsoap annotation before it is built. What it will accept is `media_id`,
    and a now-playing row that has one is a StationMedia -- whose art comes from
    the library, which we can write to.

    So one record is kept, never played, purely as somewhere to put the picture.
    Each time the song changes its art, title and artist are rewritten and the
    metadata push points at it.
    """

    def __init__(self, config: Config, database: db.ThreadLocalDB, timeout: float = 30.0) -> None:
        self.config = config
        self._db = database
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._art_sent: str | None = None

    @property
    def settings(self) -> dict:
        return settings.load(self._db.conn)

    def close(self) -> None:
        self.client.close()

    def _headers(self, current: dict) -> dict[str, str]:
        return {"X-API-Key": current["azuracast_api_key"]}

    def _url(self, current: dict, path: str) -> str:
        return f"{current['azuracast_base_url'].rstrip('/')}{path}"

    def media_id(self) -> str | None:
        """The carrier record, creating it the first time it is needed."""
        stored = db.get_meta(self._db.conn, MEDIA_ID_KEY)
        if stored:
            return stored

        current = self.settings
        silence = _silent_mp3()
        if silence is None:
            return None
        station = current["azuracast_station_id"]
        try:
            response = self.client.post(
                self._url(current, f"/api/station/{station}/files"),
                json={"path": CARRIER_PATH, "file": base64.b64encode(silence).decode()},
                headers=self._headers(current),
            )
            response.raise_for_status()
            created = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("could not create the artwork carrier: %s", exc)
            return None

        found = str(created.get("id") or "")
        if not found:
            log.warning("artwork carrier was created without an id: %s", created)
            return None
        db.set_meta(self._db.conn, MEDIA_ID_KEY, found)
        log.info("created the artwork carrier as media %s", found)
        return found

    def artwork(self, metadata: Metadata) -> bytes | None:
        """The image itself, from disk where we have it and the network otherwise."""
        if not metadata.art:
            return None
        name = metadata.art.rsplit("/art/", 1)[-1]
        local = self.config.art_dir / name
        if "/art/" in metadata.art and local.is_file():
            return local.read_bytes()
        try:
            response = self.client.get(metadata.art)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            log.warning("could not read artwork %s: %s", metadata.art, exc)
            return None

    def publish(self, media_id: str, metadata: Metadata) -> bool:
        """Put this song's picture and name on the carrier record."""
        current = self.settings
        station = current["azuracast_station_id"]

        if metadata.art and metadata.art != self._art_sent:
            image = self.artwork(metadata)
            if image is not None:
                try:
                    response = self.client.post(
                        self._url(current, f"/api/station/{station}/art/{media_id}"),
                        files={"file": ("art.jpg", image, "image/jpeg")},
                        headers=self._headers(current),
                    )
                    response.raise_for_status()
                    self._art_sent = metadata.art
                except httpx.HTTPError as exc:
                    log.warning("artwork upload failed: %s", exc)

        # With a media_id present AzuraCast takes the song from the record, so
        # the record has to say what is playing.
        body = {"title": metadata.title, "artist": metadata.artist}
        if metadata.album:
            body["album"] = metadata.album
        try:
            response = self.client.put(
                self._url(current, f"/api/station/{station}/file/{media_id}"),
                json=body,
                headers=self._headers(current),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("could not describe the artwork carrier: %s", exc)
            return False
        return True


def _silent_mp3() -> bytes | None:
    """A second of silence, to hang artwork from. Never played."""
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg is needed once to create the artwork carrier")
        return None
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "anullsrc=r=44100:cl=stereo", "-t", "1",
             "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"],
            capture_output=True, timeout=30, check=True,
        )
        return done.stdout or None
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("could not render the carrier audio: %s", exc)
        return None


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
        self.carrier = MediaCarrier(config, database)
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
        # Artwork can only travel on a media record, so give the carrier this
        # song's picture and name before pointing now-playing at it. If any of
        # that fails the push still goes out; a song under the wrong picture
        # beats no song at all.
        media_id = self.carrier.media_id() if metadata.art else None
        if media_id and not self.carrier.publish(media_id, metadata):
            media_id = None

        if self.client.push(metadata, media_id=media_id):
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
        self.carrier.close()


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
