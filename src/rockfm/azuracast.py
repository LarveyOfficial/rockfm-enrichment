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
import hashlib
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
from .buffer import BufferReader
from .config import Config
from .lag import LagProbe
from .rockfm_api import STATION_ART
from .timeshift import DelayController, playout_position

log = logging.getLogger("rockfm.azuracast")

POLL_SECONDS = 2.0
# Where the artwork carrier lives in the station's library, and where its id is
# remembered so it is only ever created once.
CARRIER_DIR = "rockfm/carriers"
# Keyed by what the record says, so a record is written once and never edited.
CARRIER_KEY_PREFIX = "azuracast_carrier:"
# Silence is rendered at the song's real length so AzuraCast can read the
# duration off the audio. Long programmes are capped -- hours of silence is not
# worth the upload, and their length still goes on the record.
MAX_RENDER_SECONDS = 600.0
RESTART_DELAY = 5.0


@dataclass(frozen=True)
class Metadata:
    title: str
    artist: str
    album: str | None = None
    art: str | None = None
    duration: float | None = None

    def as_params(self) -> dict[str, str]:
        params = {"title": self.title, "artist": self.artist}
        if self.album:
            params["album"] = self.album
        if self.art:
            params["art"] = self.art
        if self.duration:
            # Unlike `art` and `album`, `duration` is in ALLOWED_ANNOTATIONS, so
            # it survives the filter. Without it AzuraCast times the song by the
            # carrier media record instead -- a second of silence, shown as 0:01.
            params["duration"] = f"{self.duration:.3f}"
        return params


def _span_seconds(item: dict) -> float | None:
    """How long the timeline says this stretch runs, in seconds.

    Rows built by hand (and the tests' fixtures) carry no span, so this is
    allowed to come back empty rather than invent one.
    """
    start, end = item.get("start_ms"), item.get("end_ms")
    if start is None or end is None or end <= start:
        return None
    return (end - start) / 1000.0


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
            duration=_span_seconds(item),
        )
    return Metadata(
        title=primary,
        artist=secondary or S.STATION_NAME,
        art=art,
        duration=_span_seconds(item),
    )


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
        self._listen_url = ""

    @property
    def settings(self) -> dict:
        return settings.load(self._db.conn)

    def listen_url(self) -> str:
        """The mount AzuraCast broadcasts on, asked of AzuraCast itself.

        Nothing to configure and nothing to keep in step: the station already
        knows its own mount, so the lag probe has something to listen to the
        moment the station is up.
        """
        if self._listen_url:
            return self._listen_url
        if not self.configured:
            return ""
        current = self.settings
        path = f"/api/nowplaying/{current['azuracast_station_id']}"
        try:
            response = self.client.get(self._url(current, path))
            response.raise_for_status()
            found = response.json().get("station", {}).get("listen_url", "")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            log.debug("could not learn the station's mount: %s", exc)
            return ""
        if found:
            self._listen_url = found
            log.info("listening to %s to time the broadcast", found)
        return found

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


def carrier_key(metadata: Metadata) -> str:
    """Identifies a record by everything the record will say.

    Two airings that agree on name, album and length can share a record; any
    difference makes a new one. Keying on the content is what lets a record be
    written once and then left alone.
    """
    seconds = round(metadata.duration) if metadata.duration else 0
    raw = "\x00".join(
        (metadata.artist or "", metadata.title or "", metadata.album or "", str(seconds))
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class MediaCarrier:
    """One media record per distinct song, carrying its artwork and length.

    AzuraCast will not accept artwork for now-playing: `art` is dropped from the
    Liquidsoap annotation before it is built. What it will accept is `media_id`,
    and a now-playing row that has one is a StationMedia -- whose art and
    running time come from the library, which we can write to.

    There was one shared record for a while, rewritten at every boundary. That
    worked for artwork and failed at everything else: AzuraCast decides a song
    has changed partly by the record it points at, so a constant id left it
    announcing the same track forever, never resetting the elapsed time, while
    the length was edited underneath it at the very moment it was being read.

    So each distinct song gets its own record, written once on its first airing
    and reused untouched on every later one. A boundary becomes a single call
    that names an already-correct record.
    """

    def __init__(self, config: Config, database: db.ThreadLocalDB, timeout: float = 30.0) -> None:
        self.config = config
        self._db = database
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)
        # Set only when AzuraCast rejects a length outright, which a schema will
        # keep doing. A length that is quietly ignored is reported, not latched.
        self._length_refused = False
        self._length_checked = False

    @property
    def settings(self) -> dict:
        return settings.load(self._db.conn)

    def close(self) -> None:
        self.client.close()

    def _headers(self, current: dict) -> dict[str, str]:
        return {"X-API-Key": current["azuracast_api_key"]}

    def _url(self, current: dict, path: str) -> str:
        return f"{current['azuracast_base_url'].rstrip('/')}{path}"

    def record_for(self, metadata: Metadata) -> str | None:
        """The record for this song, created and filled in the first time.

        Returns None if it could not be made, which leaves the push to go out
        without one: the right song under the wrong picture beats no song.
        """
        key = carrier_key(metadata)
        remembered = db.get_meta(self._db.conn, CARRIER_KEY_PREFIX + key)
        if remembered:
            return remembered

        media_id = self._create(metadata, key)
        if media_id is None:
            return None
        # Only ever done here. Once a record exists it is never edited again.
        self._describe_new(media_id, metadata)
        self._send_artwork(media_id, metadata)
        db.set_meta(self._db.conn, CARRIER_KEY_PREFIX + key, media_id)
        log.info(
            "carrier %s created for %s - %s (%.0fs)",
            media_id, metadata.artist, metadata.title, metadata.duration or 0,
        )
        return media_id

    def _create(self, metadata: Metadata, key: str) -> str | None:
        current = self.settings
        silence = _silent_mp3(min(metadata.duration or 1.0, MAX_RENDER_SECONDS))
        if silence is None:
            return None
        station = current["azuracast_station_id"]
        try:
            response = self.client.post(
                self._url(current, f"/api/station/{station}/files"),
                json={
                    "path": f"{CARRIER_DIR}/{key}.mp3",
                    "file": base64.b64encode(silence).decode(),
                },
                headers=self._headers(current),
            )
            response.raise_for_status()
            created = response.json() or {}
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            log.warning("could not create a carrier record: %s", exc)
            return None
        found = str(created.get("id") or "")
        if not found:
            log.warning("carrier record was created without an id: %s", created)
            return None
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

    def _send_artwork(self, media_id: str, metadata: Metadata) -> None:
        image = self.artwork(metadata)
        if image is None:
            return
        current = self.settings
        station = current["azuracast_station_id"]
        try:
            response = self.client.post(
                self._url(current, f"/api/station/{station}/art/{media_id}"),
                files={"file": ("art.jpg", image, "image/jpeg")},
                headers=self._headers(current),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("artwork upload failed: %s", exc)

    def _describe_new(self, media_id: str, metadata: Metadata) -> bool:
        """Name the freshly created record, and say how long it runs."""
        current = self.settings
        station = current["azuracast_station_id"]
        body: dict = {"title": metadata.title, "artist": metadata.artist}
        if metadata.album:
            body["album"] = metadata.album
        # The rendered silence already runs this long, up to the cap. Saying so
        # as well covers the capped case and anything that re-reads the record.
        if metadata.duration and not self._length_refused:
            body["length"] = round(metadata.duration, 3)

        if not self._describe(current, station, media_id, body):
            if "length" not in body:
                return False
            self._length_refused = True
            log.info("AzuraCast refused a track length; carriers will read as their audio")
            body.pop("length")
            return self._describe(current, station, media_id, body)

        if "length" in body and not self._length_checked:
            self._length_checked = True
            self._report_length(current, station, media_id, body["length"])
        return True

    def _describe(self, current: dict, station: str, media_id: str, body: dict) -> bool:
        try:
            response = self.client.put(
                self._url(current, f"/api/station/{station}/file/{media_id}"),
                json=body,
                headers=self._headers(current),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("could not describe carrier %s: %s", media_id, exc)
            return False
        return True

    def _report_length(self, current: dict, station: str, media_id: str, asked: float) -> None:
        """Say once whether a length we set stuck. Diagnostic only, never a switch."""
        try:
            response = self.client.get(
                self._url(current, f"/api/station/{station}/file/{media_id}"),
                headers=self._headers(current),
            )
            response.raise_for_status()
            # A body that is not a JSON object is not worth crashing over.
            kept = float((response.json() or {}).get("length") or 0)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            log.debug("could not read the carrier back: %s", exc)
            return
        if abs(kept - asked) < 1.0:
            log.info("AzuraCast keeps the track length we set (%.0fs)", kept)
        else:
            log.warning(
                "AzuraCast read back %.0fs rather than the %.0fs we set; still sending it",
                kept, asked,
            )


def _silent_mp3(seconds: float = 1.0) -> bytes | None:
    """Silence of a given length, to hang a song's artwork and duration from.

    Never played. Rendered at the song's own length so AzuraCast can read the
    duration off the audio rather than only from the field we set -- a field it
    may recompute from the file later. Low bitrate mono keeps a four-minute
    track under half a megabyte.
    """
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg is needed to create a carrier record")
        return None
    length = max(1.0, min(float(seconds), MAX_RENDER_SECONDS))
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "anullsrc=r=16000:cl=mono", "-t", f"{length:.3f}",
             "-c:a", "libmp3lame", "-b:a", "16k", "-f", "mp3", "-"],
            capture_output=True, timeout=60, check=True,
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


class _BufferView:
    """Buffer access that works from whichever thread asks.

    `BufferReader` holds a connection, and SQLite connections cannot cross
    threads, so the reader is built per call against this thread's own.
    """

    def __init__(self, config: Config, database: db.ThreadLocalDB) -> None:
        self.config = config
        self._db = database

    def read(self, start_ms: int, span_ms: int, rate: int):
        return BufferReader(self._db.conn, self.config).read(start_ms, span_ms, rate=rate)


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
        self.lag = LagProbe(
            _BufferView(config, database),
            position=self.playout_position,
            listen_url=self.client.listen_url,
        )

    def playout_position(self) -> int:
        """The buffer instant going out of our stream this moment."""
        return playout_position(self.config, self.delay.current)

    def current(self) -> sqlite3.Row | None:
        # What AzuraCast is broadcasting now left us a while ago -- HLS,
        # Liquidsoap and Icecast each hold some of it. The push takes none of
        # that path, so without stepping back by the measured delay the title
        # would change well before the song does.
        position = self.playout_position() - self.lag.lag_ms
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
        # Artwork and running time can only travel on a media record, so name
        # the record for this song and let AzuraCast read both off it. Created
        # on a song's first airing and reused untouched afterwards, so a
        # boundary is one call against something already correct. If it cannot
        # be made the push still goes out: the right song under the wrong
        # picture beats no song at all.
        media_id = self.carrier.record_for(metadata)

        if self.client.push(metadata, media_id=media_id):
            self._last = key
            return True
        return False

    def run(self) -> None:
        log.info("metadata bridge started")
        timer = threading.Thread(target=self.lag.run, name="lag-probe", daemon=True)
        timer.start()
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                log.warning("metadata tick failed: %s", exc)
            self.stop_event.wait(POLL_SECONDS)
        self.lag.stop()
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
