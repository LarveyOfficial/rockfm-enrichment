"""Delayed HLS playout and the rich now-playing API.

Segments are served back as the bytes we recorded, with only their ID3 TIT2
frame rewritten to the label our analyzer worked out -- the audio itself is
never re-encoded. Playlist timestamps are shifted forward by the current delay
so the delayed stream presents as live.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import appearance, db, hlsutil, labels, settings
from . import strings_es as S
from .config import Config
from .dashboard import DASHBOARD_HTML
from .timeshift import DelayController, playout_position, source_wallclock

log = logging.getLogger("rockfm.playout")

SEGMENT_PREFIX = "s"
M3U8_TYPE = "application/vnd.apple.mpegurl"
AAC_TYPE = "audio/aac"
GAP_TOLERANCE_MS = 1500


class Playout:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._db = db.ThreadLocalDB(config.db_path)
        self.delay = DelayController(config)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.conn

    # --- helpers ---

    def position_ms(self, now: datetime | None = None) -> int:
        return playout_position(self.config, self.delay.current, now)

    def current_item(self, now: datetime | None = None) -> sqlite3.Row | None:
        return db.timeline_at(self.conn, self.position_ms(now))

    def is_safe_boundary(self) -> bool:
        """True when we are not in the middle of a song, so a delay jump is tolerable."""
        item = self.current_item()
        return item is None or item["kind"] != S.KIND_CANCION

    def window(self) -> list[sqlite3.Row]:
        """Segments to serve right now, or nothing if the buffer has a hole here.

        Asking for the newest segments at or before the playout position would,
        during a hole, hand back whatever was recorded before it -- re-stamped
        to look current, and re-served until the hole ends. A restart would
        therefore replay its last few seconds on a loop rather than admit the
        audio is missing. Silence is the honest answer.
        """
        position = self.position_ms()
        rows = db.segments_upto(self.conn, position, self.config.playlist_segments)
        if rows:
            newest = rows[-1]
            behind = position - (newest["pdt_ms"] + newest["duration_ms"])
            if behind > GAP_TOLERANCE_MS:
                return []
        return rows

    # --- playlists ---

    def media_playlist(self) -> str:
        rows = self.window()
        if not rows:
            state, _remaining = self.buffer_state()
            detail = {
                "gap": "nothing was recorded for this moment",
                "filling": "buffer is still filling",
            }.get(state, "no audio available")
            raise HTTPException(status_code=503, detail=detail)

        shift = timedelta(seconds=self.delay.current)
        out: list[hlsutil.OutSegment] = []
        previous: sqlite3.Row | None = None
        for row in rows:
            discontinuity = False
            if previous is not None:
                expected = previous["pdt_ms"] + previous["duration_ms"]
                if row["pdt_ms"] - expected > GAP_TOLERANCE_MS:
                    discontinuity = True
            pdt = datetime.fromtimestamp(row["pdt_ms"] / 1000, tz=UTC) + shift
            out.append(
                hlsutil.OutSegment(
                    uri=f"{SEGMENT_PREFIX}{row['pdt_ms']}.aac",
                    duration=row["duration_ms"] / 1000,
                    pdt=pdt,
                    discontinuity=discontinuity,
                )
            )
            previous = row

        target = max(1, round(max(item.duration for item in out)))
        return hlsutil.build_media_playlist(out, rows[0]["seq"], target)

    def segment_bytes(self, pdt_ms: int) -> bytes:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE pdt_ms = ?", (pdt_ms,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="segment not in buffer")
        path = self.config.segments_dir / row["relpath"]
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="segment file missing") from None

        item = db.timeline_at(self.conn, pdt_ms)
        label = labels.stream_title(
            dict(item) if item else {}, self.language, self.appearance
        )
        try:
            return hlsutil.rewrite_tit2(data, label)
        except Exception as exc:  # never fail playout over a metadata rewrite
            log.warning("TIT2 rewrite failed for %s: %s", pdt_ms, exc)
            return data

    # --- now playing ---

    @property
    def settings(self) -> dict:
        return settings.load(self.conn)

    @property
    def language(self) -> str:
        return self.settings["display_language"]

    @property
    def appearance(self) -> dict[str, dict[str, str]]:
        return appearance.load(self.conn, self.language)

    def _present(self, item: sqlite3.Row | None) -> tuple[str, str, str | None]:
        if item is None:
            return S.STATION_NAME, "", None
        return labels.present(dict(item), self.language, self.appearance)

    def _render(self, item: sqlite3.Row | None) -> tuple[str, str]:
        primary, secondary, _ = self._present(item)
        return primary, secondary

    def _describe(self, item: sqlite3.Row | None, position_ms: int) -> dict:
        primary, secondary, art = self._present(item)
        if item is None:
            return {
                "kind": S.KIND_DESCONOCIDO,
                "primary": primary,
                "secondary": secondary,
                "art": None,
                "duration": None,
                "elapsed": None,
                "remaining": None,
            }
        duration_ms = item["end_ms"] - item["start_ms"]
        elapsed_ms = max(0, min(position_ms - item["start_ms"], duration_ms))
        return {
            "kind": item["kind"],
            "primary": primary,
            "secondary": secondary,
            "title": item["title"],
            "artist": item["artist"],
            "album": item["album"],
            "year": item["year"],
            "art": art,
            "show": item["show_title"],
            "presenters": item["show_lead"],
            "confidence": item["confidence"],
            "source": item["source"],
            "started_at": _iso(item["start_ms"]),
            "ends_at": _iso(item["end_ms"]),
            "duration": round(duration_ms / 1000, 3),
            "elapsed": round(elapsed_ms / 1000, 3),
            "remaining": round((duration_ms - elapsed_ms) / 1000, 3),
        }

    def now_playing(self) -> dict:
        now = datetime.now(UTC)
        position = self.position_ms(now)
        current = db.timeline_at(self.conn, position)
        upcoming = db.timeline_after(self.conn, position)
        earliest = db.earliest_segment(self.conn)
        latest = db.latest_segment(self.conn)
        state, remaining = self.buffer_state()
        return {
            "station": {"name": S.STATION_NAME, "language": self.language},
            "delay_seconds": self.delay.current,
            "delay_hours": round(self.delay.current / 3600, 2),
            "pending_delay_seconds": self.delay.pending.delay if self.delay.pending else None,
            "source_time": source_wallclock(self.config, now).isoformat(),
            "source_timezone": self.config.source_tz_name,
            "playout_time": _iso(position),
            "now_playing": self._describe(current, position),
            "next": self._describe(upcoming, position) if upcoming else None,
            "buffer": {
                "state": state,
                "seconds_until_ready": remaining,
                "segments": db.segment_count(self.conn),
                "earliest": _iso(earliest["pdt_ms"]) if earliest else None,
                "latest": _iso(latest["pdt_ms"]) if latest else None,
                "seconds": (
                    (latest["pdt_ms"] - earliest["pdt_ms"]) / 1000
                    if earliest and latest
                    else 0
                ),
            },
        }

    def buffer_state(self) -> tuple[str, float | None]:
        """Where we are between a cold start and being able to play.

        A fresh install has nothing to play until the buffer spans the whole
        delay -- six hours, by default. That is the design working, not a
        fault, so it gets its own state rather than looking like a failure.
        """
        position = self.position_ms()
        covering = db.segment_at(self.conn, position)
        if covering is not None:
            behind = position - (covering["pdt_ms"] + covering["duration_ms"])
            if behind <= GAP_TOLERANCE_MS:
                return "ready", 0.0
            # Recorded either side but not here: a restart, or an outage.
            following = db.segments_from(self.conn, position, 1)
            return "gap", (following[0]["pdt_ms"] - position) / 1000 if following else None
        earliest = db.earliest_segment(self.conn)
        if earliest is None:
            return "empty", None
        if position < earliest["pdt_ms"]:
            return "filling", (earliest["pdt_ms"] - position) / 1000
        return "stalled", None

    # --- dashboard support ---

    def timeline(self, start_ms: int, end_ms: int) -> list[dict]:
        """Everything the analyzer worked out over a span."""
        items = []
        look = self.appearance
        language = self.language
        for row in db.timeline_between(self.conn, start_ms, end_ms):
            primary, secondary, art = labels.present(dict(row), language, look)
            items.append(
                {
                    "start": row["start_ms"],
                    "end": row["end_ms"],
                    "duration": (row["end_ms"] - row["start_ms"]) / 1000,
                    "kind": row["kind"],
                    "primary": primary,
                    "secondary": secondary,
                    "title": row["title"],
                    "artist": row["artist"],
                    "album": row["album"],
                    "year": row["year"],
                    "art": art,
                    "show": row["show_title"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                }
            )
        return items

    def status(self) -> dict:
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        earliest = db.earliest_segment(self.conn)
        latest = db.latest_segment(self.conn)
        state, remaining = self.buffer_state()
        cursor_raw = db.get_meta(self.conn, db.ANALYZER_CURSOR_KEY)
        cursor = int(cursor_raw) if cursor_raw else None
        heartbeat_raw = db.get_meta(self.conn, db.ANALYZER_HEARTBEAT_KEY)
        heartbeat = int(heartbeat_raw) if heartbeat_raw else None

        counts = {
            "segments": db.segment_count(self.conn),
            "items": self.conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0],
            "songs": self.conn.execute(
                "SELECT COUNT(*) FROM timeline WHERE kind = ?", (S.KIND_CANCION,)
            ).fetchone()[0],
        }

        recognizer_raw = db.get_meta(self.conn, db.RECOGNIZER_STATE_KEY)
        try:
            recognizer = json.loads(recognizer_raw) if recognizer_raw else None
        except ValueError:
            recognizer = None

        return {
            "now": now_ms,
            "delay_seconds": self.delay.current,
            "recognizer": recognizer,
            "source_timezone": self.config.source_tz_name,
            "source_time": source_wallclock(self.config, datetime.now(UTC)).isoformat(),
            "buffer": {
                "state": state,
                "seconds_until_ready": remaining,
                "earliest": earliest["pdt_ms"] if earliest else None,
                "latest": latest["pdt_ms"] if latest else None,
                "seconds": (
                    (latest["pdt_ms"] - earliest["pdt_ms"]) / 1000 if earliest and latest else 0
                ),
                "ingest_lag_seconds": (now_ms - latest["pdt_ms"]) / 1000 if latest else None,
            },
            "analyzer": {
                "cursor": cursor,
                "state": db.get_meta(self.conn, db.ANALYZER_STATE_KEY),
                "progress": db.get_meta(self.conn, db.ANALYZER_PROGRESS_KEY) or None,
                "last_active_seconds": (now_ms - heartbeat) / 1000 if heartbeat else None,
                # How far the analyzer still is from the newest recorded audio.
                # In steady state this stays small; a growing number means it is
                # falling behind the recorder.
                "behind_live_seconds": (
                    (latest["pdt_ms"] - cursor) / 1000 if cursor and latest else None
                ),
            },
            "playout": {"position": self.position_ms()},
            "gaps": [
                {"after": row["after_pdt_ms"], "before": row["before_pdt_ms"],
                 "missing_seconds": row["missing_ms"] / 1000}
                for row in db.gaps_between(
                    self.conn, earliest["pdt_ms"] if earliest else 0, now_ms
                )
            ],
            "counts": counts,
        }

    def replay_playlist(self, start_ms: int, duration_s: float) -> str:
        """A complete playlist for one stretch of the buffer.

        Marked VOD so a player can seek through it: this is for auditing what
        the analyzer decided, not for following the delayed live edge.
        """
        rows = db.segments_between(self.conn, start_ms, start_ms + int(duration_s * 1000))
        if not rows:
            raise HTTPException(status_code=404, detail="nothing recorded for that span")

        out: list[hlsutil.OutSegment] = []
        previous: sqlite3.Row | None = None
        for row in rows:
            discontinuity = previous is not None and (
                row["pdt_ms"] - (previous["pdt_ms"] + previous["duration_ms"]) > GAP_TOLERANCE_MS
            )
            out.append(
                hlsutil.OutSegment(
                    # Absolute, unlike the live playlist. This one is served
                    # from the root, so a relative URI would resolve to /s*.aac
                    # and every segment would 404.
                    uri=f"/hls/{SEGMENT_PREFIX}{row['pdt_ms']}.aac",
                    duration=row["duration_ms"] / 1000,
                    pdt=datetime.fromtimestamp(row["pdt_ms"] / 1000, tz=UTC),
                    discontinuity=discontinuity,
                )
            )
            previous = row
        target = max(1, round(max(item.duration for item in out)))
        return hlsutil.build_media_playlist(out, rows[0]["seq"], target, vod=True)

    def health(self) -> dict:
        latest = db.latest_segment(self.conn)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        lag = (now_ms - latest["pdt_ms"]) / 1000 if latest else None
        state, remaining = self.buffer_state()
        # Health tracks whether we are recording, not whether the delay has
        # elapsed yet. Reporting failure throughout a six-hour fill would have
        # every orchestrator calling a perfectly healthy container sick.
        ingesting = lag is not None and lag < 120
        return {
            "ok": ingesting and state in {"ready", "filling", "empty", "gap"},
            "state": state,
            "ingest_lag_seconds": lag,
            "playout_ready": state == "ready",
            "seconds_until_ready": remaining,
            "segments": db.segment_count(self.conn),
            "delay_seconds": self.delay.current,
        }


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).isoformat()


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config()
    config.ensure_dirs()
    state = Playout(config)

    async def delay_watcher() -> None:
        while True:
            await asyncio.sleep(60)
            try:
                state.delay.consider(safe=state.is_safe_boundary())
            except Exception as exc:
                log.warning("delay update failed: %s", exc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(delay_watcher())
        yield
        task.cancel()

    app = FastAPI(title="RockFM Enrichment", lifespan=lifespan)
    app.state.playout = state

    def no_cache(body: str, media_type: str) -> Response:
        return Response(
            content=body,
            media_type=media_type,
            headers={"Cache-Control": "no-cache, no-store", "Access-Control-Allow-Origin": "*"},
        )

    @app.get("/hls/playlist.m3u8")
    def master() -> Response:
        return no_cache(hlsutil.build_master_playlist("chunks.m3u8"), M3U8_TYPE)

    @app.get("/hls/chunks.m3u8")
    def chunks() -> Response:
        return no_cache(state.media_playlist(), M3U8_TYPE)

    @app.get("/hls/{name}.aac")
    def segment(name: str) -> Response:
        if not name.startswith(SEGMENT_PREFIX):
            raise HTTPException(status_code=404, detail="unknown segment")
        try:
            pdt_ms = int(name[len(SEGMENT_PREFIX) :])
        except ValueError:
            raise HTTPException(status_code=404, detail="unknown segment") from None
        return Response(
            content=state.segment_bytes(pdt_ms),
            media_type=AAC_TYPE,
            headers={"Cache-Control": "public, max-age=60", "Access-Control-Allow-Origin": "*"},
        )

    @app.get("/art/{name}")
    def art(name: str) -> FileResponse:
        # Artwork is cached under a hashed name; refuse anything else so a
        # crafted path cannot walk out of the cache directory.
        if not name.endswith(".jpg") or not name[:-4].isalnum():
            raise HTTPException(status_code=404, detail="unknown artwork")
        path = config.art_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="unknown artwork")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400",
                     "Access-Control-Allow-Origin": "*"},
        )

    @app.get("/api/nowplaying")
    def nowplaying() -> JSONResponse:
        return JSONResponse(
            state.now_playing(), headers={"Access-Control-Allow-Origin": "*"}
        )

    @app.get("/api/timeline")
    def timeline(start: int | None = None, end: int | None = None, hours: float = 2.0) -> JSONResponse:
        latest = db.latest_segment(state.conn)
        earliest = db.earliest_segment(state.conn)
        anchor = latest["pdt_ms"] if latest else int(datetime.now(UTC).timestamp() * 1000)
        end_ms = end if end is not None else anchor
        if start is not None:
            start_ms = start
        else:
            # Asking for two hours when only twenty minutes exist would squeeze
            # everything into a sliver at the right-hand edge, so the window is
            # clamped to what has actually been recorded.
            start_ms = end_ms - int(hours * 3600 * 1000)
            if earliest is not None:
                start_ms = max(start_ms, earliest["pdt_ms"])
        return JSONResponse(
            {"range": {"start": start_ms, "end": end_ms},
             "items": state.timeline(start_ms, end_ms)},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.get("/api/appearance")
    def get_appearance() -> JSONResponse:
        return JSONResponse(
            {
                "configurable": list(appearance.CONFIGURABLE),
                "fields": list(appearance.FIELDS),
                "defaults": appearance.defaults(state.language),
                "current": state.appearance,
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.put("/api/appearance")
    async def put_appearance(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="expected a JSON object") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        saved = appearance.save(state.conn, payload, state.language)
        return JSONResponse({"current": saved}, headers={"Access-Control-Allow-Origin": "*"})

    @app.post("/api/reanalyze")
    def reanalyze(minutes: float | None = None) -> JSONResponse:
        """Send the analyzer back over audio it has already scanned.

        Its cursor only moves forward, so a fix to how audio is read never
        reaches anything already scanned. With no argument the whole buffer is
        re-read. With `minutes`, only the tail is -- and that is usually what is
        wanted: when one stretch was read wrong, redoing twenty minutes beats
        redoing six hours at four seconds a lookup.

        Rewinding also clears what was written for that stretch, because it was
        written from a reading of it now known to be wrong.
        """
        earliest = db.earliest_segment(state.conn)
        latest = db.latest_segment(state.conn)
        if minutes is None or earliest is None or latest is None:
            state.conn.execute(
                "DELETE FROM meta WHERE key = ?", (db.ANALYZER_CURSOR_KEY,)
            )
            from_ms = earliest["pdt_ms"] if earliest else None
        else:
            from_ms = max(
                earliest["pdt_ms"], int(latest["pdt_ms"] - minutes * 60_000)
            )
            db.clear_timeline_span(
                state.conn, from_ms, latest["pdt_ms"] + latest["duration_ms"] + 1
            )
            db.set_meta(state.conn, db.ANALYZER_CURSOR_KEY, str(from_ms))
        log.info("re-analysing from %s", _iso(from_ms) if from_ms else "the start")
        return JSONResponse(
            {
                "restarted": True,
                "from": _iso(from_ms) if from_ms else None,
                "buffered_seconds": state.status()["buffer"]["seconds"],
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        return JSONResponse(
            {
                "fields": {
                    name: {"type": kind, "default": default}
                    for name, (default, _env, kind) in settings.FIELDS.items()
                },
                "current": settings.public(state.settings),
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.put("/api/settings")
    async def put_settings(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="expected a JSON object") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        saved = settings.save(state.conn, payload)
        return JSONResponse(
            {"current": settings.public(saved)},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(state.status(), headers={"Access-Control-Allow-Origin": "*"})

    @app.get("/replay.m3u8")
    def replay(start: int, duration: float = 900.0) -> Response:
        return no_cache(state.replay_playlist(start, duration), M3U8_TYPE)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/health")
    def health() -> JSONResponse:
        payload = state.health()
        return JSONResponse(payload, status_code=200 if payload["ok"] else 503)

    @app.get("/", response_class=HTMLResponse)
    def player() -> HTMLResponse:
        return HTMLResponse(PLAYER_HTML)

    return app


PLAYER_HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RockFM · en diferido</title>
<style>
:root{color-scheme:dark;--bg:#0c0c0f;--fg:#f4f4f7;--dim:#8e8e99;--line:#23232b;--accent:#e03131}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  display:grid;place-items:center;min-height:100vh;padding:24px}
.card{width:min(400px,100%);text-align:center}
.artwrap{position:relative;border-radius:14px;overflow:hidden;background:#17171d;aspect-ratio:1}
img{width:100%;height:100%;object-fit:cover;display:block}
.kind{position:absolute;top:10px;left:10px;padding:3px 10px;border-radius:999px;
  background:rgba(0,0,0,.6);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase}
h1{font-size:1.2rem;margin:16px 0 2px;line-height:1.3}
p.sub{margin:0;color:var(--dim);font-size:.92rem}
.bar{height:4px;background:var(--line);border-radius:2px;margin:16px 0 6px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--accent);width:0;transition:width .6s linear}
.row{display:flex;justify-content:space-between;font-size:.76rem;color:var(--dim);
  font-variant-numeric:tabular-nums}
button{margin-top:18px;padding:11px 26px;border:0;border-radius:999px;background:var(--accent);
  color:#fff;font-size:1rem;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.meta{margin-top:14px;font-size:.72rem;color:var(--dim)}
</style></head><body>
<div class="card">
  <div class="artwrap"><img id="art" alt=""><span class="kind" id="kind"></span></div>
  <h1 id="primary">RockFM</h1>
  <p class="sub" id="secondary"></p>
  <div class="bar"><span id="fill"></span></div>
  <div class="row"><span id="elapsed">--:--</span><span id="duration">--:--</span></div>
  <button id="play">Escuchar</button>
  <div class="meta" id="meta"></div>
  <div class="meta"><a href="/dashboard" style="color:inherit">panel de control</a></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.17/hls.min.js"></script>
<script>
const SRC = '/hls/playlist.m3u8';
const audio = new Audio();
audio.preload = 'none';
const btn = document.getElementById('play');
let attached = false;

function attach() {
  if (attached) return;
  attached = true;
  if (window.Hls && Hls.isSupported()) {
    const hls = new Hls({ liveSyncDurationCount: 3 });
    hls.loadSource(SRC);
    hls.attachMedia(audio);
  } else {
    audio.src = SRC;   // Safari and iOS play HLS natively
  }
}

btn.addEventListener('click', () => {
  attach();
  if (audio.paused) audio.play().catch(() => {}); else audio.pause();
});
audio.addEventListener('play',  () => { btn.textContent = 'Pausar'; });
audio.addEventListener('pause', () => { btn.textContent = 'Escuchar'; });

const mmss = s => s == null ? '--:--'
  : Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
const set = (id, text) => {
  const el = document.getElementById(id);
  if (el.textContent !== text) el.textContent = text;
};

async function refresh() {
  try {
    const data = await (await fetch('/api/nowplaying', { cache: 'no-store' })).json();
    const np = data.now_playing;
    set('primary', np.primary || 'RockFM');
    set('secondary', np.secondary || '');
    set('kind', np.kind === 'cancion' ? '' : (np.kind || ''));
    const art = document.getElementById('art');
    const wanted = np.art || '';
    // An <img> with an empty src renders a broken-image icon, so drop the
    // attribute entirely and let the placeholder background show through.
    if (wanted) {
      if (art.getAttribute('src') !== wanted) art.setAttribute('src', wanted);
      art.hidden = false;
    } else {
      art.removeAttribute('src');
      art.hidden = true;
    }
    set('elapsed', mmss(np.elapsed));
    set('duration', mmss(np.duration));
    document.getElementById('fill').style.width =
      (np.duration ? Math.min(100, 100 * np.elapsed / np.duration) : 0) + '%';
    // source_time is a Madrid wall clock. Without an explicit timeZone the
    // browser would helpfully re-render it in the viewer's own zone, which is
    // precisely the thing this line exists to tell them.
    const madrid = new Date(data.source_time).toLocaleTimeString('es-ES', {
      hour: '2-digit', minute: '2-digit', timeZone: data.source_timezone || 'Europe/Madrid',
    });
    const buf = data.buffer || {};
    if (buf.state === 'filling' || buf.state === 'empty') {
      const left = Math.max(0, Math.round((buf.seconds_until_ready || 0) / 60));
      const hours = Math.floor(left / 60), mins = left % 60;
      set('primary', 'Rellenando el búfer');
      set('secondary', `listo en ${hours ? hours + ' h ' : ''}${mins} min`);
      set('kind', '');
      set('elapsed', '--:--');
      set('duration', '--:--');
      btn.disabled = true;
    } else {
      btn.disabled = false;
    }
    set('meta', `Diferido ${data.delay_hours} h · en España son las ${madrid}`);
  } catch (err) {
    set('meta', 'sin conexión con el servidor');
  }
}
refresh();
setInterval(refresh, 3000);
</script></body></html>
"""


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    config = Config()
    uvicorn.run(create_app(config), host=config.http_host, port=config.http_port)


if __name__ == "__main__":
    main()
