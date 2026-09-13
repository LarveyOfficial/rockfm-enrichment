"""Delayed HLS playout and the rich now-playing API.

Segments are served back as the bytes we recorded, with only their ID3 TIT2
frame rewritten to the label our analyzer worked out -- the audio itself is
never re-encoded. Playlist timestamps are shifted forward by the current delay
so the delayed stream presents as live.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse

from . import db, hlsutil, labels
from . import strings_es as S
from .config import Config
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
        return db.segments_upto(
            self.conn, self.position_ms(), self.config.playlist_segments
        )

    # --- playlists ---

    def media_playlist(self) -> str:
        rows = self.window()
        if not rows:
            raise HTTPException(status_code=503, detail="buffer is still filling")

        shift = timedelta(seconds=self.delay.current)
        out: list[hlsutil.OutSegment] = []
        previous: sqlite3.Row | None = None
        for row in rows:
            discontinuity = False
            if previous is not None:
                expected = previous["pdt_ms"] + previous["duration_ms"]
                if row["pdt_ms"] - expected > GAP_TOLERANCE_MS:
                    discontinuity = True
            pdt = datetime.fromtimestamp(row["pdt_ms"] / 1000, tz=timezone.utc) + shift
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
        primary, secondary = self._render(item)
        label = f"{primary} - {secondary}" if secondary else primary
        try:
            return hlsutil.rewrite_tit2(data, label)
        except Exception as exc:  # never fail playout over a metadata rewrite
            log.warning("TIT2 rewrite failed for %s: %s", pdt_ms, exc)
            return data

    # --- now playing ---

    def _render(self, item: sqlite3.Row | None) -> tuple[str, str]:
        language = self.config.display_language
        if item is None:
            return S.STATION_NAME, ""
        return labels.render(dict(item), language)

    def _describe(self, item: sqlite3.Row | None, position_ms: int) -> dict:
        primary, secondary = self._render(item)
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
            "art": item["art_url"],
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
        now = datetime.now(timezone.utc)
        position = self.position_ms(now)
        current = db.timeline_at(self.conn, position)
        upcoming = db.timeline_after(self.conn, position)
        earliest = db.earliest_segment(self.conn)
        latest = db.latest_segment(self.conn)
        return {
            "station": {"name": S.STATION_NAME, "language": self.config.display_language},
            "delay_seconds": self.delay.current,
            "delay_hours": round(self.delay.current / 3600, 2),
            "pending_delay_seconds": self.delay.pending.delay if self.delay.pending else None,
            "source_time": source_wallclock(self.config, now).isoformat(),
            "playout_time": _iso(position),
            "now_playing": self._describe(current, position),
            "next": self._describe(upcoming, position) if upcoming else None,
            "buffer": {
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

    def health(self) -> dict:
        latest = db.latest_segment(self.conn)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        lag = (now_ms - latest["pdt_ms"]) / 1000 if latest else None
        position = self.position_ms()
        playable = db.segment_at(self.conn, position) is not None
        healthy = bool(latest) and lag is not None and lag < 120 and playable
        return {
            "ok": healthy,
            "ingest_lag_seconds": lag,
            "playout_ready": playable,
            "segments": db.segment_count(self.conn),
            "delay_seconds": self.delay.current,
        }


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


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

    @app.get("/api/nowplaying")
    def nowplaying() -> JSONResponse:
        return JSONResponse(
            state.now_playing(), headers={"Access-Control-Allow-Origin": "*"}
        )

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
<title>RockFM · diferido</title>
<style>
:root{color-scheme:dark;--bg:#0d0d10;--fg:#f2f2f5;--dim:#9a9aa5;--accent:#e03131}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;
display:grid;place-items:center;min-height:100vh;padding:24px}
.card{width:min(420px,100%);text-align:center}
img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:12px;background:#1a1a20}
h1{font-size:1.25rem;margin:18px 0 4px}
p{margin:0;color:var(--dim);font-size:.95rem}
.bar{height:4px;background:#26262e;border-radius:2px;margin:18px 0 6px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--accent);width:0;transition:width .5s linear}
.row{display:flex;justify-content:space-between;font-size:.8rem;color:var(--dim)}
button{margin-top:18px;padding:10px 22px;border:0;border-radius:999px;background:var(--accent);
color:#fff;font-size:1rem;cursor:pointer}
.meta{margin-top:14px;font-size:.75rem;color:var(--dim)}
</style></head><body>
<div class="card">
  <img id="art" alt="">
  <h1 id="primary">RockFM</h1>
  <p id="secondary"></p>
  <div class="bar"><span id="fill"></span></div>
  <div class="row"><span id="elapsed">--:--</span><span id="duration">--:--</span></div>
  <button id="play">Escuchar</button>
  <div class="meta" id="meta"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.17/hls.min.js"></script>
<script>
const audio = new Audio();
const src = '/hls/playlist.m3u8';
document.getElementById('play').onclick = () => {
  if (audio.src || window.Hls?.isSupported()) { audio.play(); return; }
};
if (window.Hls && Hls.isSupported()) { const h = new Hls(); h.loadSource(src); h.attachMedia(audio); }
else { audio.src = src; }
document.getElementById('play').addEventListener('click', () => audio.play());
const fmt = s => s == null ? '--:--' : `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
async function tick() {
  try {
    const r = await fetch('/api/nowplaying'); const d = await r.json(); const n = d.now_playing;
    document.getElementById('primary').textContent = n.primary || 'RockFM';
    document.getElementById('secondary').textContent = n.secondary || '';
    const art = document.getElementById('art');
    if (n.art && art.src !== n.art) art.src = n.art;
    document.getElementById('elapsed').textContent = fmt(n.elapsed);
    document.getElementById('duration').textContent = fmt(n.duration);
    document.getElementById('fill').style.width =
      (n.duration ? Math.min(100, 100 * n.elapsed / n.duration) : 0) + '%';
    document.getElementById('meta').textContent =
      `Diferido ${d.delay_hours} h · hora en España ${new Date(d.source_time).toLocaleTimeString('es-ES')}`;
  } catch (e) {}
}
tick(); setInterval(tick, 3000);
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
