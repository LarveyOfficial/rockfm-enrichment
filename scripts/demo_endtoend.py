"""Full pipeline against live radio, with a short delay so it finishes quickly.

Records, analyzes, classifies, then serves the delayed stream and checks that
what comes out of the now-playing API and out of the segments' ID3 matches what
the analyzer worked out.

    python scripts/demo_endtoend.py --seconds 400 --delay 180
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from rockfm import db, hlsutil, ingest, playout  # noqa: E402
from rockfm.analyzer import Analyzer  # noqa: E402
from rockfm.classify.decide import Classifier  # noqa: E402
from rockfm.config import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=400)
    parser.add_argument("--delay", type=int, default=0,
                        help="0 picks a delay that lands mid-song in what was just analyzed")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "rockfm.ingest", "uvicorn", "uvicorn.error"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_dir = Path(args.data_dir or f"/tmp/rockfm-e2e-{int(time.time())}")
    config = Config(data_dir=data_dir, http_port=args.port)
    config.ensure_dirs()

    ingestor = ingest.Ingestor(config, db.ThreadLocalDB(config.db_path))
    threading.Thread(target=ingestor.run, daemon=True).start()
    print(f"recording {args.seconds}s (delay {args.delay}s) ...", flush=True)
    time.sleep(args.seconds)

    conn = db.connect(config.db_path)
    print("\nanalyzing + classifying ...\n", flush=True)
    analyzer = Analyzer(config, conn)
    while analyzer.run_once() > 0:
        pass
    Classifier(config, conn).run_once()

    # In production the delay is ~6h, so playout only ever reaches audio the
    # analyzer finished with hours ago. Compressing that into a few minutes means
    # picking a delay that lands inside what was just analyzed -- otherwise the
    # position falls in the trailing run the analyzer deliberately holds back so
    # it can finish the song once more audio arrives.
    delay = args.delay
    if not delay:
        row = conn.execute(
            "SELECT start_ms, end_ms FROM timeline WHERE kind = 'cancion'"
            " ORDER BY start_ms DESC LIMIT 1"
        ).fetchone()
        if row is None:
            print("no songs were identified; nothing to serve")
            ingestor.stop_event.set()
            return
        midpoint = (row["start_ms"] + row["end_ms"]) // 2
        delay = max(30, int((time.time() * 1000 - midpoint) / 1000))
    print(f"serving with a {delay}s delay so playout sits mid-song\n")
    config = Config(data_dir=data_dir, http_port=args.port, delay_seconds_override=delay)

    app = playout.create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(2)

    base = f"http://127.0.0.1:{args.port}"
    client = httpx.Client(timeout=20)

    print("=== /api/nowplaying ===")
    payload = client.get(f"{base}/api/nowplaying").json()
    now = payload["now_playing"]
    for field in ("kind", "primary", "secondary", "art", "duration", "elapsed", "remaining"):
        print(f"  {field:<10}: {now.get(field)}")
    print(f"  delay     : {payload['delay_hours']}h   Madrid clock: {payload['source_time']}")

    print("\n=== ID3 written into the served segment ===")
    chunks = client.get(f"{base}/hls/chunks.m3u8").text
    media = hlsutil.parse_media(chunks, f"{base}/hls/")
    segment = client.get(media.segments[len(media.segments) // 2].uri).content
    print(f"  TIT2      : {hlsutil._read_tit2(segment)}")

    if now.get("art"):
        art = client.get(f"{base}{now['art']}")
        print(f"  artwork   : {art.status_code} {len(art.content)} bytes")

    print("\n=== ffmpeg decodes the delayed stream ===")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", f"{base}/hls/playlist.m3u8",
         "-t", "10", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    for line in result.stderr.splitlines():
        if "Audio:" in line or "time=" in line:
            print("  " + line.strip())
    print(f"  ffmpeg exit: {result.returncode}")

    ingestor.stop_event.set()


if __name__ == "__main__":
    main()
