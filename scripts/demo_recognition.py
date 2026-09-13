"""Sweep a stretch of live RockFM and report what the audio actually is.

Probes are deliberately short (default 12 s) and densely spaced: a long clip
risks straddling a song change, which is exactly the boundary the analyzer is
trying to locate.

    python scripts/demo_recognition.py --seconds 240 --step 12
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rockfm import audio, db, ingest  # noqa: E402
from rockfm.config import Config  # noqa: E402
from rockfm.recognize import build  # noqa: E402

RATE = 16000
MADRID = ZoneInfo("Europe/Madrid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=240, help="how long to record")
    parser.add_argument("--probe", type=int, default=12, help="probe length in seconds")
    parser.add_argument("--step", type=int, default=12, help="seconds between probes")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between lookups")
    parser.add_argument("--recognizer", default="shazamio")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    data_dir = Path(args.data_dir or f"/tmp/rockfm-demo-{int(time.time())}")
    config = Config(data_dir=data_dir)
    config.ensure_dirs()

    ingestor = ingest.Ingestor(config, db.ThreadLocalDB(config.db_path))
    threading.Thread(target=ingestor.run, daemon=True).start()
    print(f"recording {args.seconds}s of live RockFM into {data_dir} ...", flush=True)
    time.sleep(args.seconds)
    ingestor.stop_event.set()
    time.sleep(1)

    conn = db.connect(config.db_path)
    rows = list(conn.execute("SELECT pdt_ms, relpath FROM segments ORDER BY pdt_ms"))
    if not rows:
        print("no segments captured")
        return

    samples = audio.decode_segment_files(
        [config.segments_dir / row["relpath"] for row in rows], rate=RATE
    )
    base_ms = rows[0]["pdt_ms"]
    recognizer = build(args.recognizer)
    print(f"{len(rows)} segments = {len(samples) / RATE:.0f}s of audio\n")
    print(f"{'MADRID':<10} {'offset':>7}  RECOGNISED FROM THE AUDIO")
    print("-" * 68)

    matched = total = 0
    for start in range(0, max(0, int(len(samples) / RATE) - args.probe) + 1, args.step):
        probe = samples[start * RATE : (start + args.probe) * RATE]
        if probe.size < args.probe * RATE // 2:
            break
        result = recognizer.recognize(probe, RATE)
        total += 1
        matched += result is not None
        clock = datetime.fromtimestamp(
            (base_ms + start * 1000) / 1000, tz=timezone.utc
        ).astimezone(MADRID)
        found = (
            f"{result.artist} - {result.title}"
            if result
            else "(no match: talk / ads / jingle)"
        )
        print(f"{clock:%H:%M:%S}   {start:>5}s  {found}", flush=True)
        time.sleep(args.pause)

    print(f"\n{matched}/{total} probes identified a song")


if __name__ == "__main__":
    main()
