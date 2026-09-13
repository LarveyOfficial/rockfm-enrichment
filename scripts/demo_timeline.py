"""Record a stretch of RockFM, run the analyzer over it, and print the timeline.

This is the acceptance check for recognition: it shows each item the analyzer
committed, with boundaries, duration, how it was identified and how many
external recogniser calls it cost.

    python scripts/demo_timeline.py --seconds 600
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rockfm import db, ingest, labels  # noqa: E402
from rockfm.analyzer import Analyzer  # noqa: E402
from rockfm.classify.decide import Classifier  # noqa: E402
from rockfm.config import Config  # noqa: E402

MADRID = ZoneInfo("Europe/Madrid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--skip-record", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("rockfm.ingest").setLevel(logging.WARNING)

    data_dir = Path(args.data_dir or f"/tmp/rockfm-timeline-{int(time.time())}")
    config = Config(data_dir=data_dir)
    config.ensure_dirs()

    if not args.skip_record:
        ingestor = ingest.Ingestor(config, db.ThreadLocalDB(config.db_path))
        threading.Thread(target=ingestor.run, daemon=True).start()
        print(f"recording {args.seconds}s into {data_dir} ...", flush=True)
        time.sleep(args.seconds)
        ingestor.stop_event.set()
        time.sleep(1)

    conn = db.connect(config.db_path)
    analyzer = Analyzer(config, conn)
    print("\nanalyzing ...\n", flush=True)
    started = time.time()
    while analyzer.run_once() > 0:
        pass
    elapsed = time.time() - started

    classified = 0
    if not args.no_classify:
        print("classifying non-song stretches ...\n", flush=True)
        classifier = Classifier(config, conn)
        classified = classifier.run_once()

    rows = list(conn.execute("SELECT * FROM timeline ORDER BY start_ms"))
    print(f"\n{'MADRID':<10} {'DUR':>7}  {'KIND':<12}  WHAT IS SHOWN")
    print("-" * 96)
    for row in rows:
        clock = datetime.fromtimestamp(row["start_ms"] / 1000, tz=UTC).astimezone(MADRID)
        primary, secondary = labels.render(dict(row))
        shown = primary + (f"   ·   {secondary}" if secondary else "")
        print(
            f"{clock:%H:%M:%S}  {(row['end_ms'] - row['start_ms']) / 1000:6.1f}s  "
            f"{row['kind']:<12}  {shown}"
        )
    songs = [r for r in rows if r["kind"] == "cancion"]
    print(
        f"\n{len(rows)} items ({len(songs)} songs, {classified} stretches classified) | "
        f"{analyzer.external_calls} external recogniser calls | analysis took {elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
