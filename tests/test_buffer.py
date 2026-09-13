import pytest

from rockfm import db
from rockfm.buffer import BufferReader
from rockfm.config import Config

pytest.importorskip("numpy")


def _write_segment(config: Config, pdt_ms: int, relpath: str, payload: bytes) -> None:
    target = config.segments_dir / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def test_available_spans_the_recorded_range(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    for index in range(3):
        pdt = 1_000_000 + index * 6000
        db.insert_segment(
            conn, pdt_ms=pdt, seq=index, duration_ms=6000,
            relpath=f"{index}.aac", size=1, fetched_ms=0,
        )
    reader = BufferReader(conn, config)
    span = reader.available()
    assert span is not None
    assert span.start_ms == 1_000_000
    assert span.end_ms == 1_000_000 + 3 * 6000
    assert span.duration_ms == 18000


def test_available_is_none_when_empty(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    reader = BufferReader(db.connect(config.db_path), config)
    assert reader.available() is None


def test_read_returns_nothing_outside_the_buffer(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    db.insert_segment(
        conn, pdt_ms=1_000_000, seq=0, duration_ms=6000,
        relpath="0.aac", size=1, fetched_ms=0,
    )
    reader = BufferReader(conn, config)
    assert reader.read(500_000, 1000).size == 0


def test_gap_detection(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    db.record_gap(
        conn, after_pdt_ms=1_000_000, before_pdt_ms=1_030_000,
        missing_ms=30_000, reason="missed_segments", detected_ms=0,
    )
    reader = BufferReader(conn, config)
    assert reader.has_gap(999_000, 1_100_000)
    assert not reader.has_gap(2_000_000, 2_100_000)


def test_pruning_drops_segments_and_their_timeline_but_keeps_fingerprints(tmp_path):
    """Learned fingerprints must outlive the audio they came from."""
    from rockfm.fingerprint import FingerprintIndex
    from rockfm.ingest import Ingestor

    config = Config(data_dir=tmp_path, buffer_hours=1)
    config.ensure_dirs()
    database = db.ThreadLocalDB(config.db_path)
    conn = database.conn

    now_ms = int(__import__("time").time() * 1000)
    old, recent = now_ms - 3 * 3600 * 1000, now_ms - 60_000
    for pdt, name in ((old, "old.aac"), (recent, "new.aac")):
        target = config.segments_dir / name
        target.write_bytes(b"x")
        db.insert_segment(
            conn, pdt_ms=pdt, seq=0, duration_ms=6000, relpath=name, size=1, fetched_ms=0
        )
    db.upsert_timeline(conn, {"start_ms": old, "end_ms": old + 6000, "kind": "cancion"}, 0)
    db.upsert_timeline(conn, {"start_ms": recent, "end_ms": recent + 6000, "kind": "cancion"}, 0)
    FingerprintIndex(conn).add(kind="music", key="a|b", hashes=[(1, 0), (2, 1)], title="b")

    removed = Ingestor(config, database).prune()

    assert removed == 1
    assert not (config.segments_dir / "old.aac").exists()
    assert (config.segments_dir / "new.aac").exists()
    assert db.segment_count(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM fp_hashes").fetchone()[0] == 2
