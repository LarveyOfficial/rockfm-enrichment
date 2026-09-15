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


def test_pruning_drops_segments_and_their_timeline(tmp_path):
    """Audio ages out of the buffer, and what was said about it goes with it."""
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

    removed = Ingestor(config, database).prune()

    assert removed == 1
    assert not (config.segments_dir / "old.aac").exists()
    assert (config.segments_dir / "new.aac").exists()
    assert db.segment_count(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 1


def test_timeline_upsert_replaces_overlapping_rows_atomically(tmp_path):
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)

    db.upsert_timeline(conn, {"start_ms": 0, "end_ms": 10_000, "kind": "cancion"}, 0)
    db.upsert_timeline(conn, {"start_ms": 10_000, "end_ms": 20_000, "kind": "programa"}, 0)
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 2

    # A row straddling both replaces both, leaving exactly one.
    db.upsert_timeline(conn, {"start_ms": 5_000, "end_ms": 15_000, "kind": "programa"}, 0)
    rows = list(conn.execute("SELECT kind, start_ms, end_ms FROM timeline"))
    assert len(rows) == 1
    assert rows[0]["kind"] == "programa"


def test_abutting_timeline_rows_do_not_displace_each_other(tmp_path):
    """Contiguous items share a boundary exactly; that must not count as overlap."""
    config = Config(data_dir=tmp_path)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    db.upsert_timeline(conn, {"start_ms": 0, "end_ms": 10_000, "kind": "cancion"}, 0)
    db.upsert_timeline(conn, {"start_ms": 10_000, "end_ms": 20_000, "kind": "cancion"}, 0)
    assert conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0] == 2


# --- a hole in the recording must stay a hole -------------------------------


def _recorded(tmp_path, pdts, duration_ms=6000):
    """A buffer holding segments at the given timestamps, and a reader for it."""
    from rockfm.buffer import BufferReader

    config = Config(data_dir=tmp_path, buffer_hours=6)
    config.ensure_dirs()
    conn = db.connect(config.db_path)
    for index, pdt in enumerate(pdts):
        (config.segments_dir / f"{pdt}.aac").write_bytes(b"x")
        db.insert_segment(conn, pdt_ms=pdt, seq=index, duration_ms=duration_ms,
                          relpath=f"{pdt}.aac", size=1, fetched_ms=0)
    conn.commit()
    return BufferReader(conn, config)


def _decode_as(monkeypatch, marker_for, duration_ms=6000):
    """Decode each run to a constant, so where it lands is visible."""
    import numpy as np

    from rockfm import buffer as buffer_module

    def decode(paths, rate):
        marker = marker_for(int(paths[0].stem))
        return np.full(int(len(paths) * duration_ms * rate / 1000), marker, dtype=np.float32)

    monkeypatch.setattr(buffer_module, "decode_segment_files", decode)


def test_audio_after_a_hole_keeps_its_own_timestamps(tmp_path, monkeypatch):
    """Closing the recording up around a hole moves everything after it earlier.

    Fourteen minutes went missing one night, and the songs that followed were
    written into the timeline inside the gap they came after -- identified
    correctly, an hour of them at the wrong time.
    """
    import numpy as np

    rate = 8000
    reader = _recorded(tmp_path, [0, 6000, 60_000, 66_000])
    _decode_as(monkeypatch, lambda pdt: 1.0 if pdt < 60_000 else 2.0)

    span = reader.read(0, 72_000, rate=rate)
    at = lambda ms: span[int(ms * rate / 1000)]

    assert at(0) == 1.0 and at(11_000) == 1.0, "the first run moved"
    assert at(30_000) == 0.0, "the hole was closed up instead of left silent"
    assert at(60_500) == 2.0, "audio after the hole did not keep its own time"
    assert np.count_nonzero(span == 0.0) > 0


def test_a_short_recording_is_not_padded_out_at_the_end(tmp_path, monkeypatch):
    """A short read is how the analyzer knows it reached the edge of the tape."""
    rate = 8000
    reader = _recorded(tmp_path, [0, 6000])
    _decode_as(monkeypatch, lambda _pdt: 1.0)

    span = reader.read(0, 600_000, rate=rate)
    assert span.size == int(12_000 * rate / 1000)


def test_a_run_starting_before_the_span_is_trimmed_not_shifted(tmp_path, monkeypatch):
    rate = 8000
    reader = _recorded(tmp_path, [0, 6000, 12_000])
    _decode_as(monkeypatch, lambda _pdt: 1.0)

    span = reader.read(9_000, 6_000, rate=rate)
    assert span.size == int(6_000 * rate / 1000)
    assert float(span[0]) == 1.0
