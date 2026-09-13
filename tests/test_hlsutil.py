from datetime import UTC, datetime

import pytest

from rockfm import hlsutil

MASTER = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=64113,CODECS="mp4a.40.2"
chunks.m3u8
"""

MEDIA = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:7
#EXT-X-MEDIA-SEQUENCE:129160
#EXT-X-PROGRAM-DATE-TIME:2026-09-13T20:27:10.601Z
#EXTINF:6.016,
l_41866_774960042_129160.aac
#EXT-X-PROGRAM-DATE-TIME:2026-09-13T20:27:16.617Z
#EXTINF:6.016,
l_41866_774966058_129161.aac
"""

BASE = "https://example.com/live/chunks.m3u8"


def test_master_playlists_are_recognised():
    assert hlsutil.is_master_playlist(MASTER)
    assert not hlsutil.is_master_playlist(MEDIA)


def test_master_variants_resolve_to_absolute_urls():
    assert hlsutil.parse_master(MASTER, "https://example.com/live/playlist.m3u8") == [
        "https://example.com/live/chunks.m3u8"
    ]


def test_media_playlist_parsing():
    playlist = hlsutil.parse_media(MEDIA, BASE)
    assert playlist.media_sequence == 129160
    assert playlist.target_duration == 7
    assert len(playlist.segments) == 2

    first = playlist.segments[0]
    assert first.seq == 129160
    assert first.duration == pytest.approx(6.016)
    assert first.uri.endswith("l_41866_774960042_129160.aac")
    assert first.pdt == datetime(2026, 9, 13, 20, 27, 10, 601000, tzinfo=UTC)
    assert playlist.segments[1].seq == 129161


def test_program_date_time_round_trips():
    moment = hlsutil.parse_datetime("2026-09-13T20:27:10.601Z")
    assert hlsutil.format_datetime(moment) == "2026-09-13T20:27:10.601Z"


def test_generated_playlist_is_parseable_again():
    moment = hlsutil.parse_datetime("2026-09-13T20:27:10.601Z")
    body = hlsutil.build_media_playlist(
        [
            hlsutil.OutSegment(uri="s1.aac", duration=6.016, pdt=moment),
            hlsutil.OutSegment(uri="s2.aac", duration=6.016, pdt=moment, discontinuity=True),
        ],
        media_sequence=42,
        target_duration=7,
    )
    assert "#EXT-X-DISCONTINUITY" in body
    parsed = hlsutil.parse_media(body, BASE)
    assert parsed.media_sequence == 42
    assert len(parsed.segments) == 2
    assert parsed.segments[0].pdt == moment


# --- ID3 -------------------------------------------------------------------


def _syncsafe(value: int) -> bytes:
    return bytes(((value >> 21) & 0x7F, (value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F))


def _frame(frame_id: bytes, payload: bytes) -> bytes:
    return frame_id + _syncsafe(len(payload)) + b"\x00\x00" + payload


def _tag(*frames: bytes) -> bytes:
    body = b"".join(frames)
    return b"ID3" + bytes((4, 0, 0)) + _syncsafe(len(body)) + body


PRIV = _frame(b"PRIV", b"com.apple.streaming.transportStreamTimestamp\x00" + b"\x00" * 8)
ADTS = b"\xff\xf1" + b"\x00" * 200  # stand-in for the audio payload


def test_splits_leading_tags_from_the_audio():
    data = _tag(PRIV) + _tag(_frame(b"TIT2", b"\x03OLD - LABEL")) + ADTS
    tags, offset = hlsutil.split_id3(data)
    assert [f.frame_id for tag in tags for f in tag.frames] == ["PRIV", "TIT2"]
    assert data[offset:] == ADTS


def test_rewrite_replaces_the_label_and_leaves_audio_untouched():
    data = _tag(PRIV) + _tag(_frame(b"TIT2", b"\x03OLD - LABEL")) + ADTS
    out = hlsutil.rewrite_tit2(data, "Blondie - Denis")
    assert hlsutil._read_tit2(out) == "Blondie - Denis"
    _, offset = hlsutil.split_id3(out)
    assert out[offset:] == ADTS


def test_rewrite_preserves_the_apple_timestamp_frame():
    data = _tag(PRIV) + _tag(_frame(b"TIT2", b"\x03OLD")) + ADTS
    out = hlsutil.rewrite_tit2(data, "New")
    tags, _ = hlsutil.split_id3(out)
    privs = [f for tag in tags for f in tag.frames if f.frame_id == "PRIV"]
    assert len(privs) == 1
    assert b"transportStreamTimestamp" in privs[0].data
    # The first tag must stay byte-identical: players time off it.
    assert out.startswith(_tag(PRIV))


def test_rewrite_adds_a_label_when_there_was_none():
    data = _tag(PRIV) + ADTS
    out = hlsutil.rewrite_tit2(data, "Only Label")
    assert hlsutil._read_tit2(out) == "Only Label"
    _, offset = hlsutil.split_id3(out)
    assert out[offset:] == ADTS


def test_rewrite_handles_audio_with_no_tags_at_all():
    out = hlsutil.rewrite_tit2(ADTS, "Fresh")
    assert hlsutil._read_tit2(out) == "Fresh"


def test_rewrite_keeps_non_ascii_intact():
    data = _tag(PRIV) + ADTS
    out = hlsutil.rewrite_tit2(data, "El Pirata y su banda - Álex Clavero")
    assert hlsutil._read_tit2(out) == "El Pirata y su banda - Álex Clavero"


def test_reading_a_label_that_is_absent_returns_none():
    assert hlsutil._read_tit2(_tag(PRIV) + ADTS) is None
