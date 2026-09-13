"""HLS playlist parsing/generation and ID3 handling for packed-AAC segments.

RockFM serves packed audio (raw ADTS `.aac`) rather than MPEG-TS, so each
segment is prefixed by one or more ID3v2 tags. Per Apple's packed-audio spec the
*first* tag carries a PRIV frame
(`com.apple.streaming.transportStreamTimestamp`) that players use for timing.

We write a TIT2 frame on the way out so ordinary HLS players show our own
label. We never read the upstream one: it is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

ID3_MAGIC = b"ID3"


# --- playlists --------------------------------------------------------------


@dataclass(frozen=True)
class MediaSegment:
    seq: int
    uri: str
    duration: float
    pdt: datetime | None

    @property
    def pdt_ms(self) -> int | None:
        if self.pdt is None:
            return None
        return int(self.pdt.timestamp() * 1000)

    @property
    def duration_ms(self) -> int:
        return int(round(self.duration * 1000))


@dataclass(frozen=True)
class MediaPlaylist:
    target_duration: float
    media_sequence: int
    segments: tuple[MediaSegment, ...]


def parse_datetime(raw: str) -> datetime:
    """Parse an EXT-X-PROGRAM-DATE-TIME value into an aware UTC datetime."""
    value = raw.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_master_playlist(text: str) -> bool:
    return "#EXT-X-STREAM-INF" in text


def parse_master(text: str, base_url: str) -> list[str]:
    """Return variant playlist URLs, highest bandwidth first."""
    variants: list[tuple[int, str]] = []
    bandwidth = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-STREAM-INF"):
            bandwidth = 0
            for part in line.split(":", 1)[-1].split(","):
                key, _, value = part.partition("=")
                if key.strip().upper() == "BANDWIDTH":
                    bandwidth = int(value.strip() or 0)
        elif line and not line.startswith("#"):
            variants.append((bandwidth, urljoin(base_url, line)))
            bandwidth = 0
    variants.sort(key=lambda item: item[0], reverse=True)
    return [url for _, url in variants]


def parse_media(text: str, base_url: str) -> MediaPlaylist:
    target_duration = 0.0
    media_sequence = 0
    segments: list[MediaSegment] = []
    pending_duration = 0.0
    pending_pdt: datetime | None = None
    index = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-TARGETDURATION:"):
            target_duration = float(line.split(":", 1)[1])
        elif line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            media_sequence = int(line.split(":", 1)[1])
        elif line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pending_pdt = parse_datetime(line.split(":", 1)[1])
        elif line.startswith("#EXTINF:"):
            pending_duration = float(line.split(":", 1)[1].split(",")[0])
        elif not line.startswith("#"):
            segments.append(
                MediaSegment(
                    seq=media_sequence + index,
                    uri=urljoin(base_url, line),
                    duration=pending_duration or target_duration,
                    pdt=pending_pdt,
                )
            )
            index += 1
            pending_duration = 0.0
            pending_pdt = None

    return MediaPlaylist(
        target_duration=target_duration,
        media_sequence=media_sequence,
        segments=tuple(segments),
    )


def format_datetime(moment: datetime) -> str:
    """Render an EXT-X-PROGRAM-DATE-TIME value with millisecond precision."""
    utc = moment.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class OutSegment:
    uri: str
    duration: float
    pdt: datetime
    discontinuity: bool = False


def build_media_playlist(
    segments: list[OutSegment], media_sequence: int, target_duration: int
) -> str:
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}",
    ]
    for segment in segments:
        if segment.discontinuity:
            lines.append("#EXT-X-DISCONTINUITY")
        lines.append(f"#EXT-X-PROGRAM-DATE-TIME:{format_datetime(segment.pdt)}")
        lines.append(f"#EXTINF:{segment.duration:.3f},")
        lines.append(segment.uri)
    return "\n".join(lines) + "\n"


def build_master_playlist(variant_uri: str, bandwidth: int = 64000) -> str:
    return (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS="mp4a.40.2"\n'
        f"{variant_uri}\n"
    )


# --- ID3 --------------------------------------------------------------------


def _syncsafe_decode(raw: bytes) -> int:
    value = 0
    for byte in raw:
        value = (value << 7) | (byte & 0x7F)
    return value


def _syncsafe_encode(value: int) -> bytes:
    return bytes(
        (
            (value >> 21) & 0x7F,
            (value >> 14) & 0x7F,
            (value >> 7) & 0x7F,
            value & 0x7F,
        )
    )


@dataclass(frozen=True)
class Id3Frame:
    frame_id: str
    flags: bytes
    data: bytes


def _parse_frames(body: bytes, syncsafe: bool) -> list[Id3Frame] | None:
    """Parse frames out of a tag body; None if the layout looks wrong."""
    frames: list[Id3Frame] = []
    offset = 0
    while offset + 10 <= len(body):
        frame_id = body[offset : offset + 4]
        if not frame_id.strip(b"\x00"):
            break  # padding
        if not all(0x30 <= byte <= 0x5A for byte in frame_id):
            return None
        raw_size = body[offset + 4 : offset + 8]
        size = _syncsafe_decode(raw_size) if syncsafe else int.from_bytes(raw_size, "big")
        end = offset + 10 + size
        if size < 0 or end > len(body):
            return None
        frames.append(
            Id3Frame(
                frame_id=frame_id.decode("latin1"),
                flags=body[offset + 8 : offset + 10],
                data=body[offset + 10 : end],
            )
        )
        offset = end
    return frames


@dataclass(frozen=True)
class Id3Tag:
    version: tuple[int, int]
    flags: int
    frames: tuple[Id3Frame, ...]
    raw: bytes
    syncsafe_frames: bool


def split_id3(data: bytes) -> tuple[list[Id3Tag], int]:
    """Parse the leading ID3 tags; returns (tags, offset of the audio payload)."""
    tags: list[Id3Tag] = []
    offset = 0
    while (
        len(data) >= offset + 10
        and data[offset : offset + 3] == ID3_MAGIC
        and data[offset + 3] != 0xFF
    ):
        major = data[offset + 3]
        revision = data[offset + 4]
        flags = data[offset + 5]
        size = _syncsafe_decode(data[offset + 6 : offset + 10])
        end = offset + 10 + size
        if end > len(data):
            break
        body = data[offset + 10 : end]
        # v2.4 mandates syncsafe frame sizes but encoders disagree; try both.
        prefer_syncsafe = major >= 4
        frames = _parse_frames(body, prefer_syncsafe)
        used_syncsafe = prefer_syncsafe
        if frames is None:
            frames = _parse_frames(body, not prefer_syncsafe)
            used_syncsafe = not prefer_syncsafe
        if frames is None:
            break
        tags.append(
            Id3Tag(
                version=(major, revision),
                flags=flags,
                frames=tuple(frames),
                raw=data[offset:end],
                syncsafe_frames=used_syncsafe,
            )
        )
        offset = end
    return tags, offset


def _decode_text_frame(data: bytes) -> str:
    if not data:
        return ""
    encoding, payload = data[0], data[1:]
    codecs = {0: "latin1", 1: "utf-16", 2: "utf-16-be", 3: "utf-8"}
    text = payload.decode(codecs.get(encoding, "utf-8"), errors="replace")
    return text.rstrip("\x00").strip()


def _read_tit2(data: bytes) -> str | None:
    """Read a TIT2 label back. Used by tests to verify our own rewrite."""
    tags, _ = split_id3(data)
    for tag in tags:
        for frame in tag.frames:
            if frame.frame_id == "TIT2":
                return _decode_text_frame(frame.data) or None
    return None


def _build_tag(frames: list[Id3Frame], syncsafe: bool) -> bytes:
    body = bytearray()
    for frame in frames:
        size = len(frame.data)
        raw_size = _syncsafe_encode(size) if syncsafe else size.to_bytes(4, "big")
        body += frame.frame_id.encode("latin1")[:4].ljust(4, b"\x00")
        body += raw_size
        body += frame.flags[:2].ljust(2, b"\x00")
        body += frame.data
    return b"ID3" + bytes((4, 0, 0)) + _syncsafe_encode(len(body)) + bytes(body)


def rewrite_tit2(data: bytes, text: str) -> bytes:
    """Return `data` with its TIT2 replaced by `text`.

    The first ID3 tag is preserved byte-for-byte because it carries the Apple
    transport-stream timestamp PRIV frame that players rely on for timing. Our
    label goes into a following tag, and any pre-existing TIT2 is dropped.
    """
    tags, payload_offset = split_id3(data)
    payload = data[payload_offset:]

    new_frame = Id3Frame(
        frame_id="TIT2", flags=b"\x00\x00", data=b"\x03" + text.encode("utf-8")
    )

    if not tags:
        return _build_tag([new_frame], syncsafe=True) + payload

    first = tags[0]
    carried: list[Id3Frame] = []
    if any(frame.frame_id == "TIT2" for frame in first.frames):
        # Rebuild the first tag without its TIT2, keeping PRIV and friends.
        kept = [frame for frame in first.frames if frame.frame_id != "TIT2"]
        head = _build_tag(kept, syncsafe=first.syncsafe_frames)
    else:
        head = first.raw
    for tag in tags[1:]:
        carried.extend(frame for frame in tag.frames if frame.frame_id != "TIT2")

    return head + _build_tag([*carried, new_frame], syncsafe=True) + payload
