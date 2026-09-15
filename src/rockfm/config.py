"""Environment configuration: the things that cannot change while running.

Where the data lives, how much to keep, which recogniser to load,
and the timezone the delay is computed against. Everything else -- the AzuraCast
integration, display language, public URL, classification thresholds -- is a
runtime setting stored in the database and edited from the dashboard, because
restarting to change it would tear a hole in the recording. See settings.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_SOURCE = "https://rockfm-cope.flumotion.com/playlist.m3u8"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _opt_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class Config:
    # --- source ---
    source_url: str = field(default_factory=lambda: _env("SOURCE_URL", DEFAULT_SOURCE))
    user_agent: str = field(default_factory=lambda: _env("USER_AGENT", "Mozilla/5.0"))
    referer: str = field(default_factory=lambda: _env("REFERER", "https://www.rockfm.fm/"))

    # --- storage ---
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    buffer_hours: int = field(default_factory=lambda: _int("BUFFER_HOURS", 24))

    # --- ingest ---
    poll_interval: float = field(default_factory=lambda: _float("POLL_INTERVAL", 3.0))
    http_timeout: float = field(default_factory=lambda: _float("HTTP_TIMEOUT", 8.0))

    # --- time shift ---
    source_tz_name: str = field(default_factory=lambda: _env("SOURCE_TZ", "Europe/Madrid"))
    local_tz_name: str = field(default_factory=lambda: _env("LOCAL_TZ", "America/New_York"))
    delay_seconds_override: int | None = field(default_factory=lambda: _opt_int("DELAY_SECONDS"))
    lead_time_minutes: int = field(default_factory=lambda: _int("LEAD_TIME_MINUTES", 30))

    # --- playout ---
    http_host: str = field(default_factory=lambda: _env("HTTP_HOST", "0.0.0.0"))
    http_port: int = field(default_factory=lambda: _int("HTTP_PORT", 8080))
    playlist_segments: int = field(default_factory=lambda: _int("PLAYLIST_SEGMENTS", 6))

    # --- recognition / classification ---
    recognizer: str = field(default_factory=lambda: _env("RECOGNIZER", "shazamio"))

    # --- azuracast ---
    # Only the codec stays here: changing it means restarting the ffmpeg source
    # anyway. Everything else about the integration -- including whether it runs
    # at all -- is a runtime setting, see settings.py.
    azuracast_dj_codec: str = field(default_factory=lambda: _env("AZURACAST_DJ_CODEC", "mp3"))

    @property
    def source_tz(self) -> ZoneInfo:
        return ZoneInfo(self.source_tz_name)

    @property
    def local_tz(self) -> ZoneInfo:
        return ZoneInfo(self.local_tz_name)

    @property
    def segments_dir(self) -> Path:
        return self.data_dir / "segments"

    @property
    def art_dir(self) -> Path:
        return self.data_dir / "art"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rockfm.db"

    @property
    def http_headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Referer": self.referer}

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.segments_dir, self.art_dir):
            path.mkdir(parents=True, exist_ok=True)


def load() -> Config:
    return Config()
