"""Runtime settings, editable from the dashboard.

Anything that might reasonably change while the container is running lives here
rather than in the environment. Restarting to change an AzuraCast password costs
a gap in the recording and throws away whatever the analyzer was part way
through, which is a steep price for a typo.

What stays in the environment is what genuinely cannot change without a restart
or a rebuild: where the data lives, how much of it to keep, which recogniser and
and the timezone the delay is computed against.

Stored as JSON in the meta table, so no migration is needed. An environment
variable still seeds a setting the first time, so an existing deployment keeps
working and its values simply become editable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Final

from . import db

log = logging.getLogger("rockfm.settings")

META_KEY: Final = "settings"
MASK: Final = "••••••••"

# name -> (default, environment variable that seeds it, type)
FIELDS: Final[dict[str, tuple[Any, str, str]]] = {
    "public_url": ("", "PUBLIC_URL", "text"),
    "display_language": ("es", "DISPLAY_LANGUAGE", "text"),
    # Two separate judgements that happen to be about short stretches of audio.
    # One asks whether a non-song stretch is worth naming at all; the other how
    # much of a gap between two songs is just the crossfade. A station with long
    # crossfades and short jingles needs them set differently.
    # Seconds to leave between external lookups. Shazam answered forty-five
    # calls out of forty-five at one every four seconds and refused with 429 at
    # one every two, so four is the floor: this can slow lookups down further
    # but is clamped to it on the way up.
    "recognizer_interval_seconds": (4.0, "RECOGNIZER_INTERVAL_SECONDS", "number"),
    "min_nonmusic_seconds": (5.0, "MIN_NONMUSIC_SECONDS", "number"),
    "max_seam_seconds": (10.0, "MAX_SEAM_SECONDS", "number"),
    "azuracast_enabled": (False, "", "bool"),
    "azuracast_base_url": ("", "AZURACAST_BASE_URL", "text"),
    "azuracast_station_id": ("", "AZURACAST_STATION_ID", "text"),
    "azuracast_api_key": ("", "AZURACAST_API_KEY", "secret"),
    "azuracast_dj_url": ("", "AZURACAST_DJ_URL", "text"),
    "azuracast_dj_password": ("", "AZURACAST_DJ_PASSWORD", "secret"),
}

SECRETS: Final = frozenset(name for name, (_d, _e, kind) in FIELDS.items() if kind == "secret")

_lock = threading.Lock()


def _coerce(name: str, value: Any) -> Any:
    default, _env, kind = FIELDS[name]
    try:
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if kind == "number":
            return float(value)
        return str(value).strip()
    except (TypeError, ValueError):
        return default


def defaults() -> dict[str, Any]:
    """Built-in defaults, with any environment variable applied on top."""
    values: dict[str, Any] = {}
    for name, (default, env_name, _kind) in FIELDS.items():
        raw = os.environ.get(env_name, "").strip() if env_name else ""
        values[name] = _coerce(name, raw) if raw else default
    # Configuring AzuraCast through the environment implies wanting it on.
    if values["azuracast_base_url"] and values["azuracast_api_key"]:
        values["azuracast_enabled"] = True
    return values


def load(conn: sqlite3.Connection) -> dict[str, Any]:
    values = defaults()
    raw = db.get_meta(conn, META_KEY)
    if not raw:
        return values
    try:
        stored = json.loads(raw)
    except (TypeError, ValueError):
        log.warning("stored settings are not valid JSON; using defaults")
        return values
    for name in FIELDS:
        if name in stored:
            values[name] = _coerce(name, stored[name])
    return values


def save(conn: sqlite3.Connection, incoming: dict) -> dict[str, Any]:
    """Merge an update in. Unknown keys are ignored; masked secrets are kept."""
    with _lock:
        values = load(conn)
        for name in FIELDS:
            if name not in incoming:
                continue
            value = incoming[name]
            # The UI never receives a secret, so it cannot send one back. A
            # masked value means "leave this alone" rather than "set it to dots".
            if name in SECRETS and isinstance(value, str) and value.strip() in {MASK, ""}:
                continue
            values[name] = _coerce(name, value)
        db.set_meta(conn, META_KEY, json.dumps(values))
    return values


def public(values: dict[str, Any]) -> dict[str, Any]:
    """The same settings with secrets masked, safe to send to a browser."""
    return {
        name: (MASK if name in SECRETS and value else value)
        for name, value in values.items()
    }
