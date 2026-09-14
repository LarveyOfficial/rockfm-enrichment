"""What each kind of timeline item shows, editable from the dashboard.

Only adverts need this. Songs carry their own title, artist and cover;
presenter talk and unplaced audio already take the real programme name,
presenters and artwork from RockFM's own schedule, which beats anything a fixed
default could say. An advert break is the one thing with nothing of its own.

Stored as JSON in the meta table rather than as new columns, so an existing
install picks it up on upgrade with no migration.

A blank field falls back to the built-in Spanish default.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Final

from . import db
from . import strings_es as S

META_KEY: Final = "appearance"
FIELDS: Final = ("title", "artist", "art")

# Gaps are never shown to anyone, songs describe themselves, and the rest take
# their name and artwork from the station's schedule. Only adverts are left.
CONFIGURABLE: Final = (S.KIND_PUBLICIDAD,)


def defaults(language: str = "es") -> dict[str, dict[str, str]]:
    return {
        S.KIND_PUBLICIDAD: {
            "title": S.text("publicidad.primary", language),
            "artist": S.text("publicidad.secondary", language),
            "art": "",
        },
    }


def load(conn: sqlite3.Connection, language: str = "es") -> dict[str, dict[str, str]]:
    settings = defaults(language)
    raw = db.get_meta(conn, META_KEY)
    if not raw:
        return settings
    try:
        stored = json.loads(raw)
    except (TypeError, ValueError):
        return settings
    for kind in CONFIGURABLE:
        for field in FIELDS:
            value = (stored.get(kind) or {}).get(field)
            if isinstance(value, str):
                settings[kind][field] = value.strip()
    return settings


def save(
    conn: sqlite3.Connection, incoming: dict, language: str = "es"
) -> dict[str, dict[str, str]]:
    """Merge an update in and persist it. Unknown kinds and fields are ignored."""
    settings = load(conn, language)
    for kind in CONFIGURABLE:
        for field in FIELDS:
            value = (incoming.get(kind) or {}).get(field)
            if isinstance(value, str):
                settings[kind][field] = value.strip()
    db.set_meta(conn, META_KEY, json.dumps(settings))
    return settings
