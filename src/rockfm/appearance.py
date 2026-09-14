"""What each kind of timeline item shows, editable from the dashboard.

Songs carry their own title, artist and cover. Everything else -- adverts,
presenter talk, and stretches we could not place -- has nothing of its own, so
what a player displays for them is a matter of taste. This holds that choice.

Stored as JSON in the meta table rather than as new columns, so an existing
install picks it up on upgrade with no migration.

A blank field means "use what we already know": the programme name and its
artwork for presenter talk, the station name for anything unrecognised. That
way the defaults stay useful and an override is only needed where the built-in
answer is not wanted.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Final

from . import db
from . import strings_es as S

META_KEY: Final = "appearance"
FIELDS: Final = ("title", "artist", "art")

# Gaps in the recording are not shown to anyone, so they need no appearance.
CONFIGURABLE: Final = (S.KIND_PUBLICIDAD, S.KIND_PROGRAMA, S.KIND_DESCONOCIDO)


def defaults(language: str = "es") -> dict[str, dict[str, str]]:
    return {
        S.KIND_PUBLICIDAD: {
            "title": S.text("publicidad.primary", language),
            "artist": S.text("publicidad.secondary", language),
            "art": "",
        },
        # Blank throughout: presenter talk shows the real programme and its
        # artwork, which is better than anything a fixed default could say.
        S.KIND_PROGRAMA: {"title": "", "artist": "", "art": ""},
        S.KIND_DESCONOCIDO: {"title": S.STATION_NAME, "artist": "", "art": ""},
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
