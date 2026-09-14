"""Display strings, Spanish by default.

The station is Spanish and so is everything it says between songs, so adverts
and presenter talk are labelled in Spanish. These are only the built-in
defaults: what each kind actually shows is configurable and lives in the
database, see appearance.py.
"""

from __future__ import annotations

from typing import Final

KIND_CANCION: Final = "cancion"
KIND_PUBLICIDAD: Final = "publicidad"
KIND_PROGRAMA: Final = "programa"
KIND_DESCONOCIDO: Final = "desconocido"

# Four kinds, and every one of them is something we can actually establish: a
# song we identified, audio we have heard before, someone talking, or none of
# the above. Earlier versions also had "noticias" and "sintonia", but nothing
# detected them -- they were guesses from the clock (top of the hour must be
# news; a repeat during the overnight block must be an ident) dressed up as
# findings. Rows written by those versions still render, as desconocido.
ALL_KINDS: Final = (KIND_CANCION, KIND_PUBLICIDAD, KIND_PROGRAMA, KIND_DESCONOCIDO)

STATION_NAME: Final = "RockFM"

ES: Final[dict[str, str]] = {
    "station": STATION_NAME,
    "publicidad.primary": "Publicidad",
    "publicidad.secondary": "Volvemos enseguida",
    "desconocido.primary": STATION_NAME,
    "programa.secondary_fallback": "En directo",
    "now_playing": "Sonando ahora",
    "next": "A continuación",
    "filling": "Rellenando el búfer",
    "filling.ready_in": "listo en",
}

EN: Final[dict[str, str]] = {
    "station": STATION_NAME,
    "publicidad.primary": "Advertisements",
    "publicidad.secondary": "Back shortly",
    "desconocido.primary": STATION_NAME,
    "programa.secondary_fallback": "Live",
    "now_playing": "Now playing",
    "next": "Up next",
    "filling": "Filling the buffer",
    "filling.ready_in": "ready in",
}

TABLES: Final[dict[str, dict[str, str]]] = {"es": ES, "en": EN}


def table(language: str = "es") -> dict[str, str]:
    return TABLES.get(language.lower(), ES)


def text(key: str, language: str = "es") -> str:
    return table(language).get(key, ES.get(key, key))
