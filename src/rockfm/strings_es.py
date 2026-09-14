"""Display strings, Spanish by default.

The station is Spanish and so is everything it says between songs, so ads, DJ
talk and news are labelled in Spanish. Strings live here rather than inline so
another language can be dropped in without touching classification logic.
"""

from __future__ import annotations

from typing import Final

KIND_CANCION: Final = "cancion"
KIND_PUBLICIDAD: Final = "publicidad"
KIND_PROGRAMA: Final = "programa"
KIND_NOTICIAS: Final = "noticias"
KIND_SINTONIA: Final = "sintonia"
KIND_DESCONOCIDO: Final = "desconocido"

ALL_KINDS: Final = (
    KIND_CANCION,
    KIND_PUBLICIDAD,
    KIND_PROGRAMA,
    KIND_NOTICIAS,
    KIND_SINTONIA,
    KIND_DESCONOCIDO,
)

STATION_NAME: Final = "RockFM"

ES: Final[dict[str, str]] = {
    "station": STATION_NAME,
    "publicidad.primary": "Publicidad",
    "publicidad.secondary": "Volvemos enseguida",
    "noticias.primary": "Noticias",
    "noticias.secondary": "RockFM Noticias",
    "sintonia.primary": STATION_NAME,
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
    "noticias.primary": "News",
    "noticias.secondary": "RockFM News",
    "sintonia.primary": STATION_NAME,
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
