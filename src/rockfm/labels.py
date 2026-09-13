"""Turn a timeline row into the two lines a player shows."""

from __future__ import annotations

from typing import Any

from . import strings_es as S


def song_primary(artist: str | None, title: str | None) -> str:
    artist = (artist or "").strip()
    title = (title or "").strip()
    if artist and title:
        return f"{artist} — {title}"
    return title or artist or S.STATION_NAME


def song_secondary(album: str | None, year: int | None) -> str:
    parts = [part for part in ((album or "").strip(), str(year) if year else "") if part]
    return " · ".join(parts)


def render(row: dict[str, Any], language: str = "es") -> tuple[str, str]:
    """Return (primary, secondary) display lines for a timeline row."""
    kind = row.get("kind") or S.KIND_DESCONOCIDO
    show = (row.get("show_title") or "").strip()
    presenters = (row.get("show_lead") or "").strip()

    if kind == S.KIND_CANCION:
        return (
            song_primary(row.get("artist"), row.get("title")),
            song_secondary(row.get("album"), row.get("year")),
        )
    if kind == S.KIND_PUBLICIDAD:
        return (
            S.text("publicidad.primary", language),
            S.text("publicidad.secondary", language),
        )
    if kind == S.KIND_NOTICIAS:
        return (
            S.text("noticias.primary", language),
            show or S.text("noticias.secondary", language),
        )
    if kind == S.KIND_PROGRAMA:
        return (
            show or S.STATION_NAME,
            presenters or S.text("programa.secondary_fallback", language),
        )
    # sintonia / desconocido: never guess, fall back to the station and show name.
    return (S.text(f"{kind}.primary", language) or S.STATION_NAME, show)
