"""Turn a timeline row into what a listener actually sees."""

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


def present(
    row: dict[str, Any],
    language: str = "es",
    appearance: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str, str | None]:
    """Return (primary, secondary, art) for a timeline row.

    Songs speak for themselves. For everything else the operator's configured
    appearance wins, and a blank setting falls back to whatever we do know --
    the programme name and artwork, or the station name.
    """
    kind = row.get("kind") or S.KIND_DESCONOCIDO
    if kind not in S.ALL_KINDS:
        kind = S.KIND_DESCONOCIDO  # rows from versions that had more kinds
    art = row.get("art_url")

    if kind == S.KIND_CANCION:
        return (
            song_primary(row.get("artist"), row.get("title")),
            song_secondary(row.get("album"), row.get("year")),
            art,
        )

    show = (row.get("show_title") or "").strip()
    presenters = (row.get("show_lead") or "").strip()
    if kind == S.KIND_PROGRAMA:
        # Presenter talk names the real programme; nothing fixed beats that.
        fallback = (
            show or S.STATION_NAME,
            presenters or S.text("programa.secondary_fallback", language),
        )
    elif kind == S.KIND_PUBLICIDAD:
        fallback = (
            S.text("publicidad.primary", language),
            S.text("publicidad.secondary", language),
        )
    else:
        fallback = (S.text("desconocido.primary", language) or S.STATION_NAME, show)

    configured = (appearance or {}).get(kind, {})
    primary = configured.get("title") or fallback[0]
    secondary = configured.get("artist") or fallback[1]
    return primary, secondary, configured.get("art") or art


def render(
    row: dict[str, Any],
    language: str = "es",
    appearance: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    primary, secondary, _ = present(row, language, appearance)
    return primary, secondary


def stream_title(
    row: dict[str, Any],
    language: str = "es",
    appearance: dict[str, dict[str, str]] | None = None,
) -> str:
    """The single line a plain HLS or Icecast player shows.

    Players expect "Artist - Title", so songs use exactly that rather than the
    two-line form built for our own UI.
    """
    if not row:
        return S.STATION_NAME
    if row.get("kind") == S.KIND_CANCION:
        artist = (row.get("artist") or "").strip()
        title = (row.get("title") or "").strip()
        if artist and title:
            return f"{artist} - {title}"
        return title or artist or S.STATION_NAME
    primary, secondary = render(row, language, appearance)
    if secondary and secondary != primary:
        return f"{primary} - {secondary}"
    return primary
