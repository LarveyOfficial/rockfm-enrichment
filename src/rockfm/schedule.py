"""What is scheduled to be on air, in the station's own timezone.

Used to name non-music stretches ("El Pirata y su banda" rather than a blank),
and to supply one classification prior: the overnight block is billed as an
hour of rock without breaks, so a repeat during it is more likely a programme
promo than an advert.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

from .rockfm_api import Programme, RockFmApi

log = logging.getLogger("rockfm.schedule")

REFRESH_SECONDS = 6 * 3600
NO_BREAK_MARKERS = ("sin pausa", "sin publicidad")


class Schedule:
    """Caches the weekly schedule and answers 'what is on air at this moment'."""

    def __init__(self, api: RockFmApi | None = None) -> None:
        self._api = api or RockFmApi()
        self._lock = threading.Lock()
        self._days: dict[int, list[Programme]] = {}
        self._fallback: Programme | None = None
        self._loaded_at = 0.0

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            if not force and self._days and time.monotonic() - self._loaded_at < REFRESH_SECONDS:
                return
            try:
                self._days = self._api.schedule()
                self._fallback = self._api.fallback_programme()
                self._loaded_at = time.monotonic()
                log.info(
                    "schedule loaded: %d programmes across the week",
                    sum(len(day) for day in self._days.values()),
                )
            except Exception as exc:
                log.warning("schedule refresh failed: %s", exc)

    def at(self, moment: datetime) -> Programme | None:
        """`moment` must already be in the station's timezone."""
        self.refresh()
        minute = moment.hour * 60 + moment.minute
        candidates = [
            programme
            for programme in self._days.get(moment.weekday(), [])
            if programme.start_minute <= minute < programme.end_minute
        ]
        if not candidates:
            return self._fallback
        # Longer entries are the main show; a shorter overlapping one is more specific.
        candidates.sort(key=lambda p: p.end_minute - p.start_minute)
        return candidates[0]

    def is_no_break_block(self, moment: datetime) -> bool:
        """True during blocks the station advertises as running without adverts."""
        programme = self.at(moment)
        if programme is None:
            return False
        haystack = f"{programme.lead} {programme.description}".casefold()
        return any(marker in haystack for marker in NO_BREAK_MARKERS)
