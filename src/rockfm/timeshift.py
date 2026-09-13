"""Wall-clock time shifting between the source and local timezones.

The delay is not a constant. Spain and the United States change DST on
different dates, so `Europe/Madrid` -> `America/New_York` is 6 h for most of the
year but 5 h during two windows (e.g. 2026-10-25 -> 2026-11-01 and
2027-03-14 -> 2027-03-28). Matching Madrid's wall clock means recomputing the
offset rather than hardcoding six hours.

When the offset changes, playout has to jump an hour forwards (skipping audio)
or backwards (replaying it). Doing that mid-song is jarring, so the change is
deferred until playout reports a safe boundary -- with a hard deadline so a long
music block cannot postpone it forever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .config import Config

log = logging.getLogger("rockfm.timeshift")

# Never defer a DST correction longer than this.
MAX_DEFERRAL = timedelta(minutes=30)


def target_delay_seconds(config: Config, now: datetime | None = None) -> int:
    """Seconds of delay needed so source wall-clock time matches local wall-clock time."""
    if config.delay_seconds_override is not None:
        return config.delay_seconds_override
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    source_offset = moment.astimezone(config.source_tz).utcoffset() or timedelta()
    local_offset = moment.astimezone(config.local_tz).utcoffset() or timedelta()
    return int((source_offset - local_offset).total_seconds())


def source_wallclock(config: Config, now: datetime | None = None) -> datetime:
    """The source-timezone wall clock currently being played out."""
    moment = now or datetime.now(UTC)
    return (moment - timedelta(seconds=target_delay_seconds(config, moment))).astimezone(
        config.source_tz
    )


def playout_position(config: Config, delay_seconds: int, now: datetime | None = None) -> int:
    """Epoch ms in the recorded buffer that should be airing right now."""
    moment = now or datetime.now(UTC)
    return int((moment.timestamp() - delay_seconds) * 1000)


@dataclass
class PendingChange:
    delay: int
    since: datetime


class DelayController:
    """Tracks the applied delay and defers DST changes to a safe boundary."""

    def __init__(self, config: Config, now: datetime | None = None) -> None:
        self.config = config
        moment = now or datetime.now(UTC)
        self.current: int = target_delay_seconds(config, moment)
        self.pending: PendingChange | None = None
        log.info("initial delay: %d s (%.1f h)", self.current, self.current / 3600)

    def consider(self, now: datetime | None = None, *, safe: bool = False) -> int:
        """Recompute the target delay and apply it when allowed.

        `safe` should be True when playout is not in the middle of a song.
        """
        moment = now or datetime.now(UTC)
        target = target_delay_seconds(self.config, moment)

        if target == self.current:
            self.pending = None
            return self.current

        if self.pending is None or self.pending.delay != target:
            self.pending = PendingChange(delay=target, since=moment)
            log.info(
                "delay change pending: %d s -> %d s (%.1f h -> %.1f h)",
                self.current,
                target,
                self.current / 3600,
                target / 3600,
            )

        overdue = moment - self.pending.since >= MAX_DEFERRAL
        if safe or overdue:
            direction = "skipping" if target < self.current else "replaying"
            log.info(
                "applying delay change %d s -> %d s (%s %.0f min of audio, %s)",
                self.current,
                target,
                direction,
                abs(target - self.current) / 60,
                "safe boundary" if safe else "deferral deadline",
            )
            self.current = target
            self.pending = None

        return self.current
