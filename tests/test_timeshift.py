from datetime import datetime, timedelta, timezone

import time_machine

from rockfm.config import Config
from rockfm.timeshift import DelayController, MAX_DEFERRAL, target_delay_seconds

HOUR = 3600


def cfg(**kw):
    return Config(**kw)


def at(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


def test_normal_offset_is_six_hours():
    assert target_delay_seconds(cfg(), at("2026-09-13T20:00:00")) == 6 * HOUR


def test_autumn_dst_window_is_five_hours():
    # EU falls back 2026-10-25, the US not until 2026-11-01.
    assert target_delay_seconds(cfg(), at("2026-10-24T12:00:00")) == 6 * HOUR
    assert target_delay_seconds(cfg(), at("2026-10-26T12:00:00")) == 5 * HOUR
    assert target_delay_seconds(cfg(), at("2026-11-02T12:00:00")) == 6 * HOUR


def test_spring_dst_window_is_five_hours():
    # The US springs forward 2027-03-14, the EU not until 2027-03-28.
    assert target_delay_seconds(cfg(), at("2027-03-13T12:00:00")) == 6 * HOUR
    assert target_delay_seconds(cfg(), at("2027-03-20T12:00:00")) == 5 * HOUR
    assert target_delay_seconds(cfg(), at("2027-03-29T12:00:00")) == 6 * HOUR


def test_override_wins():
    assert target_delay_seconds(cfg(delay_seconds_override=90), at("2026-09-13T20:00:00")) == 90


def test_change_is_deferred_until_a_safe_boundary():
    controller = DelayController(cfg(), at("2026-10-24T12:00:00"))
    assert controller.current == 6 * HOUR

    mid_song = at("2026-10-26T12:00:00")
    assert controller.consider(mid_song, safe=False) == 6 * HOUR
    assert controller.pending is not None and controller.pending.delay == 5 * HOUR

    assert controller.consider(mid_song + timedelta(seconds=30), safe=True) == 5 * HOUR
    assert controller.pending is None


def test_change_applies_anyway_once_overdue():
    controller = DelayController(cfg(), at("2026-10-24T12:00:00"))
    start = at("2026-10-26T12:00:00")
    controller.consider(start, safe=False)
    assert controller.current == 6 * HOUR
    assert controller.consider(start + MAX_DEFERRAL, safe=False) == 5 * HOUR


def test_offset_returning_to_normal_replays_an_hour():
    controller = DelayController(cfg(), at("2026-10-26T12:00:00"))
    assert controller.current == 5 * HOUR
    assert controller.consider(at("2026-11-02T12:00:00"), safe=True) == 6 * HOUR
