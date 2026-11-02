"""Tests for the 6-field cron parser and next-run search."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tau_schedule_prompt.cron import (
    CronParseError,
    describe_cron,
    is_valid_cron,
    parse_cron,
)
from tau_schedule_prompt.timeutils import to_local_naive

LOCAL = to_local_naive(datetime(2026, 2, 13, 15, 30, 0, tzinfo=UTC))


def local_naive(moment: datetime) -> datetime:
    """Return a timestamp's local wall-clock fields for timezone-safe asserts."""
    return to_local_naive(moment)


def test_parses_every_minute_pattern() -> None:
    fields = parse_cron("0 * * * * *")
    assert fields.seconds == frozenset({0})
    assert fields.minutes == frozenset(range(60))
    assert not fields.dom_restricted
    assert not fields.dow_restricted


def test_rejects_five_field_cron_with_guidance() -> None:
    with pytest.raises(CronParseError) as error:
        parse_cron("* * * * *")
    assert "6 fields" in str(error.value)


@pytest.mark.parametrize(
    "expression",
    [
        "0 0 0 30 2 *",  # impossible date, but syntactically fine
        "*/15 * * * * *",
        "0 0 9 * * MON-FRI",
        "0 0 0 1 JAN *",
        "0 0,30 * * * *",
        "0 0 0 * * 7",
    ],
)
def test_accepts_common_shapes(expression: str) -> None:
    assert is_valid_cron(expression)


@pytest.mark.parametrize(
    "expression", ["0 0 * * *", "61 * * * * *", "0 0 * * * Q", "*/0 * * * * *"]
)
def test_rejects_invalid_shapes(expression: str) -> None:
    assert not is_valid_cron(expression)


def test_next_after_every_minute() -> None:
    fields = parse_cron("0 * * * * *")
    nxt = fields.next_after(datetime(2026, 2, 13, 15, 30, 5, tzinfo=UTC))
    assert nxt is not None
    assert local_naive(nxt) == (LOCAL.replace(second=0) + timedelta(minutes=1))


def test_next_after_hourly_and_daily() -> None:
    hourly = parse_cron("0 0 * * * *").next_after(datetime(2026, 2, 13, 15, 30, 0, tzinfo=UTC))
    assert hourly is not None
    assert local_naive(hourly) == (LOCAL.replace(minute=0, second=0) + timedelta(hours=1))

    # Build the pattern from the local clock so the assertion holds in any zone.
    target = LOCAL + timedelta(minutes=45)
    daily = parse_cron(f"0 {target.minute} {target.hour} * * *").next_after(
        datetime(2026, 2, 13, 15, 30, 0, tzinfo=UTC)
    )
    assert daily is not None
    assert local_naive(daily) == target.replace(second=0)


def test_month_rollover() -> None:
    fields = parse_cron("0 0 1 1 * *")  # first day of month at 01:00
    nxt = fields.next_after(datetime(2026, 2, 13, 15, 30, 0, tzinfo=UTC))
    assert nxt is not None
    assert local_naive(nxt) == LOCAL.replace(month=3, day=1, hour=1, minute=0, second=0)


def test_impossible_date_returns_none() -> None:
    fields = parse_cron("0 0 0 30 2 *")
    assert fields.next_after(datetime(2026, 2, 13, tzinfo=UTC)) is None


def test_weekday_range_selects_monday() -> None:
    fields = parse_cron("0 0 9 * * 1-5")  # 9am Mon-Fri (local)
    friday_evening = datetime(2026, 2, 13, 20, 0, 0, tzinfo=UTC)
    nxt = fields.next_after(friday_evening)
    assert nxt is not None
    assert local_naive(nxt).weekday() == 0  # Monday
    assert local_naive(nxt).hour == 9


def test_dom_and_dow_are_ored_when_both_restricted() -> None:
    fields = parse_cron("0 0 0 1 * 1")  # the 1st OR any Monday
    days = set()
    cursor = datetime(2026, 3, 1, tzinfo=UTC)
    for _ in range(6):
        nxt = fields.next_after(cursor)
        assert nxt is not None
        local = local_naive(nxt)
        days.add((local.day, local.weekday()))
        cursor = nxt
    assert any(day == 1 for day, _ in days)
    assert any(weekday == 0 for _, weekday in days)


def test_step_and_list_fields() -> None:
    fields = parse_cron("0 */15 9,17 * * *")
    assert fields.minutes == frozenset({0, 15, 30, 45})
    assert fields.hours == frozenset({9, 17})


def test_describe_cron_covers_presets_and_falls_back() -> None:
    assert describe_cron("0 * * * * *") == "every minute"
    assert describe_cron("0 */5 * * * *") == "every 5 min"
    assert describe_cron("0 0 * * * *") == "every hour"
    assert describe_cron("0 0 0 * * *") == "daily"
    assert describe_cron("0 0 */4 * * *") == "every 4h"
    assert describe_cron("0 0 7 * * *") == "daily at 07:00"
    assert describe_cron("0 0 0 15 * *") == "0 0 0 15 * *"
