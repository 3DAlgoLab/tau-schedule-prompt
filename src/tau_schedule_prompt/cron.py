"""A dependency-free 6-field cron parser with next-run calculation.

Pi's extension leans on the ``croner`` npm package; this module keeps the same
user-facing contract (6 fields, seconds first, ``* */n a-b a,b`` forms plus
three-letter month/weekday names) without adding a third-party dependency to
the extension. Field semantics follow Vixie cron, including the classic rule
that a restricted day-of-month *and* day-of-week are OR-ed together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .timeutils import from_local_naive, to_local_naive

CRON_FIELDS = ("second", "minute", "hour", "day of month", "month", "day of week")
FIELD_RANGES: tuple[tuple[int, int], ...] = (
    (0, 59),
    (0, 59),
    (0, 23),
    (1, 31),
    (1, 12),
    (0, 6),
)
MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
DOW_NAMES = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")

_MAX_YEAR_SPAN = 5


class CronParseError(ValueError):
    """Raised when a schedule is not a valid 6-field cron expression."""


@dataclass(frozen=True, slots=True)
class CronFields:
    """Allowed values per cron field, ready for next-run search."""

    seconds: frozenset[int]
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool

    def matches_day(self, local: datetime) -> bool:
        """Apply the Vixie cron day rule for a naive local date."""
        dom_ok = local.day in self.days_of_month
        # Python's weekday() is Monday-first; cron's is Sunday-first (0-6, 7==0).
        cron_dow = (local.weekday() + 1) % 7
        dow_ok = cron_dow in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        if self.dom_restricted:
            return dom_ok
        if self.dow_restricted:
            return dow_ok
        return True

    def next_after(self, after: datetime) -> datetime | None:
        """Return the first firing strictly after ``after``, or ``None``.

        ``None`` means "no match within the next five years", which covers
        impossible dates such as ``0 0 0 30 2 * *``.
        """
        local = to_local_naive(after).replace(microsecond=0) + timedelta(seconds=1)
        horizon = local.year + _MAX_YEAR_SPAN
        while local.year <= horizon:
            if local.month not in self.months:
                local = _start_of_next_month(local)
                continue
            if not self.matches_day(local):
                local = _start_of_next_day(local)
                continue
            if local.hour not in self.hours:
                nxt = _next_allowed(self.hours, local.hour)
                local = (
                    local.replace(hour=nxt, minute=0, second=0)
                    if nxt is not None
                    else _start_of_next_day(local)
                )
                continue
            if local.minute not in self.minutes:
                nxt = _next_allowed(self.minutes, local.minute)
                local = (
                    local.replace(minute=nxt, second=0)
                    if nxt is not None
                    else (local.replace(minute=0, second=0) + timedelta(hours=1))
                )
                continue
            if local.second not in self.seconds:
                nxt = _next_allowed(self.seconds, local.second)
                local = (
                    local.replace(second=nxt)
                    if nxt is not None
                    else (local.replace(second=0) + timedelta(minutes=1))
                )
                continue
            return from_local_naive(local)
        return None


def parse_cron(expression: str) -> CronFields:
    """Parse a 6-field cron expression into :class:`CronFields`.

    Raises :class:`CronParseError` with an actionable message, mirroring the
    guidance Pi's extension gives the model for 5-field input.
    """
    fields = expression.split()
    if len(fields) == 5:
        raise CronParseError(
            "Cron expression must have 6 fields (second minute hour dom month dow), got 5. "
            "Example: '0 * * * * *' for every minute"
        )
    if len(fields) != 6:
        raise CronParseError(
            "Cron expression must have 6 fields (second minute hour dom month dow), "
            f"got {len(fields)}. Example: '0 * * * * *' for every minute"
        )

    values: list[frozenset[int]] = []
    restricted: list[bool] = []
    for index, (field, (low, high)) in enumerate(zip(fields, FIELD_RANGES, strict=True)):
        allowed = _parse_field(field, low, high, field_kind=index)
        values.append(allowed)
        restricted.append(field.strip() != "*")

    return CronFields(
        seconds=values[0],
        minutes=values[1],
        hours=values[2],
        days_of_month=values[3],
        months=values[4],
        days_of_week=values[5],
        dom_restricted=restricted[3],
        dow_restricted=restricted[5],
    )


def is_valid_cron(expression: str) -> bool:
    """Return whether ``expression`` parses as a 6-field cron schedule."""
    try:
        parse_cron(expression)
    except CronParseError:
        return False
    return True


def _parse_field(field: str, low: int, high: int, *, field_kind: int) -> frozenset[int]:
    allowed: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise CronParseError(f"Empty cron part in {CRON_FIELDS[field_kind]!r} field")
        body, separator, step_text = part.partition("/")
        if separator and not step_text.isdigit():
            raise CronParseError(f"Invalid step {step_text!r} in {CRON_FIELDS[field_kind]!r} field")
        step = int(step_text) if separator else 1
        if step < 1:
            raise CronParseError(f"Step must be >= 1 in {CRON_FIELDS[field_kind]!r} field")
        if body == "*":
            start, end = low, high
        elif "-" in body.lstrip("-"):
            start_text, _, end_text = body.partition("-")
            start = _parse_value(start_text, low, high, field_kind)
            end = _parse_value(end_text, low, high, field_kind)
        else:
            start = _parse_value(body, low, high, field_kind)
            end = high if separator else start
        if start > end:
            # Wrap-around ranges such as Friday(5)-Tuesday(2) are valid cron.
            allowed.update(range(start, high + 1))
            allowed.update(range(low, end + 1))
            continue
        allowed.update(range(start, end + 1, step))
    if not allowed:
        raise CronParseError(f"No values matched in {CRON_FIELDS[field_kind]!r} field")
    return frozenset(allowed)


def _parse_value(text: str, low: int, high: int, field_kind: int) -> int:
    name_tables: dict[int, tuple[tuple[str, int], ...]] = {
        4: tuple((name, index + 1) for index, name in enumerate(MONTH_NAMES)),
        5: tuple((name, index) for index, name in enumerate(DOW_NAMES)),
    }
    candidate = text.strip().lower()
    for name, value in name_tables.get(field_kind, ()):
        if candidate == name:
            return value
    if not candidate.isdigit():
        raise CronParseError(
            f"Invalid value {text!r} in {CRON_FIELDS[field_kind]!r} field (expected {low}-{high})"
        )
    value = int(candidate)
    if field_kind == 5 and value == 7:  # cron accepts 7 as Sunday
        return 0
    if not low <= value <= high:
        raise CronParseError(
            f"Value {text!r} out of range {low}-{high} in {CRON_FIELDS[field_kind]!r} field"
        )
    return value


def _next_allowed(allowed: frozenset[int], current: int) -> int | None:
    candidates = [value for value in allowed if value > current]
    return min(candidates) if candidates else None


def _start_of_next_day(local: datetime) -> datetime:
    return local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _start_of_next_month(local: datetime) -> datetime:
    first = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return (first + timedelta(days=32)).replace(day=1)


_HUMANIZED_CRON: dict[str, str] = {
    "* * * * * *": "every second",
    "0 * * * * *": "every minute",
    "0 */5 * * * *": "every 5 min",
    "0 */10 * * * *": "every 10 min",
    "0 */15 * * * *": "every 15 min",
    "0 */30 * * * *": "every 30 min",
    "0 0 * * * *": "every hour",
    "0 0 */2 * * *": "every 2 hours",
    "0 0 */3 * * *": "every 3 hours",
    "0 0 */6 * * *": "every 6 hours",
    "0 0 0 * * *": "daily",
    "0 0 0 * * 0": "weekly",
    "0 0 0 1 * *": "monthly",
    "0 0 9 * * 1-5": "9am weekdays",
    "0 0 0 * * 1-5": "weekends off",
    "0 0 0 * * 0,6": "weekends",
}

_EVERY_MINUTES = re.compile(r"^0 \*/(\d+) \* \* \* \*$")
_EVERY_HOURS = re.compile(r"^0 0 \*/(\d+) \* \* \*$")
_DAILY_AT = re.compile(r"^0 0 (\d+) \* \* \*$")


def describe_cron(expression: str) -> str:
    """Humanise common cron patterns; fall back to the raw text.

    Never guesses a description for a pattern it does not recognise, so a
    displayed schedule is always either exact or the literal expression.
    """
    normalized = " ".join(expression.split())
    preset = _HUMANIZED_CRON.get(normalized)
    if preset is not None:
        return preset
    if (match := _EVERY_MINUTES.match(normalized)) is not None:
        return f"every {match.group(1)} min"
    if (match := _EVERY_HOURS.match(normalized)) is not None:
        return f"every {match.group(1)}h"
    if (match := _DAILY_AT.match(normalized)) is not None:
        return f"daily at {int(match.group(1)):02d}:00"
    return normalized


__all__ = ["CronFields", "CronParseError", "describe_cron", "is_valid_cron", "parse_cron"]
