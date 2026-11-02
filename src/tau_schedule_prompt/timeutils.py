"""Clock helpers so every module agrees on how time is stored and compared.

Timestamps are persisted as timezone-aware UTC ISO-8601 strings. Cron
expressions, like Pi's ``croner`` behaviour, are evaluated in the machine's
*local* calendar (wall-clock) time, so conversion between the two goes through
the local naive forms in :func:`to_local_naive` / :func:`from_local_naive`.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0)


def to_iso(moment: datetime) -> str:
    """Render ``moment`` as a UTC ISO-8601 string with second precision."""
    if moment.tzinfo is None:
        moment = from_local_naive(moment)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def parse_datetime(value: str | None) -> datetime | None:
    """Parse a stored/user-supplied timestamp; ``None`` when it is not a date.

    A bare ISO timestamp without an offset is interpreted as local wall-clock
    time, which matches how users type ``2026-02-13T15:30``.
    """
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return from_local_naive(parsed)
    return parsed.astimezone(UTC)


def to_local_naive(moment: datetime) -> datetime:
    """Return ``moment`` as a naive local wall-clock timestamp."""
    aware = moment if moment.tzinfo is not None else from_local_naive(moment)
    return aware.astimezone().replace(tzinfo=None)


def from_local_naive(moment: datetime) -> datetime:
    """Attach the machine's local timezone to a naive wall-clock timestamp.

    ``datetime.timestamp()`` resolves DST gaps/overshoots the way the platform
    does, which keeps cron arithmetic consistent with the system clock. A time
    that does not exist locally (spring-forward gap) is pushed forward rather
    than raising, so a schedule can never crash the scheduler loop.
    """
    naive = moment.replace(tzinfo=None)
    try:
        return datetime.fromtimestamp(naive.timestamp(), UTC)
    except (OverflowError, OSError, ValueError):  # pragma: no cover - platform tz
        return naive.replace(tzinfo=UTC)


def format_relative(moment: datetime, *, now: datetime | None = None) -> str:
    """Format ``moment`` as a compact relative label, e.g. ``in 5m``/``2h ago``."""
    reference = now or utc_now()
    diff = (moment - reference).total_seconds()
    absolute = abs(diff)
    seconds = int(absolute)
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24
    if days > 0:
        span = f"{days}d"
    elif hours > 0:
        span = f"{hours}h"
    elif minutes > 0:
        span = f"{minutes}m"
    else:
        span = f"{seconds}s"
    return f"in {span}" if diff > 0 else f"{span} ago"


def format_short(moment: datetime) -> str:
    """Render a timestamp compactly in local time, e.g. ``Feb 13 15:30``."""
    local = to_local_naive(moment)
    return f"{local.strftime('%b')} {local.day} {local:%H:%M}"


def truncate(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` characters, marking the cut with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


__all__ = [
    "format_relative",
    "format_short",
    "from_local_naive",
    "parse_datetime",
    "to_iso",
    "to_local_naive",
    "truncate",
    "utc_now",
]
