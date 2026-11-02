"""Schedule validation, next-run maths, and the asyncio scheduling loop.

Pi's extension leans on ``croner`` timers plus ``setInterval``/``setTimeout``.
Tau runs inside one asyncio event loop shared with the session, so the port
uses a single wake-up loop that computes the earliest pending fire across all
loaded jobs. That removes a whole class of bugs the Pi version documents (leaked
timers on every ``session_start``, stale closure snapshots advancing counters)
and keeps relative-time labels fresh because the same loop caps its sleep at the
widget refresh interval.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from .cron import CronFields, CronParseError, describe_cron, parse_cron
from .storage import JobStorage
from .timeutils import format_short, parse_datetime, to_iso, utc_now
from .types import JobType, ScheduleJob

RELATIVE_RE = re.compile(r"^\+(\d+)(s|m|h|d)$")
INTERVAL_RE = re.compile(r"^(\d+)(s|m|h|d)$")
UNIT_MS: dict[str, int] = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}

MIN_ONCE_LEAD_MS = 5_000
MAX_TICK_SECONDS = 30.0
MIN_TICK_SECONDS = 0.25

_CRON_CACHE: dict[str, CronFields] = {}


class ScheduleError(ValueError):
    """Raised for an unusable schedule; the message is model- and user-facing."""


@dataclass(frozen=True, slots=True)
class ResolvedSchedule:
    """A validated schedule: the resolved form to persist plus interval ms."""

    schedule: str
    interval_ms: int | None = None


def parse_relative_delta(value: str, *, now: datetime | None = None) -> datetime | None:
    """Resolve ``+10s``/``+5m``/``+1h``/``+2d`` into an absolute timestamp."""
    match = RELATIVE_RE.match(value.strip())
    if match is None:
        return None
    reference = now or utc_now()
    return reference + timedelta(milliseconds=int(match.group(1)) * UNIT_MS[match.group(2)])


def parse_interval_ms(value: str) -> int | None:
    """Parse an interval string such as ``5m`` or ``1h`` into milliseconds."""
    match = INTERVAL_RE.match(value.strip())
    if match is None:
        return None
    return int(match.group(1)) * UNIT_MS[match.group(2)]


def validate_schedule(
    job_type: JobType,
    schedule: str,
    *,
    now: datetime | None = None,
) -> ResolvedSchedule:
    """Validate and resolve one schedule, raising :class:`ScheduleError`.

    ``once`` accepts relative time and ISO timestamps and is stored as an ISO
    string; ``interval`` stores the raw duration; ``cron`` stores the expression.
    """
    reference = now or utc_now()
    text = schedule.strip()
    if not text:
        raise ScheduleError("'schedule' must not be empty")

    if job_type == "interval":
        interval_ms = parse_interval_ms(text)
        if interval_ms is None:
            raise ScheduleError(
                f"Invalid interval format: {schedule}. Use a format like '5m', '1h', '30s'"
            )
        return ResolvedSchedule(schedule=text, interval_ms=interval_ms)

    if job_type == "once":
        relative = parse_relative_delta(text, now=reference)
        if relative is not None:
            return ResolvedSchedule(schedule=to_iso(relative))
        parsed = parse_datetime(text)
        if parsed is None:
            raise ScheduleError(
                f"Invalid timestamp: {schedule}. Use an ISO timestamp or relative time "
                "like '+10s', '+5m'"
            )
        delay_ms = (parsed - reference).total_seconds() * 1000
        if delay_ms < 0:
            raise ScheduleError(
                f"Timestamp is in the past: {to_iso(parsed)}. Current time: {to_iso(reference)}"
            )
        if delay_ms < MIN_ONCE_LEAD_MS:
            seconds = max(1, round(delay_ms / 1000))
            raise ScheduleError(
                f"Timestamp is too soon ({seconds}s). For delays under 5s use relative time "
                f"like '+{seconds}s', or schedule at least 5s in the future."
            )
        return ResolvedSchedule(schedule=to_iso(parsed))

    try:
        parse_cron(text)
    except CronParseError as exc:
        # Helpful redirects: an interval or relative string typed with
        # type='cron' is the most common model mistake.
        if parse_interval_ms(text) is not None:
            raise ScheduleError(
                f"'{text}' is an interval, not a cron expression. Pass type='interval' for "
                "recurring runs at that cadence, or type='once' with a leading '+' for one "
                "delayed run."
            ) from exc
        if RELATIVE_RE.match(text) is not None:
            raise ScheduleError(
                f"'{text}' is a one-shot delay. Pass type='once' with schedule='{text}'."
            ) from exc
        raise ScheduleError(f"Invalid cron expression: {exc}") from exc
    return ResolvedSchedule(schedule=" ".join(text.split()))


def describe_schedule(job_type: JobType, schedule: str) -> str:
    """Render a resolved schedule as a short human-readable phrase."""
    if job_type == "interval":
        return f"every {schedule}"
    if job_type == "once":
        parsed = parse_datetime(schedule)
        return format_short(parsed) if parsed is not None else schedule
    return describe_cron(schedule)


def next_run_at(job: ScheduleJob, now: datetime) -> datetime | None:
    """Return the job's next firing strictly after ``now`` (``None``: never).

    This is the *display* view: it answers "when will this run next?" from the
    caller's instant, so a stale interval job skips ahead instead of reporting a
    past time. :func:`pending_slot` is the scheduling view used by the loop.
    """
    if not job.enabled:
        return None
    if job.type == "once":
        parsed = parse_datetime(job.schedule)
        return parsed if parsed is not None and parsed > now else None
    if job.type == "interval":
        step = _interval_step(job)
        if step is None:
            return None
        base = parse_datetime(job.last_run) or parse_datetime(job.created_at) or now
        candidate = base + step
        # Never burst-fire missed intervals after a long pause (laptop sleep):
        # skip ahead to the first slot in the future.
        while candidate <= now:
            candidate += step
        return candidate
    return _parse_cached(job.schedule).next_after(now)


def pending_slot(job: ScheduleJob, now: datetime, *, watermark: datetime) -> datetime | None:
    """Return the earliest slot at or before ``now`` this session has not run.

    ``watermark`` is the instant this scheduler started watching. Slots before it
    are deliberately ignored: Tau only fires while a session is open in the
    directory, so a slot that passed while every session was closed stays
    un-fired instead of firing the moment a session boots (matching Pi's
    behaviour of arming croner from "now"). Because a fire records its actual
    run time as ``last_run``, a job can never catch up in a burst.
    """
    if not job.enabled:
        return None
    if job.type == "once":
        if job.last_run is not None:
            return None
        scheduled = parse_datetime(job.schedule)
        if scheduled is None or scheduled > now or scheduled < watermark:
            return None
        return scheduled
    reference = max(parse_datetime(job.last_run) or watermark, watermark)
    if job.type == "interval":
        step = _interval_step(job)
        if step is None:
            return None
        candidate = reference + step
        return candidate if candidate <= now else None
    upcoming = _parse_cached(job.schedule).next_after(reference)
    return upcoming if upcoming is not None and upcoming <= now else None


def _interval_step(job: ScheduleJob) -> timedelta | None:
    """Return a job's interval as a timedelta, tolerating a missing field."""
    if not job.interval_ms:
        return None
    return timedelta(milliseconds=job.interval_ms)


def is_loaded_for(job: ScheduleJob, session_id: str | None) -> bool:
    """Return whether this session should load ``job``.

    Jobs keep their ``session`` binding so two Tau sessions in one directory do
    not double-fire. An unbound job (hand-edited to drop the key) is shared.
    """
    return not job.session or job.session == session_id


def _parse_cached(expression: str) -> CronFields:
    key = " ".join(expression.split())
    cached = _CRON_CACHE.get(key)
    if cached is None:
        cached = parse_cron(key)
        _CRON_CACHE[key] = cached
    return cached


class Scheduler:
    """Drives one asyncio loop that fires every job this session owns."""

    def __init__(
        self,
        storage: JobStorage,
        *,
        session_id: str | None,
        fire: Callable[[ScheduleJob], Awaitable[None]],
        on_change: Callable[[], None],
        clock: Callable[[], datetime] = utc_now,
        watermark: datetime | None = None,
    ) -> None:
        self._storage = storage
        self._session_id = session_id
        self._fire = fire
        self._on_change = on_change
        self._now = clock
        self._watermark = watermark
        self._task: asyncio.Task[None] | None = None
        self._fire_tasks: set[asyncio.Task[None]] = set()
        self._running_jobs: set[str] = set()
        self._wake = asyncio.Event()
        self._stopped = False

    def start(self) -> None:
        """Clear stale ``running`` markers left by an interrupted run, then loop.

        The watermark marks "now" for scheduling purposes: slots that passed
        while no session was open are not retroactively fired.
        """
        self._stopped = False
        if self._watermark is None:
            self._watermark = self._now()
        for job in self.loaded_jobs():
            if job.last_status == "running":
                self._storage.update(job.id, {"last_status": None})
        if self._task is None or self._task.done():
            self._wake.clear()
            self._task = asyncio.create_task(self._loop(), name="tau-schedule-prompt")
        self._on_change()

    async def stop(self) -> None:
        """Cancel the loop and any in-flight fire so nothing outlives the session."""
        self._stopped = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for task in list(self._fire_tasks):
            task.cancel()
        if self._fire_tasks:
            await asyncio.gather(*self._fire_tasks, return_exceptions=True)
        self._fire_tasks.clear()
        self._running_jobs.clear()

    def wake(self) -> None:
        """Re-evaluate due jobs immediately after a tool or UI change."""
        self._wake.set()

    def loaded_jobs(self) -> list[ScheduleJob]:
        """Jobs this session is responsible for (own session or unbound)."""
        return [job for job in self._storage.all() if is_loaded_for(job, self._session_id)]

    def next_run(self, job: ScheduleJob, *, now: datetime | None = None) -> datetime | None:
        """Return ``job``'s next firing (used by the widget and tool output)."""
        return next_run_at(job, now or self._now())

    def job_state(self, job: ScheduleJob) -> str:
        """Return a one-word status label for ``job`` for display purposes."""
        if not job.enabled:
            return "disabled"
        if job.id in self._running_jobs:
            return "running"
        return job.last_status or "pending"

    async def _loop(self) -> None:
        while not self._stopped:
            self._wake.clear()
            delay = MAX_TICK_SECONDS
            try:
                delay = await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad job must not kill the loop
                print(f"tau-schedule-prompt: scheduler tick failed: {exc}", file=sys.stderr)
            # Repaint on every wake, not only on state changes: the sleep is capped at
            # MAX_TICK_SECONDS, so the widget's relative labels ("in 5m") stay honest.
            self._on_change()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=delay)

    async def _tick(self) -> float:
        """Fire everything due, then return how long the loop may sleep.

        The sleep is capped at :data:`MAX_TICK_SECONDS` so the widget's relative
        labels ("in 5m") stay honest even when nothing fires for hours.
        """
        now = self._now()
        watermark = self._watermark or now
        delay = MAX_TICK_SECONDS
        fired = False
        for job in self.loaded_jobs():
            due = pending_slot(job, now, watermark=watermark)
            if due is None:
                self._retire_past_once(job, now, watermark=watermark)
                upcoming = next_run_at(job, now)
                if upcoming is not None and upcoming > now:
                    delay = min(delay, (upcoming - now).total_seconds())
                continue
            if job.id in self._running_jobs:
                continue
            self._spawn(job)
            fired = True
        if fired:
            self._on_change()
            return MIN_TICK_SECONDS
        return min(MAX_TICK_SECONDS, max(MIN_TICK_SECONDS, delay))

    def _retire_past_once(self, job: ScheduleJob, now: datetime, *, watermark: datetime) -> None:
        """Disable an enabled one-shot whose moment already passed.

        A ``once`` job can go stale while no Tau session is open (jobs only fire
        while a session runs in the directory). Rather than silently never
        firing, it is marked disabled/errored so the widget and tool say why.
        """
        if job.type != "once" or not job.enabled or job.last_run is not None:
            return
        scheduled = parse_datetime(job.schedule)
        if scheduled is None or scheduled > now or scheduled >= watermark:
            return
        self._storage.update(job.id, {"enabled": False, "last_status": "error", "next_run": None})
        self._on_change()

    def _spawn(self, job: ScheduleJob) -> None:
        task = asyncio.create_task(self._run_fire(job), name=f"schedule-fire:{job.id}")
        self._fire_tasks.add(task)
        task.add_done_callback(self._fire_tasks.discard)

    async def _run_fire(self, job: ScheduleJob) -> None:
        """Own one firing: status transitions in storage, delivery in ``fire``.

        The job record is re-read before each write because the tool, the menu,
        or a hand edit may have changed it while the fire was queued.
        """
        self._running_jobs.add(job.id)
        self._storage.update(job.id, {"last_status": "running"})
        self._on_change()
        ok = True
        try:
            await self._fire(job)
        except asyncio.CancelledError:
            self._storage.update(job.id, {"last_status": None})
            raise
        except Exception as exc:  # noqa: BLE001 - job failures are reported, not fatal
            ok = False
            print(
                f"tau-schedule-prompt: job {job.name} failed to fire: {exc}",
                file=sys.stderr,
            )
        finally:
            self._running_jobs.discard(job.id)
        self._write_stats(job.id, self._now(), ok=ok)
        self._on_change()

    def _write_stats(self, job_id: str, now: datetime, *, ok: bool) -> None:
        """Advance one job's counters from storage, never from a stale snapshot."""
        current = self._storage.get(job_id)
        if current is None:  # removed mid-flight: nothing to record
            return
        updates: dict[str, object] = {
            "last_run": to_iso(now),
            "last_status": "success" if ok else "error",
            "run_count": current.run_count + 1,
        }
        if current.type == "once":
            updates["enabled"] = False
            updates["next_run"] = None
        else:
            upcoming = next_run_at(current, now)
            updates["next_run"] = to_iso(upcoming) if upcoming is not None else None
        self._storage.update(job_id, updates)


__all__ = [
    "MAX_TICK_SECONDS",
    "ResolvedSchedule",
    "ScheduleError",
    "Scheduler",
    "describe_schedule",
    "is_loaded_for",
    "next_run_at",
    "parse_interval_ms",
    "parse_relative_delta",
    "pending_slot",
    "validate_schedule",
]
