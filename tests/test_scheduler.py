"""Tests for schedule validation, next-run maths, and the asyncio loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tau_schedule_prompt.scheduler import (
    ScheduleError,
    Scheduler,
    describe_schedule,
    is_loaded_for,
    next_run_at,
    parse_interval_ms,
    parse_relative_delta,
    validate_schedule,
)
from tau_schedule_prompt.storage import JobStorage
from tau_schedule_prompt.timeutils import to_iso
from tau_schedule_prompt.types import ScheduleJob, job_from_json


def cron_job(**overrides: object) -> ScheduleJob:
    """Build a loaded cron job for scheduler tests."""
    data: dict[str, object] = {
        "id": "job-1",
        "name": "heartbeat",
        "prompt": "check the build",
        "schedule": "* * * * * *",
        "type": "cron",
        "session": "sess-1",
        "created_at": to_iso(datetime(2026, 2, 13, 15, 0, tzinfo=UTC)),
    }
    data.update(overrides)
    job = job_from_json(data)  # type: ignore[arg-type]
    assert job is not None
    return job


# ---------------------------------------------------------------- validation


def test_relative_delta_resolves_to_absolute_time(frozen_now: datetime) -> None:
    resolved = validate_schedule("once", "+10s", now=frozen_now)
    assert resolved.schedule == to_iso(frozen_now + timedelta(seconds=10))
    assert parse_relative_delta("+2h", now=frozen_now) == frozen_now + timedelta(hours=2)


def test_once_rejects_past_and_too_soon(frozen_now: datetime) -> None:
    past = to_iso(frozen_now - timedelta(minutes=5))
    with pytest.raises(ScheduleError, match="in the past"):
        validate_schedule("once", past, now=frozen_now)
    with pytest.raises(ScheduleError, match="too soon"):
        validate_schedule("once", to_iso(frozen_now + timedelta(seconds=2)), now=frozen_now)
    with pytest.raises(ScheduleError, match="Invalid timestamp"):
        validate_schedule("once", "tomorrow-ish", now=frozen_now)


def test_interval_validation_and_units() -> None:
    assert validate_schedule("interval", "5m").interval_ms == 300_000
    assert parse_interval_ms("1h") == 3_600_000
    assert parse_interval_ms("12d") == 12 * 86_400_000
    with pytest.raises(ScheduleError, match="Invalid interval"):
        validate_schedule("interval", "5 minutes")


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("5m", "type='interval'"),
        ("+10s", "type='once'"),
        ("0 0 * * *", "Invalid cron"),
    ],
)
def test_cron_errors_redirect_to_the_right_type(text: str, expect: str) -> None:
    with pytest.raises(ScheduleError, match=expect):
        validate_schedule("cron", text)


def test_validate_trims_and_normalises_cron_whitespace() -> None:
    assert validate_schedule("cron", "  0   0 *  * * * ").schedule == "0 0 * * * *"


def test_describe_schedule_covers_all_three_types() -> None:
    assert describe_schedule("interval", "5m") == "every 5m"
    assert describe_schedule("cron", "0 */5 * * * *") == "every 5 min"
    described = describe_schedule("once", to_iso(datetime(2026, 2, 13, 20, 0, tzinfo=UTC)))
    assert described  # locale-dependent rendering, but never empty


# ----------------------------------------------------------------- next runs


def test_interval_catches_up_without_bursting(frozen_now: datetime) -> None:
    job = cron_job(
        type="interval", schedule="5m", interval_ms=300_000, created_at=to_iso(frozen_now)
    )
    much_later = frozen_now + timedelta(minutes=47)
    upcoming = next_run_at(job, much_later)
    assert upcoming is not None
    assert upcoming > much_later
    assert upcoming <= much_later + timedelta(minutes=5)


def test_disabled_jobs_and_stale_once_jobs_never_fire(frozen_now: datetime) -> None:
    assert next_run_at(cron_job(enabled=False), frozen_now) is None
    stale = cron_job(type="once", schedule=to_iso(frozen_now - timedelta(minutes=1)), enabled=True)
    assert next_run_at(stale, frozen_now) is None


def test_is_loaded_for_respects_session_binding() -> None:
    bound = cron_job(session="sess-1")
    assert is_loaded_for(bound, "sess-1")
    assert not is_loaded_for(bound, "sess-2")
    shared = cron_job(session=None)
    assert is_loaded_for(shared, "any-session")


# --------------------------------------------------------------------- loop


class Recorder:
    """Collects fires and change notifications from the scheduler."""

    def __init__(self) -> None:
        self.fired: list[str] = []
        self.changes = 0

    async def fire(self, job: ScheduleJob) -> None:
        self.fired.append(job.name)

    def on_change(self) -> None:
        self.changes += 1


def build_scheduler(
    cwd: Path,
    jobs: list[ScheduleJob],
    clock: datetime,
    *,
    watermark_shift_seconds: int = 10,
) -> tuple[Scheduler, Recorder, JobStorage]:
    """Wire a scheduler over a temp store with a fixed clock and start watermark."""
    storage = JobStorage(cwd)
    storage.save(jobs)
    recorder = Recorder()
    scheduler = Scheduler(
        storage,
        session_id="sess-1",
        fire=recorder.fire,
        on_change=recorder.on_change,
        clock=lambda: clock,
        watermark=clock - timedelta(seconds=watermark_shift_seconds),
    )
    return scheduler, recorder, storage


@pytest.mark.anyio
async def test_tick_fires_due_cron_job_and_records_stats(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    scheduler, recorder, storage = build_scheduler(project, [cron_job()], clock)

    await scheduler._tick()  # noqa: SLF001 - exercising the loop body directly
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert recorder.fired == ["heartbeat"]
    saved = storage.get("job-1")
    assert saved is not None
    assert saved.run_count == 1
    assert saved.last_status == "success"
    assert saved.last_run == to_iso(clock)
    assert saved.next_run is not None
    assert saved.next_run > to_iso(clock)


@pytest.mark.anyio
async def test_slot_before_the_watermark_is_not_retroactively_fired(
    project: Path,
) -> None:
    """A cron slot that passed while no session ran stays un-fired."""
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    hourly = cron_job(schedule="0 0 * * * *")  # last slot: 15:00
    scheduler, recorder, _ = build_scheduler(project, [hourly], clock, watermark_shift_seconds=60)

    await scheduler._tick()  # noqa: SLF001
    await asyncio.sleep(0)

    assert recorder.fired == []


@pytest.mark.anyio
async def test_tick_disables_one_shot_after_firing(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    job = cron_job(
        id="once-1", name="reminder", type="once", schedule=to_iso(clock - timedelta(seconds=1))
    )
    scheduler, recorder, storage = build_scheduler(project, [job], clock)

    await scheduler._tick()  # noqa: SLF001
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert recorder.fired == ["reminder"]
    saved = storage.get("once-1")
    assert saved is not None
    assert saved.enabled is False
    assert saved.run_count == 1
    assert saved.next_run is None


@pytest.mark.anyio
async def test_tick_retires_stale_one_shot_jobs(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    job = cron_job(
        id="stale-1", name="missed", type="once", schedule=to_iso(clock - timedelta(hours=2))
    )
    scheduler, _, storage = build_scheduler(project, [job], clock)

    delay = await scheduler._tick()  # noqa: SLF001

    assert delay > 0
    saved = storage.get("stale-1")
    assert saved is not None
    assert saved.enabled is False
    assert saved.last_status == "error"


@pytest.mark.anyio
async def test_tick_skips_jobs_bound_to_other_sessions(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    other = cron_job(id="job-2", name="theirs", session="sess-2")
    scheduler, recorder, _ = build_scheduler(project, [other], clock)

    await scheduler._tick()  # noqa: SLF001
    await asyncio.sleep(0)

    assert recorder.fired == []


@pytest.mark.anyio
async def test_tick_sleep_is_capped_and_positive(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    far = cron_job(schedule="0 0 0 1 1 *")  # next: New Year, hours away
    scheduler, _, _ = build_scheduler(project, [far], clock)

    delay = await scheduler._tick()  # noqa: SLF001
    assert 0 < delay <= 30.0


@pytest.mark.anyio
async def test_failure_records_error_status(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)

    async def boom(job: ScheduleJob) -> None:
        raise RuntimeError("delivery exploded")

    storage = JobStorage(project)
    storage.save([cron_job()])
    scheduler = Scheduler(
        storage,
        session_id="sess-1",
        fire=boom,
        on_change=lambda: None,
        clock=lambda: clock,
        watermark=clock - timedelta(seconds=10),
    )
    await scheduler._tick()  # noqa: SLF001
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    saved = storage.get("job-1")
    assert saved is not None
    assert saved.last_status == "error"
    assert saved.run_count == 1


@pytest.mark.anyio
async def test_start_clears_interrupted_runs_and_stop_is_idempotent(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    scheduler, _, storage = build_scheduler(project, [cron_job(last_status="running")], clock)

    scheduler.start()
    assert storage.get("job-1").last_status is None  # type: ignore[union-attr]

    await scheduler.stop()
    await scheduler.stop()


@pytest.mark.anyio
async def test_stop_cancels_an_in_flight_fire(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(job: ScheduleJob) -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    storage = JobStorage(project)
    storage.save([cron_job()])
    scheduler = Scheduler(
        storage,
        session_id="sess-1",
        fire=slow,
        on_change=lambda: None,
        clock=lambda: clock,
        watermark=clock - timedelta(seconds=10),
    )
    scheduler.start()
    await asyncio.wait_for(started.wait(), timeout=2)
    await scheduler.stop()

    assert cancelled.is_set()
    assert storage.get("job-1").last_status is None  # type: ignore[union-attr]


def test_next_run_helper_uses_injected_clock(project: Path) -> None:
    clock = datetime(2026, 2, 13, 15, 30, 30, tzinfo=UTC)
    scheduler, _, _ = build_scheduler(project, [cron_job()], clock)
    upcoming = scheduler.next_run(cron_job())
    assert upcoming is not None
    assert upcoming > clock
