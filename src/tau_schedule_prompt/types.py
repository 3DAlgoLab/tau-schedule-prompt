"""The scheduled-job record persisted in ``<cwd>/.tau/schedule-prompts.json``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .timeutils import to_iso, utc_now

JobType = Literal["cron", "once", "interval"]
JobStatus = Literal["pending", "success", "error", "running"]
JobScope = Literal["session", "workdir"]

JOB_TYPES: tuple[JobType, ...] = ("cron", "once", "interval")
DEFAULT_JOB_TYPE: JobType = "cron"


@dataclass(slots=True)
class ScheduleJob:
    """One scheduled prompt, storage-shaped (timestamps are ISO strings)."""

    id: str
    name: str
    schedule: str
    prompt: str
    type: JobType = "cron"
    enabled: bool = True
    interval_ms: int | None = None
    created_at: str = field(default_factory=lambda: to_iso(utc_now()))
    last_run: str | None = None
    last_status: JobStatus | None = None
    next_run: str | None = None
    run_count: int = 0
    description: str | None = None
    model: str | None = None
    notify: bool = False
    extensions: bool | list[str] = False
    skills: bool | list[str] = False
    session: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready mapping, dropping unset optional fields."""
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "type": self.type,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "run_count": self.run_count,
        }
        optional = {
            "interval_ms": self.interval_ms,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "next_run": self.next_run,
            "description": self.description,
            "model": self.model,
            "notify": self.notify or None,
            "extensions": self.extensions or None,
            "skills": self.skills or None,
            "session": self.session,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload

    def apply_updates(self, updates: dict[str, object]) -> ScheduleJob:
        """Apply ``updates`` in place and return ``self`` (keys are field names)."""
        for key, value in updates.items():
            if key not in self.__dataclass_fields__:
                raise KeyError(f"unknown job field: {key}")
            setattr(self, key, value)
        return self


def job_from_json(raw: Any) -> ScheduleJob | None:
    """Coerce one stored record into a :class:`ScheduleJob`, ignoring junk rows.

    The file is hand-editable on purpose (to unbind a job from a session, for
    example), so parsing is defensive: unknown keys are dropped and a row that
    cannot supply the required fields is skipped rather than breaking the load.
    """
    if not isinstance(raw, dict):
        return None
    identifier = raw.get("id")
    name = raw.get("name")
    schedule = raw.get("schedule")
    prompt = raw.get("prompt")
    if not all(isinstance(value, str) and value for value in (identifier, name, schedule, prompt)):
        return None

    raw_type = raw.get("type")
    job_type: JobType = raw_type if raw_type in JOB_TYPES else DEFAULT_JOB_TYPE
    interval_ms = raw.get("interval_ms")
    return ScheduleJob(
        id=str(identifier),
        name=str(name),
        schedule=str(schedule),
        prompt=str(prompt),
        type=job_type,
        enabled=bool(raw.get("enabled", True)),
        interval_ms=interval_ms if isinstance(interval_ms, int) and interval_ms > 0 else None,
        created_at=_text(raw.get("created_at")) or "",
        last_run=_text(raw.get("last_run")),
        last_status=_status(raw.get("last_status")),
        next_run=_text(raw.get("next_run")),
        run_count=_count(raw.get("run_count")),
        description=_text(raw.get("description")),
        model=_text(raw.get("model")),
        notify=bool(raw.get("notify", False)),
        extensions=_bool_or_names(raw.get("extensions")),
        skills=_bool_or_names(raw.get("skills")),
        session=_text(raw.get("session")),
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _status(value: Any) -> JobStatus | None:
    return value if value in ("pending", "success", "error", "running") else None


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


def _bool_or_names(value: Any) -> bool | list[str]:
    if value is True:
        return True
    if isinstance(value, list):
        names = [item for item in value if isinstance(item, str) and item]
        return names
    return False


__all__ = [
    "DEFAULT_JOB_TYPE",
    "JOB_TYPES",
    "JobScope",
    "JobStatus",
    "JobType",
    "ScheduleJob",
    "job_from_json",
]
