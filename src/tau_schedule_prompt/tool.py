"""The ``schedule_prompt`` tool: the model-facing job management surface.

Ports Pi's ``createCronTool`` onto Tau's :class:`tau_agent.tools.AgentTool`.
Tau's executor receives no extension context, so everything session-specific
arrives through :class:`ToolDeps` — small callables the extension wires to its
per-session state, which also keeps this module unit-testable without a session.

Errors are returned as tool text (``✗ Error: …``) rather than raised, matching
Pi: the model reads the message and can correct itself, and a bad ``add`` never
shows up as an unhandled extension failure.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tau_agent.messages import ImageContent, TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue

from .scheduler import ScheduleError, Scheduler, describe_schedule, validate_schedule
from .storage import JobStorage
from .subagent import is_subagent_session
from .timeutils import to_iso, utc_now
from .types import DEFAULT_JOB_TYPE, JOB_TYPES, JobScope, JobType, ScheduleJob

TOOL_NAME = "schedule_prompt"
GUARD_WINDOW = 10


@dataclass(frozen=True, slots=True)
class ToolDeps:
    """Session-scoped accessors the tool needs at call time."""

    storage: Callable[[], JobStorage]
    scheduler: Callable[[], Scheduler | None]
    default_scope: Callable[[], JobScope]
    session_id: Callable[[], str | None]
    recent_scheduled_marker: Callable[[], bool] = lambda: False


PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "list", "remove", "enable", "disable", "update", "cleanup"],
            "description": (
                "add (needs schedule + prompt), list, remove/enable/disable/update "
                "(need jobId), cleanup (drop disabled jobs)."
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "Unique job name for add; renamed for update. Auto-generated if omitted."
            ),
        },
        "schedule": {
            "type": "string",
            "description": (
                "6-field cron 'second minute hour dom month dow' (type=cron, e.g. "
                "'0 0 9 * * *' = daily 09:00), relative time '+10s'/'+5m' or ISO "
                "timestamp (type=once), or an interval '30s'/'5m'/'1h' (type=interval)."
            ),
        },
        "prompt": {
            "type": "string",
            "description": "The prompt text delivered when the job fires (required for add).",
        },
        "type": {
            "type": "string",
            "enum": list(JOB_TYPES),
            "description": (
                "Schedule kind; defaults to 'cron'. Use 'once' for +deltas/ISO, "
                "'interval' for durations."
            ),
        },
        "jobId": {
            "type": "string",
            "description": "Target job id for remove/enable/disable/update.",
        },
        "description": {"type": "string", "description": "Optional human note about the job."},
        "model": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Run the prompt in a background child session pinned to this model "
                "instead of injecting it into this conversation. Omit for inline delivery; "
                "there is no in-place way to clear it again."
            ),
        },
        "notify": {
            "type": "boolean",
            "description": (
                "With model set: wake this conversation with the subagent's result when it "
                "finishes."
            ),
        },
        "extensions": {
            "anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}],
            "description": "With model set: let the child session load extensions (default: none).",
        },
        "skills": {
            "anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}],
            "description": "With model set: let the child session load skills (default: none).",
        },
    },
    "required": ["action"],
}

DESCRIPTION = (
    "Schedule prompts to run later in this session, in the background, or on a "
    "timer. action='add' requires BOTH 'schedule' and 'prompt'. Schedule formats: "
    "6-field cron with seconds ('0 0 9 * * *' = daily 09:00, '0 */15 * * * *' = "
    "every 15 min), relative time ('+10s', '+5m'), ISO timestamp, or interval "
    "('30s', '5m', '1h'). type defaults to 'cron': use type='once' for +deltas and "
    "ISO timestamps, type='interval' for repeating durations. With 'model' set the "
    "prompt runs in a background child session on that model; without it the prompt "
    "is delivered to this conversation. Actions: add, list, remove, enable, disable, "
    "update, cleanup."
)

PROMPT_GUIDELINES: tuple[str, ...] = (
    "Scheduled prompts only fire while a Tau session is open in this project directory.",
    "For a single delayed run use type='once' with '+10s'-style time; for a repeating "
    "cadence use type='interval'; for wall-clock times use 6-field cron.",
    "Do not create a scheduled job from inside a scheduled prompt; the tool rejects "
    "that to prevent feedback loops.",
)


def _new_id(size: int = 10) -> str:
    """Return a short unique id (Pi used ``nanoid``)."""
    return uuid.uuid4().hex[:size]


def _text(message: str) -> list[TextContent | ImageContent]:
    """Return one text block typed for ``AgentToolResult.content`` (invariant list)."""
    return [TextContent(text=message)]


def _error(message: str, details: dict[str, Any]) -> AgentToolResult:
    details["error"] = message
    return AgentToolResult(content=_text(f"✗ Error: {message}"), details=details)


def _model_line(job: ScheduleJob) -> str:
    """Describe a job's subagent configuration in one line."""
    if not job.model:
        return ""
    flags = [", notifies parent" if job.notify else "", ", extensions" if job.extensions else ""]
    if job.skills:
        flags.append(", skills")
    return f"\nModel: {job.model} (runs in a background session{flags[0]}{flags[1]})"


def _list_text(jobs: list[ScheduleJob], scheduler: Scheduler | None) -> str:
    """Render the ``list`` action's model-facing text."""
    if not jobs:
        return "No scheduled jobs in this project."
    lines = ["Scheduled jobs:", ""]
    for job in jobs:
        mark = "✓" if job.enabled else "✗"
        upcoming = scheduler.next_run(job) if scheduler is not None else None
        next_str = f"Next: {to_iso(upcoming)}" if upcoming is not None else ""
        last_str = f"Last: {job.last_run}" if job.last_run else "Never run"
        lines.append(f"{mark} {job.name} ({job.id})")
        lines.append(
            f"  Type: {job.type} | Schedule: {job.schedule} "
            f"({describe_schedule(job.type, job.schedule)})"
        )
        if job.model:
            lines.append(f"  Model: {job.model}{_model_line(job).replace(chr(10), '')}")
        lines.append(f"  Prompt: {job.prompt}")
        lines.append(f"  {last_str}{' | ' + next_str if next_str else ''}")
        lines.append(f"  Runs: {job.run_count} | Status: {job.last_status or 'pending'}")
        if job.description:
            lines.append(f"  Description: {job.description}")
        lines.append("")
    return "\n".join(lines).rstrip()


def create_schedule_tool(deps: ToolDeps) -> AgentTool:
    """Build the ``schedule_prompt`` tool bound to one session's state."""

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        action = str(arguments.get("action", ""))
        details: dict[str, Any] = {"action": action, "jobs": []}
        storage = deps.storage()
        scheduler = deps.scheduler()

        try:
            if action == "add":
                return _add(deps, storage, scheduler, arguments, details)
            if action == "list":
                return _list(deps, storage, scheduler, details)
            if action == "remove":
                return _remove(deps, storage, scheduler, arguments, details)
            if action in ("enable", "disable"):
                return _set_enabled(deps, storage, scheduler, action, arguments, details)
            if action == "update":
                return _update(deps, storage, scheduler, arguments, details)
            if action == "cleanup":
                return _cleanup(deps, storage, scheduler, details)
            raise ValueError(f"Unknown action: {action}")
        except (ScheduleError, ValueError, KeyError) as exc:
            return _error(str(exc), details)

    return AgentTool(
        name=TOOL_NAME,
        label="schedule prompt",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute_fn=execute,
        prompt_snippet="Schedule prompts to run later in this project.",
        prompt_guidelines=PROMPT_GUIDELINES,
        render_call=_render_call,
        render_result=_render_result,
    )


def _guard_against_recursion(deps: ToolDeps) -> None:
    """Refuse to schedule from inside a scheduled run (Pi's loop breaker)."""
    if is_subagent_session(deps.session_id()):
        raise ValueError("Cannot create scheduled prompts from inside a scheduled subagent run.")
    if deps.recent_scheduled_marker():
        raise ValueError(
            "Cannot create scheduled prompts from within a scheduled prompt execution. "
            "This prevents infinite loops."
        )


def _add(
    deps: ToolDeps,
    storage: JobStorage,
    scheduler: Scheduler | None,
    arguments: Mapping[str, JSONValue],
    details: dict[str, Any],
) -> AgentToolResult:
    schedule = _string(arguments.get("schedule"))
    prompt = _string(arguments.get("prompt"))
    if not schedule or not prompt:
        missing = [
            name for name, value in (("'schedule'", schedule), ("'prompt'", prompt)) if not value
        ]
        raise ValueError(
            f"Missing required parameters for add action: {' and '.join(missing)}. You must "
            "provide both schedule (e.g. '+10s', '0 */15 * * * *') and prompt (the text to run)."
        )
    _guard_against_recursion(deps)
    model = arguments.get("model")
    if isinstance(model, str) and not model.strip():
        raise ValueError(
            "'model' must be a non-empty string. Omit the field for inline (no-model) jobs."
        )

    name = _string(arguments.get("name")) or f"job-{_new_id(6)}"
    if storage.has_job_named(name):
        raise ValueError(
            f'A job named "{name}" already exists. Use a different name, or remove the '
            "existing job first."
        )
    job_type = _job_type(arguments.get("type"))
    resolved = validate_schedule(job_type, schedule)

    session = deps.session_id() if deps.default_scope() == "session" else None
    job = ScheduleJob(
        id=_new_id(),
        name=name,
        schedule=resolved.schedule,
        prompt=prompt,
        type=job_type,
        enabled=True,
        interval_ms=resolved.interval_ms,
        created_at=to_iso(utc_now()),
        description=_string(arguments.get("description")) or None,
        model=_string(model) or None,
        notify=bool(arguments.get("notify", False)),
        extensions=_bool_or_names(arguments.get("extensions")),
        skills=_bool_or_names(arguments.get("skills")),
        session=session,
    )
    job.next_run = _next_run_iso(scheduler, job)
    storage.add(job)
    if scheduler is not None:
        scheduler.wake()

    details.update({"jobs": [job.to_json()], "job_id": job.id, "job_name": job.name})
    scope_note = "" if session else "\nScope: any Tau session in this directory"
    return AgentToolResult(
        content=_text(
            f"✓ Created job “{job.name}” ({job.id})\nType: {job.type}\n"
            f"Schedule: {job.schedule} — {describe_schedule(job.type, job.schedule)}\n"
            f"Prompt: {job.prompt}{_model_line(job)}{scope_note}"
        ),
        details=details,
    )


def _list(
    deps: ToolDeps, storage: JobStorage, scheduler: Scheduler | None, details: dict[str, Any]
) -> AgentToolResult:
    jobs = _visible_jobs(deps, storage)
    details["jobs"] = [job.to_json() for job in jobs]
    return AgentToolResult(content=_text(_list_text(jobs, scheduler)), details=details)


def _remove(
    deps: ToolDeps,
    storage: JobStorage,
    scheduler: Scheduler | None,
    arguments: Mapping[str, JSONValue],
    details: dict[str, Any],
) -> AgentToolResult:
    job = _require_job(deps, storage, arguments)
    if not storage.remove(job.id):
        raise ValueError(f"Failed to remove job: {job.id}")
    if scheduler is not None:
        scheduler.wake()
    details.update({"job_id": job.id, "job_name": job.name})
    return AgentToolResult(content=_text(f"✓ Removed job “{job.name}” ({job.id})"), details=details)


def _set_enabled(
    deps: ToolDeps,
    storage: JobStorage,
    scheduler: Scheduler | None,
    action: str,
    arguments: Mapping[str, JSONValue],
    details: dict[str, Any],
) -> AgentToolResult:
    job = _require_job(deps, storage, arguments)
    enabled = action == "enable"
    updates: dict[str, object] = {"enabled": enabled}
    if enabled:
        resolved = validate_schedule(job.type, job.schedule)
        updates["schedule"] = resolved.schedule
        updates["interval_ms"] = resolved.interval_ms
    updated = storage.update(job.id, updates) or job
    if scheduler is not None:
        scheduler.wake()
    details.update({"jobs": [updated.to_json()], "job_id": job.id, "job_name": job.name})
    verb = "Enabled" if enabled else "Disabled"
    return AgentToolResult(content=_text(f"✓ {verb} job “{job.name}” ({job.id})"), details=details)


def _update(
    deps: ToolDeps,
    storage: JobStorage,
    scheduler: Scheduler | None,
    arguments: Mapping[str, JSONValue],
    details: dict[str, Any],
) -> AgentToolResult:
    job = _require_job(deps, storage, arguments)
    model = arguments.get("model")
    if isinstance(model, str) and not model.strip():
        raise ValueError(
            "'model' must be a non-empty string. To switch a job from subagent back to "
            "inline, remove it and re-add it without a model."
        )
    updates: dict[str, object] = {}
    for key in ("name", "prompt", "description"):
        if key in arguments and arguments[key] is not None:
            updates[key] = str(arguments[key])
    for key in ("model", "notify", "extensions", "skills"):
        if key in arguments:
            updates[key] = _string(arguments.get("model")) if key == "model" else arguments[key]
    if "name" in updates and not str(updates["name"]).strip():
        raise ValueError("'name' must not be empty.")
    if (
        "name" in updates
        and str(updates["name"]) != job.name
        and storage.has_job_named(str(updates["name"]))
    ):
        raise ValueError(f'A job named "{updates["name"]}" already exists.')
    if _string(arguments.get("schedule")):
        resolved = validate_schedule(job.type, str(arguments["schedule"]))
        updates["schedule"] = resolved.schedule
        updates["interval_ms"] = resolved.interval_ms
    if not updates:
        raise ValueError(
            "Nothing to update: pass name, prompt, schedule, description, model, notify, "
            "extensions, or skills."
        )

    updated = storage.update(job.id, updates) or job
    if scheduler is not None:
        scheduler.wake()
    details.update({"jobs": [updated.to_json()], "job_id": job.id, "job_name": updated.name})
    return AgentToolResult(
        content=_text(
            f"✓ Updated job “{updated.name}” ({job.id})\n"
            f"Schedule: {updated.schedule} — {describe_schedule(updated.type, updated.schedule)}"
        ),
        details=details,
    )


def _cleanup(
    deps: ToolDeps,
    storage: JobStorage,
    scheduler: Scheduler | None,
    details: dict[str, Any],
) -> AgentToolResult:
    disabled = [job for job in _visible_jobs(deps, storage) if not job.enabled]
    if not disabled:
        details["jobs"] = []
        return AgentToolResult(content=_text("No disabled jobs to clean up."), details=details)
    for job in disabled:
        storage.remove(job.id)
    if scheduler is not None:
        scheduler.wake()
    details["jobs"] = [job.to_json() for job in disabled]
    listing = "\n".join(f"  - {job.name} ({job.id})" for job in disabled)
    return AgentToolResult(
        content=_text(f"✓ Removed {len(disabled)} disabled job(s):\n{listing}"), details=details
    )


def _visible_jobs(deps: ToolDeps, storage: JobStorage) -> list[ScheduleJob]:
    """Jobs this session may act on: its own plus session-less ones."""
    from .scheduler import is_loaded_for

    session_id = deps.session_id()
    return [job for job in storage.all() if is_loaded_for(job, session_id)]


def _require_job(
    deps: ToolDeps, storage: JobStorage, arguments: Mapping[str, JSONValue]
) -> ScheduleJob:
    job_id = _string(arguments.get("jobId"))
    if not job_id:
        raise ValueError("jobId is required for this action")
    job = storage.get(job_id)
    if job is None:
        raise ValueError(f"Job not found: {job_id}")
    from .scheduler import is_loaded_for

    if not is_loaded_for(job, deps.session_id()):
        raise ValueError(
            f'Job “{job.name}” belongs to another session; drop its "session" field in '
            ".tau/schedule-prompts.json to share it."
        )
    return job


def _next_run_iso(scheduler: Scheduler | None, job: ScheduleJob) -> str | None:
    upcoming = scheduler.next_run(job) if scheduler is not None else None
    return to_iso(upcoming) if upcoming is not None else None


def _job_type(value: Any) -> JobType:
    return value if value in JOB_TYPES else DEFAULT_JOB_TYPE


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool_or_names(value: Any) -> bool | list[str]:
    if isinstance(value, bool):
        return value
    if isinstance(value, Sequence):
        names = [item for item in value if isinstance(item, str) and item.strip()]
        return names
    return False


def _render_call(arguments: Mapping[str, JSONValue]) -> str:
    action = str(arguments.get("action", ""))
    target = _string(arguments.get("name")) or _string(arguments.get("jobId"))
    verbs = {
        "add": "Scheduling",
        "list": "Listing scheduled jobs",
        "remove": "Removing",
        "enable": "Enabling",
        "disable": "Disabling",
        "update": "Updating",
        "cleanup": "Cleaning up disabled jobs",
    }
    verb = verbs.get(action, action)
    return f"[cyan]{verb}[/cyan] [bold]{target}[/bold]" if target else f"[cyan]{verb}[/cyan]"


def _render_result(result: AgentToolResult, *, expanded: bool) -> str | None:
    details = result.details if isinstance(result.details, dict) else {}
    error = details.get("error")
    if error:
        return f"[red]✗ {error}[/red]"
    action = str(details.get("action", ""))
    name = details.get("job_name") or details.get("job_id") or ""
    lines = [
        f"[green]✓ {action}[/green] [bold]{name}[/bold]" if name else f"[green]✓ {action}[/green]"
    ]
    jobs = details.get("jobs")
    if action in ("list", "cleanup") and isinstance(jobs, list) and jobs:
        lines.append("")
        for raw in jobs:
            if not isinstance(raw, dict):
                continue
            mark = "✓" if raw.get("enabled", True) else "✗"
            model = f" [cyan]{raw.get('model')}[/cyan]" if raw.get("model") else ""
            lines.append(
                f" {mark} [bold]{raw.get('name')}[/bold] [dim]({raw.get('id')})[/dim] "
                f"[dim]{raw.get('type')} {raw.get('schedule')} "
                f"· runs {raw.get('run_count', 0)}[/dim]{model}"
            )
    del expanded
    return "\n".join(lines)


__all__ = [
    "DESCRIPTION",
    "PARAMETERS",
    "PROMPT_GUIDELINES",
    "TOOL_NAME",
    "ToolDeps",
    "create_schedule_tool",
]
