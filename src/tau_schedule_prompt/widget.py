"""The "below the prompt" widget listing scheduled jobs.

Pi renders a bordered table through ``ctx.ui.setWidget(..., belowEditor)``;
Tau's equivalent seam is ``context.ui.components.set_slot_widget(key, lines,
placement="below_prompt")``, where ``lines`` are Rich-markup strings the host
paints into a ``Static``. Keeping the content plain strings means this module
never imports Textual, so the extension stays frontend-agnostic and degrades to
widget-less (but still functional) in print mode.

Column widths follow Pi so the muscle memory carries over: status, name(15),
schedule(15), prompt(25), next(10), last(10), runs(3), then a model badge.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from rich.markup import escape

from .scheduler import describe_schedule, next_run_at
from .timeutils import format_relative, parse_datetime, truncate, utc_now
from .types import ScheduleJob

WIDGET_KEY = "schedule-prompts"
NAME_WIDTH = 15
SCHEDULE_WIDTH = 15
PROMPT_WIDTH = 25
TIME_WIDTH = 10
COUNT_WIDTH = 3
MODEL_WIDTH = 12
MAX_ROWS = 6

STATUS_ICONS: Mapping[str, tuple[str, str]] = {
    "disabled": ("✗", "dim"),
    "running": ("⟳", "yellow"),
    "error": ("!", "red"),
    "success": ("✓", "green"),
    "pending": ("○", "dim"),
}


def status_icon(status: str) -> str:
    """Return a colour-coded status glyph for one job state."""
    glyph, style = STATUS_ICONS.get(status, ("○", "dim"))
    return f"[{style}]{glyph}[/{style}]"


def job_status(job: ScheduleJob, running: Mapping[str, str]) -> str:
    """Return the display status for a job, preferring live run state."""
    if not job.enabled:
        return "disabled"
    if job.id in running:
        return "running"
    return job.last_status or "pending"


def _cell(text: str, width: int, *, style: str | None = None) -> str:
    """Pad-then-colour one cell (padding happens outside the markup tags)."""
    padded = escape(truncate(text, width)).ljust(width)
    return f"[{style}]{padded}[/{style}]" if style else padded


def row(job: ScheduleJob, *, now: datetime, running: Mapping[str, str] | None = None) -> str:
    """Render one padded table row for ``job``."""
    live = running or {}
    status = job_status(job, live)
    name_style = "dim" if status == "disabled" else None
    name_cell = _cell(job.name, NAME_WIDTH, style=name_style)
    schedule_cell = _cell(describe_schedule(job.type, job.schedule), SCHEDULE_WIDTH, style="dim")
    prompt_cell = _cell(job.prompt, PROMPT_WIDTH, style="dim")

    upcoming = None if job.id in live else next_run_at(job, now)
    next_label = format_relative(upcoming, now=now) if upcoming is not None else "-"
    next_cell = escape(next_label).ljust(TIME_WIDTH)

    last_moment = parse_datetime(job.last_run)
    last_label = format_relative(last_moment, now=now) if last_moment is not None else "never"
    last_cell = (
        escape(last_label).ljust(TIME_WIDTH)
        if last_moment is not None
        else f"[dim]{escape(last_label).ljust(TIME_WIDTH)}[/dim]"
    )

    count_cell = f"[bold]{str(job.run_count).rjust(COUNT_WIDTH)}[/bold]"
    badge = ""
    if job.model:
        badge_text = truncate(job.model, MODEL_WIDTH) + ("!" if job.notify else "")
        badge = f" [cyan][{escape(badge_text)}][/cyan]"
    live_note = f" [yellow]{escape(live[job.id])}[/yellow]" if job.id in live else ""
    return (
        f" {status_icon(status)} {name_cell} {schedule_cell} {prompt_cell} "
        f"{next_cell} {last_cell} {count_cell}{badge}{live_note}"
    )


class ScheduleWidget:
    """Mounts and refreshes the job table through the host's component seam."""

    def __init__(self, components: object, *, max_rows: int = MAX_ROWS) -> None:
        self._components = components
        self._max_rows = max_rows
        self._mounted = False

    def lines(
        self,
        jobs: list[ScheduleJob],
        *,
        running: Mapping[str, str] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Build the display lines; an empty list means "nothing to show"."""
        if not jobs:
            return []
        moment = now or utc_now()
        live = running or {}
        label = "job" if len(jobs) == 1 else "jobs"
        out = [f"[cyan]⏰ Scheduled[/cyan] [dim]{len(jobs)} {label}[/dim]"]
        ordered = sorted(jobs, key=lambda job: (not job.enabled, job.name))
        out.extend(row(job, now=moment, running=live) for job in ordered[: self._max_rows])
        hidden = len(ordered) - self._max_rows
        if hidden > 0:
            out.append(f"[dim]  … {hidden} more — /schedule-prompt[/dim]")
        return out

    def show(
        self,
        jobs: list[ScheduleJob],
        *,
        running: Mapping[str, str] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Mount (or re-mount) the table; unmount when there is nothing to show."""
        content = self.lines(jobs, running=running, now=now)
        if not content:
            self.hide()
            return
        setter = getattr(self._components, "set_slot_widget", None)
        if setter is None:  # frontend without a component seam
            return
        setter(WIDGET_KEY, content, placement="below_prompt")
        self._mounted = True

    def hide(self) -> None:
        """Unmount the table (idempotent)."""
        if not self._mounted:
            return
        setter = getattr(self._components, "set_slot_widget", None)
        if setter is not None:
            setter(WIDGET_KEY, None, placement="below_prompt")
        self._mounted = False


__all__ = [
    "MAX_ROWS",
    "STATUS_ICONS",
    "WIDGET_KEY",
    "ScheduleWidget",
    "job_status",
    "row",
    "status_icon",
]
