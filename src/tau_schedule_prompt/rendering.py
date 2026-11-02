"""Transcript markers for fired jobs, and the renderer that formats them.

Tau's custom-message seam (``register_message_renderer`` / ``send_custom_message``)
is the port of Pi's ``registerMessageRenderer``: ``content`` is what the model
sees, ``details`` is what the renderer formats. The renderer must return a
Rich-markup string, never a widget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from rich.markup import escape
from tau_coding.extensions import CustomMessageView, MessageRenderOptions

from .types import ScheduleJob

CUSTOM_TYPE = "scheduled_prompt"
MarkerMode = Literal["inline", "subagent_start", "subagent_done", "subagent_error"]


def marker_details(
    job: ScheduleJob,
    mode: MarkerMode = "inline",
    **extra: Any,
) -> dict[str, Any]:
    """Build the ``details`` payload carried by a job's transcript marker."""
    details: dict[str, Any] = {
        "job_id": job.id,
        "job_name": job.name,
        "prompt": job.prompt,
        "mode": mode,
    }
    if job.model:
        details["model"] = job.model
    details.update({key: value for key, value in extra.items() if value is not None})
    return details


def render_marker(view: CustomMessageView, options: MessageRenderOptions) -> str:
    """Format a scheduled-prompt marker as one transcript line."""
    details: Mapping[str, Any] = view.details or {}
    name = escape(str(details.get("job_name") or "Unknown"))
    prompt = escape(str(details.get("prompt") or ""))
    model = details.get("model")
    tag = f" [bold]({escape(str(model))})[/bold]" if model else ""
    mode = str(details.get("mode") or "inline")
    output = escape(truncate_marker(str(details.get("output") or "")))
    error = escape(truncate_marker(str(details.get("error") or "")))
    del options  # markers are single-line; expansion adds nothing useful

    if mode == "subagent_start":
        head = f"[cyan]⏰ Scheduled{tag}: {name}[/cyan]"
        return f"{head} [dim]→ “{prompt}”[/dim]" if prompt else head
    if mode == "subagent_done":
        head = f"[green]✓ Scheduled{tag} finished: {name}[/green]"
        return f"{head} [dim]→ {output}[/dim]" if output else head
    if mode == "subagent_error":
        head = f"[red]✗ Scheduled{tag} failed: {name}[/red]"
        return f"{head} [dim]→ {error}[/dim]" if error else head
    head = f"[cyan]⏰ Scheduled: {name}[/cyan]"
    return f"{head} [dim]→ “{prompt}”[/dim]" if prompt else head


def truncate_marker(text: str, limit: int = 200) -> str:
    """Keep marker output/error snippets to a single transcript row."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 1] + "…"


__all__ = ["CUSTOM_TYPE", "MarkerMode", "marker_details", "render_marker", "truncate_marker"]
