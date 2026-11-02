"""Tau entry point: the wiring Pi keeps in ``index.ts``.

Four registrations, and nothing else — every behavior lives in the tested
modules this file binds together:

- the ``schedule_prompt`` tool (:mod:`.tool`), fed per-session accessors through
  :class:`~.tool.ToolDeps` because Tau's tool executor gets no extension context;
- one transcript renderer for the ``scheduled_prompt`` custom type (:mod:`.rendering`);
- the below-prompt job table (:mod:`.widget`), repainted whenever the scheduler
  fires or a tool/menu mutation lands;
- a :class:`~.scheduler.Scheduler` started on ``session_start`` and stopped on
  ``session_shutdown``, so no timer outlives the session.

Two deliberate deviations from Pi:

- **Inline delivery is one message.** Pi sends a visible-only marker with empty
  ``content`` and then a separate ``sendUserMessage(prompt)`` so the prompt
  reaches the model once. Tau's ``send_custom_message`` puts ``content``
  straight into LLM context, so the marker *is* the delivery: one message,
  prompt text once, and no empty-content message for a provider to choke on.
- **Widget content is plain Rich-markup lines**, never a Textual widget, so the
  extension stays frontend-agnostic and degrades to widget-less in print mode.

Install for future runs with ``tau install ../tau-schedule-prompt``, or try it
once with ``tau -e ../tau-schedule-prompt``.
"""

from __future__ import annotations

from dataclasses import replace

from tau_coding.extensions import (
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    ExtensionError,
    SessionShutdownEvent,
    SessionStartEvent,
)

from .rendering import CUSTOM_TYPE, marker_details, render_marker
from .scheduler import Scheduler, describe_schedule
from .settings import ScheduleSettings, load_settings, save_settings
from .storage import JobStorage
from .subagent import ModelRef, ParentSnapshot, run_subagent, snippet
from .timeutils import to_iso
from .tool import GUARD_WINDOW, ToolDeps, create_schedule_tool
from .types import JobScope, ScheduleJob
from .widget import ScheduleWidget

COMMAND_NAME = "schedule-prompt"
COMMAND_USAGE = "[list | show | hide | pause <job> | resume <job> | remove <job> | cleanup]"
COMMAND_DESCRIPTION = "List, toggle, or remove scheduled prompts and their widget."


class ScheduleExtension:
    """One extension generation's per-session state and event handlers."""

    def __init__(self, tau: ExtensionAPI) -> None:
        self._tau = tau
        self._settings = ScheduleSettings()
        self._scheduler: Scheduler | None = None
        self._widget: ScheduleWidget | None = None

    # -- registration -------------------------------------------------------

    def register(self) -> None:
        """Attach the tool, renderer, command, and lifecycle handlers."""
        self._tau.register_tool(
            create_schedule_tool(
                ToolDeps(
                    storage=self.storage,
                    scheduler=lambda: self._scheduler,
                    default_scope=self.default_scope,
                    session_id=self.session_id,
                    recent_scheduled_marker=self.recent_scheduled_marker,
                )
            )
        )
        self._tau.register_message_renderer(CUSTOM_TYPE, render_marker)
        self._tau.on("session_start", self.on_session_start)
        self._tau.on("session_shutdown", self.on_session_shutdown)
        self._tau.register_command(
            COMMAND_NAME,
            self.handle_command,
            description=COMMAND_DESCRIPTION,
            usage=COMMAND_USAGE,
        )

    # -- ToolDeps seams (resolved per call, never snapshotted) --------------

    def storage(self) -> JobStorage:
        """Return the store for the session's current working directory."""
        return JobStorage(self._tau.context.cwd)

    def default_scope(self) -> JobScope:
        """Return the configured scope for new jobs."""
        return self._settings.default_job_scope

    def session_id(self) -> str | None:
        """Return the active session id, used to bind jobs to this session."""
        return self._tau.context.session_id

    def recent_scheduled_marker(self) -> bool:
        """Return whether a scheduled-prompt marker is already near the end."""
        transcript = self._tau.context.transcript
        return any(
            getattr(message, "custom_type", None) == CUSTOM_TYPE
            for message in transcript[-GUARD_WINDOW:]
        )

    # -- lifecycle ----------------------------------------------------------

    def on_session_start(self, event: SessionStartEvent, context: ExtensionContext) -> None:
        """Load settings, mount the widget, and start the wake-up loop."""
        del event
        self._settings = load_settings(context.cwd, context.paths.home)
        self._widget = ScheduleWidget(context.ui.components)
        scheduler = Scheduler(
            JobStorage(context.cwd),
            session_id=context.session_id,
            fire=self._fire,
            on_change=self._on_change,
        )
        self._scheduler = scheduler
        # ``session_start`` is emitted from the session's own event loop, so the
        # loop task is safe to create here; it dies with ``session_shutdown``.
        scheduler.start()

    async def on_session_shutdown(
        self, event: SessionShutdownEvent, context: ExtensionContext
    ) -> None:
        """Cancel the loop and any in-flight fire, then unmount the widget."""
        del event, context
        scheduler, self._scheduler = self._scheduler, None
        widget, self._widget = self._widget, None
        if widget is not None:
            widget.hide()
        if scheduler is not None:
            await scheduler.stop()

    # -- firing -------------------------------------------------------------

    async def _fire(self, job: ScheduleJob) -> None:
        """Deliver one firing: inline into this session, or as a child session."""
        if job.model:
            await self._fire_subagent(job)
            return
        context = self._tau.context
        self._tau.send_custom_message(
            job.prompt,
            custom_type=CUSTOM_TYPE,
            details=marker_details(job),
            deliver_as="steer" if context.is_running else "follow_up",
        )

    async def _fire_subagent(self, job: ScheduleJob) -> None:
        """Run a model-pinned job in a child session and record its outcome.

        The result never raises: :func:`~.subagent.run_subagent` reports
        failures, and the scheduler records the status. Only ``notify`` jobs
        wake the parent conversation; the rest are appended for the next run.
        """
        parent = self._snapshot()
        self._tau.send_custom_message(
            f"Scheduled job “{job.name}” started in a background session ({job.model}).",
            custom_type=CUSTOM_TYPE,
            details=marker_details(job, "subagent_start"),
            deliver_as="follow_up",
            trigger_turn=False,
        )
        result = await run_subagent(job, parent, on_activity=self._notify)
        output = snippet(result.text) if result.ok else snippet(result.error)
        if result.ok:
            details = marker_details(job, "subagent_done", output=output)
            content = f"Scheduled job “{job.name}” finished: {output or '(no output)'}"
        else:
            details = marker_details(job, "subagent_error", error=output)
            content = f"Scheduled job “{job.name}” failed: {output}"
        self._tau.send_custom_message(
            content,
            custom_type=CUSTOM_TYPE,
            details=details,
            deliver_as="follow_up",
            trigger_turn=bool(job.notify),
        )

    def _snapshot(self) -> ParentSnapshot:
        """Read the parent values a background run needs, on the session thread."""
        context = self._tau.context
        return ParentSnapshot(
            cwd=context.cwd,
            tau_home=context.paths.home,
            provider_name=context.provider_name,
            model=context.model,
            project_trusted=context.project_trusted,
            model_choices=tuple(
                ModelRef(choice.provider_name, choice.model)
                for choice in context.available_model_choices
            ),
        )

    def _notify(self, message: str) -> None:
        """Surface subagent progress without ever failing a firing."""
        try:
            self._tau.context.ui.notify(f"{COMMAND_NAME}: {message}", "info")
        except ExtensionError:
            return

    # -- widget -------------------------------------------------------------

    def _on_change(self) -> None:
        """Repaint the widget after a fire, a tick, or a tool/UI mutation."""
        try:
            context = self._tau.context
        except ExtensionError:
            return  # generation retired (/reload, quit): nothing left to paint
        self._refresh_widget(context)

    def _refresh_widget(self, context: ExtensionContext) -> None:
        widget, scheduler = self._widget, self._scheduler
        if widget is None or scheduler is None:
            return
        if not self._settings.widget_visible:
            widget.hide()
            return
        widget.show(scheduler.loaded_jobs())

    # -- /schedule-prompt ---------------------------------------------------

    def handle_command(self, args: str, context: ExtensionCommandContext) -> str:
        """Run one subcommand; returns the text Tau shows in the transcript."""
        del context
        parts = args.strip().split()
        verb = parts[0].lower() if parts else "list"
        storage = self.storage()

        if verb in ("list", "ls"):
            return self._describe_jobs(storage)
        if verb in ("show", "hide"):
            return self._set_widget_visible(verb == "show")
        if verb in ("pause", "resume"):
            return self._set_enabled(storage, parts[1:], enabled=verb == "resume")
        if verb == "remove":
            return self._remove_job(storage, parts[1:])
        if verb == "cleanup":
            return self._cleanup(storage)
        return f"Unknown /{COMMAND_NAME} subcommand: {verb}\nUsage: /{COMMAND_NAME} {COMMAND_USAGE}"

    def _describe_jobs(self, storage: JobStorage) -> str:
        jobs = storage.all()
        if not jobs:
            return "No scheduled jobs in this project."
        scheduler = self._scheduler
        lines = [f"Scheduled jobs ({len(jobs)}):"]
        for job in sorted(jobs, key=lambda item: (not item.enabled, item.name)):
            if scheduler is not None:
                state = scheduler.job_state(job)
                upcoming = scheduler.next_run(job)
            else:
                state = "disabled" if not job.enabled else (job.last_status or "pending")
                upcoming = None
            lines.append(
                f"  {'✓' if job.enabled else '✗'} {job.name} ({job.id}) — "
                f"{describe_schedule(job.type, job.schedule)} — {state}"
                + (f" | next: {to_iso(upcoming)}" if upcoming else "")
                + (f" | model: {job.model}" if job.model else "")
            )
        return "\n".join(lines)

    def _set_widget_visible(self, visible: bool) -> str:
        self._settings = replace(self._settings, widget_visible=visible)
        saved = save_settings(self._tau.context.cwd, {"widget_visible": visible})
        self._on_change()
        state = "shown" if visible else "hidden"
        note = "" if saved else " (could not write the settings file; applies this session only)"
        return f"Scheduled-job widget {state}.{note}"

    def _set_enabled(self, storage: JobStorage, tokens: list[str], *, enabled: bool) -> str:
        found = self._find_job(storage, tokens)
        if isinstance(found, str):
            return found
        storage.update(found.id, {"enabled": enabled})
        self._after_mutation()
        verb = "resumed" if enabled else "paused"
        return f"{verb.title()} job “{found.name}” ({found.id})."

    def _remove_job(self, storage: JobStorage, tokens: list[str]) -> str:
        found = self._find_job(storage, tokens)
        if isinstance(found, str):
            return found
        storage.remove(found.id)
        self._after_mutation()
        return f"Removed job “{found.name}” ({found.id})."

    def _cleanup(self, storage: JobStorage) -> str:
        disabled = [job for job in storage.all() if not job.enabled]
        for job in disabled:
            storage.remove(job.id)
        self._after_mutation()
        if not disabled:
            return "No disabled jobs to clean up."
        names = ", ".join(job.name for job in disabled)
        return f"Removed {len(disabled)} disabled job(s): {names}"

    def _find_job(self, storage: JobStorage, tokens: list[str]) -> ScheduleJob | str:
        """Resolve a job by id or name; returns an error string when ambiguous."""
        if not tokens:
            return f"Which job? Usage: /{COMMAND_NAME} <pause|resume|remove> <id or name>"
        needle = " ".join(tokens)
        jobs = storage.all()
        for job in jobs:
            if job.id == needle:
                return job
        exact = [job for job in jobs if job.name == needle]
        loose = [job for job in jobs if needle.lower() in job.name.lower()]
        matches = exact or loose
        if not matches:
            return f"No scheduled job matches “{needle}”. Try /{COMMAND_NAME} list."
        if len(matches) > 1:
            names = ", ".join(f"{job.name} ({job.id})" for job in matches)
            return f"Multiple jobs match “{needle}”: {names}"
        return matches[0]

    def _after_mutation(self) -> None:
        """Re-evaluate due jobs and repaint immediately."""
        scheduler = self._scheduler
        if scheduler is not None:
            scheduler.wake()
        self._on_change()


def setup(tau: ExtensionAPI) -> None:
    """Tau's extension entry point (synchronous, one call per load generation)."""
    ScheduleExtension(tau).register()


__all__ = ["COMMAND_DESCRIPTION", "COMMAND_NAME", "COMMAND_USAGE", "ScheduleExtension", "setup"]
