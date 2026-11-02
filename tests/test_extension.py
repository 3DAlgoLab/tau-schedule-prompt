"""Wiring tests for the Tau extension entry point (``extension.py``).

These exercise the seams Pi keeps in ``index.ts``: registrations, the session
lifecycle that starts and stops the scheduler, the widget paint, and how a
firing reaches the conversation. The host is faked — the fake only implements
what the extension actually touches — so the tests stay deterministic and never
boot a TUI or a provider. The real scheduler, storage, and tool code are used.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tau_schedule_prompt import subagent as subagent_module
from tau_schedule_prompt.extension import (
    COMMAND_NAME,
    ScheduleExtension,
    setup,
)
from tau_schedule_prompt.rendering import CUSTOM_TYPE
from tau_schedule_prompt.storage import JobStorage
from tau_schedule_prompt.tool import TOOL_NAME
from tau_schedule_prompt.types import ScheduleJob
from tau_schedule_prompt.widget import WIDGET_KEY

NOW = datetime(2026, 2, 13, 15, 30, tzinfo=UTC)


@dataclass
class FakeComponents:
    """Records slot-widget mounts; mirrors the host's ``set_slot_widget`` seam."""

    mounts: list[tuple[str, list[str] | None]] = field(default_factory=list)
    supports_components: bool = True

    def set_slot_widget(
        self, key: str, content: list[str] | None, *, placement: str = "above_prompt"
    ) -> None:
        assert placement == "below_prompt"
        self.mounts.append((key, list(content) if content is not None else None))

    @property
    def last(self) -> list[str] | None:
        return self.mounts[-1][1] if self.mounts else None


@dataclass
class FakeUi:
    components: FakeComponents = field(default_factory=FakeComponents)
    notices: list[tuple[str, str]] = field(default_factory=list)

    def notify(self, message: str, level: str = "info") -> None:
        self.notices.append((message, level))


@dataclass
class FakeChoice:
    provider_name: str
    model: str


@dataclass
class FakeContext:
    """The slice of ``ExtensionContext`` this extension reads."""

    cwd: Path
    session_id: str | None = "sess-1"
    home: Path = Path("/tmp/tau-home")
    is_running: bool = False
    transcript: tuple[Any, ...] = ()
    ui: FakeUi = field(default_factory=FakeUi)
    provider_name: str = "fake"
    model: str = "fake-model"
    project_trusted: bool = True
    available_model_choices: tuple[FakeChoice, ...] = (FakeChoice("fake", "fake-model"),)

    @property
    def paths(self) -> Any:
        return type("Paths", (), {"home": self.home})()


@dataclass
class FakeApi:
    """Records every registration and delivery the extension performs."""

    context: FakeContext
    tools: list[Any] = field(default_factory=list)
    renderers: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, Any] = field(default_factory=dict)
    subscriptions: list[str] = field(default_factory=list)
    deliveries: list[dict[str, Any]] = field(default_factory=list)

    def register_tool(self, tool: Any) -> None:
        self.tools.append(tool)

    def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
        self.renderers[custom_type] = renderer

    def register_command(self, name: str, handler: Any, **kwargs: Any) -> None:
        del kwargs
        self.commands[name] = handler

    def on(self, event: str, handler: Any) -> Any:
        self.subscriptions.append(event)
        return handler

    def send_custom_message(
        self,
        content: str,
        *,
        custom_type: str,
        details: dict[str, Any] | None = None,
        deliver_as: str = "follow_up",
        trigger_turn: bool = True,
    ) -> None:
        self.deliveries.append(
            {
                "content": content,
                "custom_type": custom_type,
                "details": details or {},
                "deliver_as": deliver_as,
                "trigger_turn": trigger_turn,
            }
        )


def job(**overrides: Any) -> ScheduleJob:
    """A due-every-minute cron job, shifted to run under the frozen clock."""
    payload: dict[str, Any] = {
        "id": "job-1",
        "name": "heartbeat",
        "schedule": "* * * * * *",
        "prompt": "check the build",
        "type": "cron",
        "enabled": True,
        "created_at": "2026-02-13T15:00:00+00:00",
        "run_count": 0,
    }
    payload.update(overrides)
    return ScheduleJob(**payload)


def seed(cwd: Path, jobs: list[ScheduleJob]) -> JobStorage:
    storage = JobStorage(cwd)
    storage.save(jobs)
    return storage


def build(project: Path, context: FakeContext | None = None) -> tuple[ScheduleExtension, FakeApi]:
    api = FakeApi(context or FakeContext(cwd=project))
    setup(api)
    return ScheduleExtension(api), api


# -- registration ------------------------------------------------------------


def test_setup_registers_tool_renderer_and_command(project: Path) -> None:
    _, api = build(project)

    assert [tool.name for tool in api.tools] == [TOOL_NAME]
    assert list(api.renderers) == [CUSTOM_TYPE]
    assert list(api.commands) == [COMMAND_NAME]
    assert api.subscriptions == ["session_start", "session_shutdown"]


@pytest.mark.anyio
async def test_inline_fire_delivers_the_prompt_once(project: Path) -> None:
    extension, api = build(project)

    await extension._fire(job())  # noqa: SLF001 - exercising the scheduler's callback

    assert len(api.deliveries) == 1
    delivery = api.deliveries[0]
    assert delivery["content"] == "check the build"
    assert delivery["custom_type"] == CUSTOM_TYPE
    assert delivery["details"]["mode"] == "inline"
    assert delivery["deliver_as"] == "follow_up"


@pytest.mark.anyio
async def test_inline_fire_steers_an_running_session(project: Path) -> None:
    context = FakeContext(cwd=project, is_running=True)
    extension, api = build(project, context)

    await extension._fire(job())  # noqa: SLF001

    assert api.deliveries[0]["deliver_as"] == "steer"


@pytest.mark.anyio
async def test_subagent_fire_records_start_and_result(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension, api = build(project, FakeContext(cwd=project, session_id=None))
    monkeypatch.setattr(
        subagent_module, "default_session_factory", _fake_child_factory("all tests green")
    )

    await extension._fire(job(model="fake-model"))  # noqa: SLF001

    modes = [delivery["details"]["mode"] for delivery in api.deliveries]
    assert modes == ["subagent_start", "subagent_done"]
    assert api.deliveries[0]["trigger_turn"] is False
    assert "all tests green" in api.deliveries[1]["content"]
    # No notify flag: the result is recorded but must not wake the parent.
    assert api.deliveries[1]["trigger_turn"] is False


@pytest.mark.anyio
async def test_subagent_notify_wakes_the_parent(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension, api = build(project)
    monkeypatch.setattr(subagent_module, "default_session_factory", _fake_child_factory("done"))

    await extension._fire(job(model="fake-model", notify=True))  # noqa: SLF001

    assert api.deliveries[-1]["trigger_turn"] is True


@pytest.mark.anyio
async def test_subagent_failure_is_recorded_not_raised(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extension, api = build(project)
    monkeypatch.setattr(
        subagent_module,
        "default_session_factory",
        _fake_child_factory("", error="provider exploded"),
    )

    await extension._fire(job(model="fake-model"))  # noqa: SLF001

    assert api.deliveries[-1]["details"]["mode"] == "subagent_error"
    assert "provider exploded" in api.deliveries[-1]["content"]


def _fake_child_factory(reply: str, *, error: str = "") -> Any:
    """Return a ``SessionFactory`` standing in for a real child session."""

    class FakeChild:
        async def prompt(self, content: str, **kwargs: Any) -> AsyncIterator[Any]:
            del content, kwargs
            if error:
                raise RuntimeError(error)
            yield type("E", (), {"type": "message_update"})()

        def cancel(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        @property
        def last_assistant_text(self) -> str:
            return reply

    async def build(_request: Any) -> Any:
        return FakeChild()

    return build


# -- lifecycle and widget ----------------------------------------------------


@pytest.mark.anyio
async def test_session_start_paints_widget_and_shutdown_clears_it(project: Path) -> None:
    seed(project, [job()])
    extension, api = build(project)

    extension.on_session_start(None, api.context)
    painted = api.context.ui.components.last
    assert api.context.ui.components.mounts[-1][0] == WIDGET_KEY
    assert painted is not None and "heartbeat" in painted[1]

    await extension.on_session_shutdown(None, api.context)
    assert api.context.ui.components.last is None
    assert extension._scheduler is None  # noqa: SLF001


@pytest.mark.anyio
async def test_scheduler_loop_fires_and_repaints(project: Path) -> None:
    seed(project, [job()])
    context = FakeContext(cwd=project)
    extension, api = build(project, context)
    extension.on_session_start(None, context)
    scheduler = extension._scheduler  # noqa: SLF001
    assert scheduler is not None
    scheduler._watermark = NOW - timedelta(seconds=90)  # noqa: SLF001

    before = len(context.ui.components.mounts)
    await scheduler._tick()  # noqa: SLF001 - exercising the loop body directly
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert api.deliveries and api.deliveries[0]["content"] == "check the build"
    assert len(context.ui.components.mounts) > before
    assert JobStorage(project).get("job-1") is not None
    await extension.on_session_shutdown(None, context)


def test_widget_hidden_by_settings_is_not_mounted(project: Path) -> None:
    seed(project, [job()])
    context = FakeContext(cwd=project)
    (project / ".tau").mkdir(parents=True, exist_ok=True)
    (project / ".tau" / "schedule-prompts-settings.json").write_text(
        '{"widget_visible": false}', encoding="utf-8"
    )
    extension, api = build(project, context)

    _run_hidden_start(extension, context)

    assert context.ui.components.mounts == []


def _run_hidden_start(extension: ScheduleExtension, context: FakeContext) -> None:
    """Start and stop under a loop: Tau always emits ``session_start`` from async code."""

    async def drive() -> None:
        extension.on_session_start(None, context)
        await extension.on_session_shutdown(None, context)

    asyncio.run(drive())


# -- /schedule-prompt --------------------------------------------------------


def command_text(api: FakeApi, args: str) -> str:
    return str(api.commands[COMMAND_NAME](args, None))


def test_command_lists_jobs_with_schedule_and_state(project: Path) -> None:
    seed(project, [job()])
    _, api = build(project)

    text = command_text(api, "")

    assert "heartbeat" in text and "every second" in text and "pending" in text


def test_command_pause_resume_remove_and_cleanup(project: Path) -> None:
    seed(project, [job(), job(id="job-2", name="nightly", enabled=False)])
    extension, api = build(project)

    assert "Paused job “heartbeat”" in command_text(api, "pause heartbeat")
    assert JobStorage(project).get("job-1").enabled is False  # type: ignore[union-attr]
    assert "Resumed" in command_text(api, "resume job-1")
    assert "Removed job “heartbeat”" in command_text(api, "remove heartbeat")
    assert JobStorage(project).get("job-1") is None
    assert "Removed 1 disabled job" in command_text(api, "cleanup")
    assert JobStorage(project).all() == []
    assert "Unknown" in command_text(api, "nonsense")
    assert extension._scheduler is None  # noqa: SLF001 - the command path needs no loop


def test_command_reports_unknown_and_ambiguous_jobs(project: Path) -> None:
    seed(project, [job(name="build a"), job(id="job-2", name="build b")])
    _, api = build(project)

    assert "No scheduled job matches" in command_text(api, "remove zzz")
    assert "Multiple jobs match" in command_text(api, "pause build")
    assert "Which job?" in command_text(api, "pause")


def test_command_hide_persists_the_widget_choice(project: Path) -> None:
    _, api = build(project)

    assert "hidden" in command_text(api, "hide")

    settings_file = project_settings_file(api.context.cwd)
    assert '"widget_visible": false' in settings_file.read_text(encoding="utf-8")
    assert "shown" in command_text(api, "show")


def project_settings_file(cwd: Path) -> Path:
    return cwd / ".tau" / "schedule-prompts-settings.json"


# -- loop guard --------------------------------------------------------------


def test_recent_marker_blocks_scheduling_from_a_scheduled_run(project: Path) -> None:
    _, api = build(project)
    extension = ScheduleExtension(api)
    tool = api.tools[0]
    marker = type("Custom", (), {"custom_type": CUSTOM_TYPE, "role": "custom"})
    api.context.transcript = (marker(),)

    result = _run(
        tool.execute(
            "call-1",
            {"action": "add", "schedule": "+10s", "prompt": "loop?", "type": "once"},
        )
    )

    assert "✗ Error" in result.content[0].text
    assert extension.recent_scheduled_marker() is True


def test_add_binds_the_job_to_the_session_by_default(project: Path) -> None:
    _, api = build(project)
    tool = api.tools[0]

    result = _run(
        tool.execute(
            "call-1",
            {"action": "add", "schedule": "+30s", "prompt": "later", "type": "once"},
        )
    )

    stored = JobStorage(project).all()
    assert len(stored) == 1
    assert stored[0].session == "sess-1"
    assert "✓ Created job" in result.content[0].text


def _run(coroutine: Any) -> Any:
    """Drive one tool call to completion without an event-loop fixture."""
    return asyncio.run(coroutine)
