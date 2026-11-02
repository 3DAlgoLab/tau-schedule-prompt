"""Run one scheduled prompt inside a fresh, in-process child session.

A job with ``model`` set runs out-of-band: instead of injecting the prompt into
the session that scheduled it, Tau builds a child ``CodingSession`` (the same
seam the external ``tau-subagents`` extension uses, see
``dev-notes/child-session-seam.md`` in the Tau repo), runs the prompt to
completion there, and reports the assistant's reply back to the parent.

The child is deliberately hermetic:

- in-memory session storage (nothing to clean up on disk),
- the parent's project-trust decision forwarded via ``trust_override`` so a
  trusted repo keeps its ``AGENTS.md``/skills instead of silently losing them,
- core coding tools only, unless the job opts into ``extensions``,
- a session id carrying :data:`SUBAGENT_SESSION_PREFIX`, which is how the tool
  refuses to schedule from inside a scheduled run (the guard survives the
  extension being loaded twice, once per session).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .types import ScheduleJob

SUBAGENT_SESSION_PREFIX = "tau-schedule-sub"
CORE_TOOL_NAMES = frozenset({"read", "write", "edit", "bash"})
OUTPUT_SNIPPET_LIMIT = 500


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A provider/model pair the parent session can actually call."""

    provider_name: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider_name}/{self.model}"


@dataclass(frozen=True, slots=True)
class ParentSnapshot:
    """Context values read on the session thread before any background work.

    ``ExtensionContext`` properties assert the extension generation is still
    active, so a background task that outlives a ``/reload`` would raise. The
    extension snapshots everything the runner needs instead of holding the
    context.
    """

    cwd: Path
    tau_home: Path
    provider_name: str
    model: str
    project_trusted: bool
    model_choices: tuple[ModelRef, ...]


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """Outcome of one child-session run."""

    ok: bool
    text: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class ChildSessionRequest:
    """Everything the factory needs to build one child session."""

    cwd: Path
    tau_home: Path
    model: str
    provider_name: str
    session_id: str
    project_trusted: bool
    extensions_enabled: bool
    skills_enabled: bool
    allowed_tool_names: frozenset[str] | None


class ChildSession(Protocol):
    """The slice of ``CodingSession`` the runner uses (test seam)."""

    def prompt(self, content: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Run one prompt, yielding session events until the agent settles."""
        ...

    def cancel(self) -> None:
        """Interrupt the running agent."""
        ...

    async def aclose(self) -> None:
        """Release the session's provider and task resources."""
        ...

    @property
    def last_assistant_text(self) -> str | None:
        """Return the last assistant message's text, if any."""
        ...


SessionFactory = Callable[[ChildSessionRequest], Awaitable[ChildSession]]
ActivityCallback = Callable[[str], None]


def new_child_session_id() -> str:
    """Return a recognisable child session id."""
    return f"{SUBAGENT_SESSION_PREFIX}-{uuid.uuid4().hex[:8]}"


def is_subagent_session(session_id: str | None) -> bool:
    """Return whether ``session_id`` belongs to a scheduled-run child session."""
    return bool(session_id) and str(session_id).startswith(SUBAGENT_SESSION_PREFIX)


def resolve_model_ref(choices: Sequence[ModelRef], needle: str) -> ModelRef | None:
    """Find the model a job asks for, mirroring Pi's fuzzy resolution ladder.

    ``provider/model`` tries an exact provider+id match first, then falls back
    to matching the id portion alone (model ids never contain the provider
    prefix). A bare needle is matched by exact id, then id substring, then
    nothing else.
    """
    text = needle.strip()
    if not text:
        return None
    if "/" in text:
        provider_name, _, model_id = text.partition("/")
        for choice in choices:
            if choice.provider_name == provider_name and choice.model == model_id:
                return choice
        text = model_id
    lowered = text.lower()
    for choice in choices:
        if choice.model.lower() == lowered:
            return choice
    for choice in choices:
        if lowered in choice.model.lower():
            return choice
    return None


def describe_available_models(choices: Sequence[ModelRef], *, limit: int = 5) -> str:
    """Render a short "Available: ..." hint for an unknown-model error."""
    if not choices:
        return "No models with configured auth."
    sample = ", ".join(str(choice) for choice in choices[:limit])
    more = f", … ({len(choices) - limit} more)" if len(choices) > limit else ""
    return f"Available: {sample}{more}"


def snippet(text: str, limit: int = OUTPUT_SNIPPET_LIMIT) -> str:
    """Clip long subagent output/errors to a transcript-friendly length."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _extensions_enabled(job: ScheduleJob) -> bool:
    """Return whether a job asked for the parent's extensions in its child."""
    value = job.extensions
    return value is True or (isinstance(value, (list, tuple)) and len(value) > 0)


def _skills_enabled(job: ScheduleJob) -> bool:
    """Return whether a job asked for skills in its child."""
    value = job.skills
    return value is True or (isinstance(value, (list, tuple)) and len(value) > 0)


async def run_subagent(
    job: ScheduleJob,
    parent: ParentSnapshot,
    *,
    session_factory: SessionFactory | None = None,
    on_activity: ActivityCallback | None = None,
) -> SubagentResult:
    """Run ``job.prompt`` in a child session pinned to ``job.model``.

    Never raises: a failure comes back as ``SubagentResult(ok=False, error=...)
    so the scheduler can record it and keep the loop alive. Cancellation (the
    session shutting down mid-run) is the one exception — it propagates so the
    scheduler can await teardown.
    """
    requested = job.model or ""
    target = resolve_model_ref(parent.model_choices, requested)
    if target is None:
        return SubagentResult(
            ok=False,
            error=(
                f"Unknown model '{requested}'. {describe_available_models(parent.model_choices)}"
            ),
        )

    build = session_factory or default_session_factory
    request = ChildSessionRequest(
        cwd=parent.cwd,
        tau_home=parent.tau_home,
        model=target.model,
        provider_name=target.provider_name,
        session_id=new_child_session_id(),
        project_trusted=parent.project_trusted,
        extensions_enabled=_extensions_enabled(job),
        skills_enabled=_skills_enabled(job),
        allowed_tool_names=None if _extensions_enabled(job) else CORE_TOOL_NAMES,
    )

    try:
        session = await build(request)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the job, not raised
        return SubagentResult(
            ok=False,
            error=f"Could not start a {target} subagent session: {exc}",
        )

    if on_activity is not None:
        on_activity(f"running on {target}")

    buffered: list[str] = []
    tool_calls = 0
    try:
        async for event in session.prompt(job.prompt, source="extension"):
            kind = getattr(event, "type", "")
            if kind == "message_start":
                message = getattr(event, "message", None)
                if getattr(message, "role", None) == "assistant":
                    buffered.clear()
            elif kind == "message_update":
                provider_event = getattr(event, "assistant_message_event", None)
                if getattr(provider_event, "type", "") == "text_delta":
                    buffered.append(str(getattr(provider_event, "delta", "")))
            elif kind == "tool_execution_start":
                tool_calls += 1
                if on_activity is not None:
                    on_activity(f"{tool_calls} tool call(s) on {target}")
        text = "".join(buffered).strip()
        if not text:
            text = (session.last_assistant_text or "").strip()
        return SubagentResult(ok=True, text=text)
    except asyncio.CancelledError:
        _request_cancel(session)
        raise
    except Exception as exc:  # noqa: BLE001 - provider/agent failures land in job state
        return SubagentResult(ok=False, error=str(exc) or exc.__class__.__name__)
    finally:
        await _close_quietly(session)


def _request_cancel(session: ChildSession) -> None:
    """Best-effort interrupt of a child session we are abandoning."""
    with contextlib.suppress(Exception):  # teardown cannot fail the fire
        session.cancel()


async def _close_quietly(session: ChildSession) -> None:
    """Close a child session even while the caller task is being cancelled."""
    try:
        await asyncio.shield(session.aclose())
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - teardown failures are not the job's fault
        pass


def default_session_factory(request: ChildSessionRequest) -> Awaitable[ChildSession]:
    """Build a real child ``CodingSession`` (the production path).

    Imports are local so importing this module never pulls the TUI/provider
    dependency graph into a test that injects its own factory.
    """
    from tau_agent.session.storage import InMemorySessionStorage
    from tau_coding.credentials import FileCredentialStore
    from tau_coding.paths import TauPaths
    from tau_coding.provider_config import ProviderConfigError, load_provider_settings
    from tau_coding.provider_runtime import create_model_provider
    from tau_coding.resources import TauResourcePaths
    from tau_coding.session import CodingSession, CodingSessionConfig

    async def build() -> ChildSession:
        settings = load_provider_settings()
        try:
            provider_config = settings.get_provider(request.provider_name)
        except ProviderConfigError as exc:
            raise RuntimeError(f"No provider configuration for '{request.provider_name}'") from exc
        provider = create_model_provider(
            provider_config,
            credential_store=FileCredentialStore(),
            model=request.model,
        )
        paths = TauPaths(home=request.tau_home)
        config = CodingSessionConfig(
            provider=provider,
            model=request.model,
            provider_name=request.provider_name,
            provider_settings=settings,
            storage=InMemorySessionStorage(),
            cwd=request.cwd,
            session_id=request.session_id,
            resource_paths=TauResourcePaths(
                root=paths.home,
                cwd=request.cwd,
                agents_root=paths.agents_home,
                paths=paths,
            ),
            extensions_enabled=request.extensions_enabled,
            skills_enabled=request.skills_enabled,
            allowed_tool_names=request.allowed_tool_names,
            # A headless child must not raise a trust prompt; it inherits the
            # parent's decision instead (Tau issue #729).
            trust_override="approve" if request.project_trusted else "decline",
            trust_interactive=False,
        )
        return await CodingSession.load(config)  # type: ignore[return-value]

    return build()


__all__ = [
    "CORE_TOOL_NAMES",
    "ActivityCallback",
    "ChildSession",
    "ChildSessionRequest",
    "ModelRef",
    "OUTPUT_SNIPPET_LIMIT",
    "ParentSnapshot",
    "SUBAGENT_SESSION_PREFIX",
    "SessionFactory",
    "SubagentResult",
    "default_session_factory",
    "describe_available_models",
    "is_subagent_session",
    "new_child_session_id",
    "resolve_model_ref",
    "run_subagent",
    "snippet",
]
