"""Two-layer settings: global defaults plus per-project overrides.

- Global:  ``<TAU_HOME>/schedule-prompts-settings.json`` (hand-edited defaults,
  never written by the extension)
- Project: ``<cwd>/.tau/schedule-prompts-settings.json`` (written by the UI)

The project file only ever stores the keys the user actually changed, so global
defaults keep bleeding through for everything else.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .types import JobScope

SETTINGS_FILENAME = "schedule-prompts-settings.json"


@dataclass(frozen=True, slots=True)
class ScheduleSettings:
    """Effective settings for one session."""

    widget_visible: bool = True
    default_job_scope: JobScope = "session"


def global_settings_path(tau_home: Path) -> Path:
    """Return the user-level settings file for a Tau home."""
    return tau_home / SETTINGS_FILENAME


def project_settings_path(cwd: Path) -> Path:
    """Return the project-level settings file for a working directory."""
    return cwd / ".tau" / SETTINGS_FILENAME


def load_settings(cwd: Path, tau_home: Path) -> ScheduleSettings:
    """Merge global and project settings; project wins."""
    merged = {
        **_read(global_settings_path(tau_home)),
        **_read(project_settings_path(cwd)),
    }
    return _settings(merged)


def save_settings(cwd: Path, change: dict[str, Any]) -> bool:
    """Apply a partial change to the *project* file; False on IO failure.

    Reading the project file (not the merged view) before writing keeps it a
    pure list of deliberate overrides.
    """
    path = project_settings_path(cwd)
    try:
        merged = {**_read(path), **change}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"tau-schedule-prompt: ignoring malformed settings at {path}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if key in _KNOWN_KEYS}


_KNOWN_KEYS = ("widget_visible", "visible", "default_job_scope", "defaultJobScope")


def _settings(values: dict[str, Any]) -> ScheduleSettings:
    settings = ScheduleSettings()
    visible = values.get("widget_visible", values.get("visible"))
    if isinstance(visible, bool):
        settings = replace(settings, widget_visible=visible)
    scope = values.get("default_job_scope", values.get("defaultJobScope"))
    if scope in ("session", "workdir"):
        settings = replace(settings, default_job_scope=scope)
    return settings


__all__ = [
    "SETTINGS_FILENAME",
    "ScheduleSettings",
    "global_settings_path",
    "load_settings",
    "project_settings_path",
    "save_settings",
]
