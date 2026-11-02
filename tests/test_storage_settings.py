"""Tests for the job record model, storage, and settings layering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau_schedule_prompt.settings import (
    ScheduleSettings,
    global_settings_path,
    load_settings,
    project_settings_path,
    save_settings,
)
from tau_schedule_prompt.storage import JobStorage
from tau_schedule_prompt.types import ScheduleJob, job_from_json


def make_job(**overrides: object) -> ScheduleJob:
    """Build a valid job with optional field overrides."""
    data = {
        "id": "abc123",
        "name": "standup",
        "prompt": "summarize yesterday",
        "schedule": "0 0 9 * * *",
        "type": "cron",
        "session": "sess-1",
        "created_at": "2026-02-13T15:30:00+00:00",
    }
    data.update(overrides)
    return job_from_json(data)  # type: ignore[arg-type]


def test_job_round_trips_through_json() -> None:
    job = make_job(interval_ms=None, run_count=2, last_status="success")
    restored = job_from_json(job.to_json())
    assert restored is not None
    assert restored == job


def test_job_from_json_rejects_incomplete_rows() -> None:
    assert job_from_json({"id": "x"}) is None
    assert job_from_json(None) is None


def test_job_apply_updates_validates_keys() -> None:
    job = make_job()
    with pytest.raises(KeyError):
        job.apply_updates({"nope": 1})
    assert job.apply_updates({"enabled": False}).enabled is False


def test_storage_round_trip_is_atomic_and_durable(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path)
    assert storage.all() == []
    storage.add(make_job())
    storage.add(make_job(id="def456", name="weekly"))
    assert [job.name for job in storage.all()] == ["standup", "weekly"]
    # No leftover temp files from the atomic write.
    leftovers = [p.name for p in (tmp_path / ".tau").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_storage_skips_corrupt_rows_and_duplicates(tmp_path: Path) -> None:
    store = tmp_path / ".tau" / "schedule-prompts.json"
    store.parent.mkdir(parents=True)
    store.write_text(
        """{"jobs": [
            {"id": "a", "name": "ok", "prompt": "p", "schedule": "* * * * * *",
             "type": "cron", "created_at": "2026-02-13T15:30:00+00:00"},
            {"id": "b"},
            {"id": "a", "name": "dupe", "prompt": "p", "schedule": "* * * * * *",
             "type": "cron", "created_at": "2026-02-13T15:30:00+00:00"}
        ]}""",
        encoding="utf-8",
    )
    jobs = JobStorage(tmp_path).all()
    assert [job.name for job in jobs] == ["ok"]


def test_storage_survives_unparseable_file(tmp_path: Path) -> None:
    store = tmp_path / ".tau" / "schedule-prompts.json"
    store.parent.mkdir(parents=True)
    store.write_text("{not json", encoding="utf-8")
    assert JobStorage(tmp_path).all() == []


def test_storage_update_and_remove(tmp_path: Path) -> None:
    storage = JobStorage(tmp_path)
    storage.add(make_job())
    updated = storage.update("abc123", {"enabled": False, "last_status": None})
    assert updated is not None and updated.enabled is False
    assert storage.update("missing", {"enabled": False}) is None
    assert storage.remove("abc123") is True
    assert storage.remove("abc123") is False
    assert storage.all() == []


def test_settings_default_to_visible_widget_and_session_scope(tmp_path: Path) -> None:
    settings = load_settings(tmp_path, tmp_path / "tau-home")
    assert settings == ScheduleSettings(widget_visible=True, default_job_scope="session")


def test_project_settings_override_global(tmp_path: Path) -> None:
    tau_home = tmp_path / "tau-home"
    tau_home.mkdir()
    global_settings_path(tau_home).write_text(
        '{"widget_visible": false, "default_job_scope": "workdir"}', encoding="utf-8"
    )
    settings = load_settings(tmp_path, tau_home)
    assert settings == ScheduleSettings(widget_visible=False, default_job_scope="workdir")

    save_settings(tmp_path, {"widget_visible": True})
    merged = load_settings(tmp_path, tau_home)
    assert merged.widget_visible is True
    assert merged.default_job_scope == "workdir"
    # The project file only stores the overridden key.
    assert json.loads(project_settings_path(tmp_path).read_text()) == {"widget_visible": True}


def test_settings_ignore_unknown_and_malformed_values(tmp_path: Path) -> None:
    tau_home = tmp_path / "tau-home"
    tau_home.mkdir()
    global_settings_path(tau_home).write_text(
        '{"bogus": 1, "widget_visible": "no"}', encoding="utf-8"
    )
    assert load_settings(tmp_path, tau_home) == ScheduleSettings()
    project_settings_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    project_settings_path(tmp_path).write_text("garbage", encoding="utf-8")
    assert load_settings(tmp_path, tau_home) == ScheduleSettings()
