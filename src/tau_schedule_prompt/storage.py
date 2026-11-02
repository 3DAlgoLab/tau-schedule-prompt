"""Atomic job persistence for ``<cwd>/.tau/schedule-prompts.json``.

The file is deliberately plain JSON so it can be hand-edited — removing a
job's ``session`` key, for example, promotes it to a project-wide schedule.
Writes go through a temp file plus ``os.replace`` so a crash mid-save cannot
truncate the store, and reads never raise: a corrupt file degrades to an empty
store with a diagnostic on stderr, matching Pi's behaviour.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from .types import ScheduleJob, job_from_json

STORE_DIRNAME = ".tau"
STORE_FILENAME = "schedule-prompts.json"
STORE_VERSION = 1


class JobStorage:
    """Read/write the project-local scheduled-prompt store."""

    def __init__(self, cwd: Path) -> None:
        self.store_dir = cwd / STORE_DIRNAME
        self.store_path = self.store_dir / STORE_FILENAME

    def load(self) -> list[ScheduleJob]:
        """Return every stored job, skipping rows that fail validation."""
        try:
            if not self.store_path.exists():
                return []
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"tau-schedule-prompt: ignoring unreadable store {self.store_path}: {exc}",
                file=sys.stderr,
            )
            return []
        if not isinstance(raw, dict):
            return []
        jobs_raw = raw.get("jobs")
        if not isinstance(jobs_raw, list):
            return []
        jobs: list[ScheduleJob] = []
        seen: set[str] = set()
        for entry in jobs_raw:
            job = job_from_json(entry)
            if job is None or job.id in seen:
                continue
            seen.add(job.id)
            jobs.append(job)
        return jobs

    def save(self, jobs: list[ScheduleJob]) -> None:
        """Persist ``jobs`` atomically (temp file plus rename)."""
        payload = {"jobs": [job.to_json() for job in jobs], "version": STORE_VERSION}
        self.store_dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(self.store_dir), prefix=f"{STORE_FILENAME}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
            os.replace(temp_name, self.store_path)
        except OSError:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
            raise

    def has_job_named(self, name: str) -> bool:
        """Return whether any stored job already uses ``name``."""
        return any(job.name == name for job in self.load())

    def add(self, job: ScheduleJob) -> None:
        """Append ``job`` and persist."""
        jobs = self.load()
        jobs.append(job)
        self.save(jobs)

    def remove(self, job_id: str) -> bool:
        """Delete the job with ``job_id``; return whether it existed."""
        jobs = self.load()
        remaining = [job for job in jobs if job.id != job_id]
        if len(remaining) == len(jobs):
            return False
        self.save(remaining)
        return True

    def update(self, job_id: str, updates: dict[str, object]) -> ScheduleJob | None:
        """Apply ``updates`` to one job and persist; ``None`` when absent."""
        jobs = self.load()
        for job in jobs:
            if job.id != job_id:
                continue
            job.apply_updates(updates)
            self.save(jobs)
            return job
        return None

    def get(self, job_id: str) -> ScheduleJob | None:
        """Return one job by id."""
        for job in self.load():
            if job.id == job_id:
                return job
        return None

    def all(self) -> list[ScheduleJob]:
        """Return every job."""
        return self.load()


__all__ = ["JobStorage", "STORE_FILENAME", "STORE_VERSION"]
