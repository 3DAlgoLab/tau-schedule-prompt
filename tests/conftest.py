"""Shared fixtures for the extension's tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio (mirrors Tau's own test suite)."""
    return "asyncio"


@pytest.fixture
def frozen_now() -> datetime:
    """A fixed aware UTC instant for deterministic schedule maths."""
    return datetime(2026, 2, 13, 15, 30, 0, tzinfo=UTC)


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Path]:
    """An isolated working directory for storage/settings tests."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    yield cwd


def shift(moment: datetime, **kwargs: int) -> datetime:
    """Return ``moment`` shifted by the given timedelta keywords."""
    return moment + timedelta(**kwargs)
