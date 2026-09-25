"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the directory holding the committed .usda fixture corpus."""
    return FIXTURES


@pytest.fixture(scope="session")
def stage_path() -> Callable[[str], Path]:
    """Return a resolver for named stages in the fixture corpus."""

    def resolve(name: str) -> Path:
        path = FIXTURES / name
        if not path.exists():
            available = ", ".join(sorted(p.name for p in FIXTURES.glob("*.usd*")))
            raise FileNotFoundError(f"missing test fixture {name!r}; available: {available}")
        return path

    return resolve
