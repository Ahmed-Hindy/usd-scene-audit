"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from usd_scene_audit.geometry import CheckErrorLog

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def fail_on_unexpected_check_errors(request, monkeypatch):
    """Fail any test during which a check silently failed and was recorded.

    The audits record a failing check in CheckErrorLog and carry on, so a bug in
    a check makes it report nothing rather than raise. A "valid data yields no
    findings" test would then pass for the wrong reason. Tests that make a check
    fail on purpose opt out with ``@pytest.mark.expects_check_errors``.
    """
    if request.node.get_closest_marker("expects_check_errors"):
        yield
        return
    recorded: list[str] = []
    real_record = CheckErrorLog.record

    def spy(self, check, subject, error):
        recorded.append(f"{check} on {subject}: {type(error).__name__}: {error}")
        return real_record(self, check, subject, error)

    monkeypatch.setattr(CheckErrorLog, "record", spy)
    yield
    if recorded:
        pytest.fail("checks failed and were recorded to CheckErrorLog:\n  " + "\n  ".join(recorded), pytrace=False)


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
