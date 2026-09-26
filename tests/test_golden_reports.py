"""Golden reports: every audit's full JSON output, pinned for every committed stage.

Refactors of the audits must not change their reports. Unit tests pin single
behaviours; these pin everything at once, in key order, for the fixture corpus
and for ``fixtures/golden_corpus``, which exercises branches the regression
fixtures do not.

To accept an intended report change, regenerate and review the diff::

    USD_AUDIT_UPDATE_GOLDEN=1 uv run pytest tests/test_golden_reports.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from usd_scene_audit import geometry, names_hierarchy, scene

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"
VOLATILE_KEYS = {"elapsed_seconds", "phase_timings"}
PREFIX_PATTERN = r"[A-Z][A-Za-z0-9]*"
# Stages that are only ever loaded as a sublayer or reference of another stage.
NOT_ROOT_STAGES = {"assets_sub.usda"}

STAGES = sorted(
    path
    for path in list(FIXTURES.glob("*.usda")) + list((FIXTURES / "golden_corpus").glob("*.usda"))
    if path.name not in NOT_ROOT_STAGES
)

# The NumPy engine is pinned: CI installs the Numba extra, where "auto" would
# resolve differently. Engine parity is tested separately in test_geometry.py.
AUDITS = {
    "geometry": lambda path: geometry.analyze(path, geometry_engine="numpy"),
    "geometry_fast_cached": lambda path: geometry.analyze(
        path, geometry_engine="numpy", audit_mode="fast", mesh_cache_mode="face-hash"
    ),
    "geometry_standard_frame1": lambda path: geometry.analyze(
        path, geometry_engine="numpy", audit_mode="standard", frame=1.0
    ),
    "names": lambda path: names_hierarchy.analyze(path),
    "names_prefix": lambda path: names_hierarchy.analyze(path, prefix_style_pattern=PREFIX_PATTERN),
    "scene": lambda path: scene.analyze(path),
    "scene_prefix": lambda path: scene.analyze(path, prefix_style_pattern=PREFIX_PATTERN),
}


def _portable(report: dict) -> dict:
    """Drop timings and make machine-specific paths comparable across checkouts and platforms."""
    text = json.dumps({key: value for key, value in report.items() if key not in VOLATILE_KEYS})
    text = text.replace("\\\\", "/")
    root = re.escape(FIXTURES.resolve().as_posix())
    text = re.sub(root, "<fixtures>", text, flags=re.IGNORECASE if os.name == "nt" else 0)
    return json.loads(text)


def _golden_path(stage: Path) -> Path:
    return GOLDEN / f"{stage.relative_to(FIXTURES).as_posix().replace('/', '__')}.json"


@pytest.mark.parametrize("stage", STAGES, ids=lambda path: path.relative_to(FIXTURES).as_posix())
def test_reports_match_golden(stage: Path) -> None:
    actual = {name: _portable(run(stage)) for name, run in AUDITS.items()}
    golden = _golden_path(stage)

    if os.environ.get("USD_AUDIT_UPDATE_GOLDEN"):
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(json.dumps(actual, indent=1) + "\n", encoding="utf-8")

    expected = json.loads(golden.read_text(encoding="utf-8"))
    assert actual == expected
    # Equal dicts can still differ in key order, which archived report diffs see.
    assert json.dumps(actual, indent=1) == json.dumps(expected, indent=1)


def test_every_golden_file_has_a_stage() -> None:
    """A deleted or renamed fixture must not leave a stale golden file behind."""
    expected = {_golden_path(stage).name for stage in STAGES}

    assert {path.name for path in GOLDEN.glob("*.json")} == expected
