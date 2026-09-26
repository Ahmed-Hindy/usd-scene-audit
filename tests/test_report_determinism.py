"""Reports must not depend on the per-process string hash seed.

Reports are archived and diffed across runs. Python randomises ``str`` hashing
per process, so any report list built by iterating a ``set`` changed order --
and, past the example cap, changed contents -- between two runs of the same
stage.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from usd_scene_audit import scene

MATERIAL_COUNT = 25
SUBLAYER_COUNT = 12
VOLATILE_KEYS = {"elapsed_seconds", "phase_timings"}


def _write_stage(root: Path) -> Path:
    """Write a stage with many surface-less materials and many sublayers."""
    for index in range(SUBLAYER_COUNT):
        (root / f"sub{index}.usda").write_text(f'#usda 1.0\ndef "Layer{index}"\n{{\n}}\n', encoding="utf-8")
    sublayers = ", ".join(f"@./sub{index}.usda@" for index in reversed(range(SUBLAYER_COUNT)))
    materials = "".join(f'def Material "Mat{index:02d}"\n{{\n}}\n' for index in range(MATERIAL_COUNT))
    path = root / "stage.usda"
    path.write_text(f"#usda 1.0\n(\n    subLayers = [{sublayers}]\n)\n{materials}", encoding="utf-8")
    return path


def _run_cli(module: str, stage: Path, out: Path, hash_seed: int) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    subprocess.run(
        [sys.executable, "-m", f"usd_scene_audit.{module}", str(stage), "--json-out", str(out)],
        check=True,
        capture_output=True,
        env=env,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    return {key: value for key, value in report.items() if key not in VOLATILE_KEYS}


@pytest.mark.parametrize("module", ["scene", "names_hierarchy", "geometry"])
def test_report_is_identical_across_hash_seeds(tmp_path: Path, module: str) -> None:
    """The same stage must produce the same report under different hash seeds."""
    stage = _write_stage(tmp_path)

    first = _run_cli(module, stage, tmp_path / "seed0.json", hash_seed=0)
    second = _run_cli(module, stage, tmp_path / "seed1.json", hash_seed=1)

    assert first == second


def test_materials_without_surface_output_follow_traversal_order(tmp_path: Path) -> None:
    """The list is reported in stage traversal order, not set order."""
    report = scene.analyze(_write_stage(tmp_path))

    assert report["materials"]["materials_without_surface_output"] == [
        f"/Mat{index:02d}" for index in range(MATERIAL_COUNT)
    ]
    assert report["materials"]["materials_without_surface_output_count"] == MATERIAL_COUNT
    assert report["materials"]["unbound_material_prim_count"] == MATERIAL_COUNT
