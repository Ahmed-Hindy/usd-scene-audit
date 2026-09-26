"""Reports must be identical across runs of the same stage.

Reports are archived and diffed across runs, so any run-to-run variation looks
like a change to the asset. Three sources of variation have been fixed:

- Python randomises ``str`` hashing per process, so report lists built by
  iterating a ``set`` changed order, and past the example cap, contents.
- ``stage.GetUsedLayers()`` returns layers in memory-address order.
- OpenUSD numbers instancing prototypes ``/__Prototype_N`` in an order that
  changes on every open, even with the same hash seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pxr import Usd

from usd_scene_audit import geometry, scene
from usd_scene_audit.geometry import PrototypePaths

MATERIAL_COUNT = 25
SUBLAYER_COUNT = 12
ASSET_COUNT = 30
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


def _asset(index: int) -> str:
    """One instanceable library asset with findings for every audit."""
    return f"""    def Xform "Asset{index:02d}"
    {{
        def Material "Mat{index:02d}_"
        {{
        }}
        def Mesh "Body_"
        {{
            point3f[] points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 1]
        }}
        def Xform "part"
        {{
        }}
        def Xform "Part"
        {{
        }}
    }}
"""


def _write_instanced_stage(root: Path) -> Path:
    """Write many instanced assets, plus a nested instance, whose prototype numbering varies per open."""
    library = "".join(_asset(index) for index in range(ASSET_COUNT))
    library += """    def Xform "Inner"
    {
        def Material "InnerMat"
        {
        }
    }
    def Xform "Outer"
    {
        def Xform "In0" (instanceable = true
            prepend references = </Lib/Inner>)
        {
        }
        def Xform "In1" (instanceable = true
            prepend references = </Lib/Inner>)
        {
        }
    }
"""
    instances = "".join(
        f'    def Xform "A{index:02d}_{copy}" (instanceable = true\n'
        f"        prepend references = </Lib/Asset{index:02d}>)\n    {{\n    }}\n"
        for index in range(ASSET_COUNT)
        for copy in range(2)
    )
    instances += "".join(
        f'    def Xform "Outer{copy}" (instanceable = true\n        prepend references = </Lib/Outer>)\n    {{\n    }}\n'
        for copy in range(2)
    )
    path = root / "instanced.usda"
    path.write_text(
        f'#usda 1.0\ndef Xform "World"\n{{\n{instances}}}\ndef Xform "Lib" (active = false)\n{{\n{library}}}\n',
        encoding="utf-8",
    )
    return path


def _run_cli(module: str, stage: Path, out: Path, hash_seed: int) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    result = subprocess.run(
        [sys.executable, "-m", f"usd_scene_audit.{module}", str(stage), "--json-out", str(out)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    return {key: value for key, value in report.items() if key not in VOLATILE_KEYS}


@pytest.mark.parametrize("module", ["scene", "names_hierarchy", "geometry"])
def test_report_is_identical_across_runs(tmp_path: Path, module: str) -> None:
    """Two runs of the same instanced stage, in different processes and hash seeds, must match.

    With 30 prototypes, OpenUSD's per-open numbering makes a collision between
    two runs vanishingly unlikely, so every module fails this without stable
    prototype handling.
    """
    stage = _write_instanced_stage(tmp_path)

    first = _run_cli(module, stage, tmp_path / "run0.json", hash_seed=0)
    second = _run_cli(module, stage, tmp_path / "run1.json", hash_seed=1)

    assert first == second


def test_prototype_contents_are_reported_under_their_first_instance(tmp_path: Path) -> None:
    """Prototype paths never reach the report; the sorted-first instance path does, nested too."""
    report = scene.analyze(_write_instanced_stage(tmp_path))
    materials = report["materials"]["materials_without_surface_output"]

    assert materials == [f"/World/A{index:02d}_0/Mat{index:02d}_" for index in range(ASSET_COUNT)] + [
        "/World/Outer0/In0/InnerMat"
    ]
    assert report["naming"]["case_collision_names"][0] == "/World/A00_0: Part, part"
    assert "__Prototype_" not in json.dumps(report)

    worst = geometry.analyze(tmp_path / "instanced.usda", 1e-12, 1e6, 1e-4)["worst_meshes"]
    assert worst[0]["path"] == "/World/A00_0/Body_"
    assert worst[0]["normalized_path"] == "/<prototype>/Body_"


def test_prototype_labels_resolve_to_valid_instance_proxies(tmp_path: Path) -> None:
    """Every stable name is a real, addressable instance-proxy path on the stage."""
    stage = Usd.Stage.Open(str(_write_instanced_stage(tmp_path)))
    prototype_paths = PrototypePaths(stage)

    labels = [prototype_paths.label(str(prototype.GetPath())) for prototype in prototype_paths.ordered()]

    assert labels == sorted(labels)
    assert len(set(labels)) == len(labels) == len(prototype_paths)
    for label in labels:
        assert stage.GetPrimAtPath(label).IsInstance() or stage.GetPrimAtPath(label).IsInstanceProxy()


def test_materials_without_surface_output_follow_traversal_order(tmp_path: Path) -> None:
    """The list is reported in stage traversal order, not set order."""
    report = scene.analyze(_write_stage(tmp_path))

    assert report["materials"]["materials_without_surface_output"] == [
        f"/Mat{index:02d}" for index in range(MATERIAL_COUNT)
    ]
    assert report["materials"]["materials_without_surface_output_count"] == MATERIAL_COUNT
    assert report["materials"]["unbound_material_prim_count"] == MATERIAL_COUNT


@pytest.mark.expects_check_errors
def test_layers_are_walked_in_identifier_order(tmp_path: Path, monkeypatch) -> None:
    """GetUsedLayers() order varies per run; check_errors must follow sorted layer identifiers."""
    stage = _write_stage(tmp_path)

    def failing_walk(layer, error_log=None):
        error_log.record("authored_asset_paths", layer.identifier, RuntimeError("boom"))
        return set()

    monkeypatch.setattr(scene, "authored_asset_paths", failing_walk)

    subjects = [entry["subject"] for entry in scene.analyze(stage)["check_errors"]["examples"]]
    file_backed = [subject for subject in subjects if not subject.startswith("anon:")]
    anonymous = [subject for subject in subjects if subject.startswith("anon:")]

    assert len(file_backed) == SUBLAYER_COUNT + 1
    assert file_backed == sorted(file_backed)
    assert subjects == file_backed + anonymous
