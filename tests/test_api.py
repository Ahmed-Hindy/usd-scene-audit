"""The Python API passes thresholds and options by name.

``geometry.analyze()`` used to take three adjacent float thresholds
positionally, and ``mesh_record()`` eleven positional arguments. Swapping two
of the floats was a silent behaviour change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
from pxr import Gf, Usd, UsdGeom

from usd_scene_audit import geometry, names_hierarchy, scene
from usd_scene_audit.geometry import MeshCheckSettings, mesh_record


def test_geometry_thresholds_are_keyword_only(stage_path) -> None:
    with pytest.raises(TypeError):
        geometry.analyze(stage_path("static_mesh_clean.usda"), 1e-12, 1e6, 1e-4)


@pytest.mark.parametrize("module", [names_hierarchy, scene])
def test_prefix_pattern_is_keyword_only(module, stage_path) -> None:
    with pytest.raises(TypeError):
        module.analyze(stage_path("static_mesh_clean.usda"), r"[A-Z]\w*")


def test_mesh_record_options_are_keyword_only(stage_path) -> None:
    stage = Usd.Stage.Open(str(stage_path("static_mesh_clean.usda")))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))

    with pytest.raises(TypeError):
        mesh_record(prim, MeshCheckSettings())


def test_library_and_cli_share_threshold_defaults(stage_path, monkeypatch) -> None:
    """analyze() with no options must audit exactly like the CLI with no flags."""
    report = geometry.analyze(stage_path("static_mesh_clean.usda"))
    assert report["thresholds"] == {
        "zero_area_epsilon": geometry.DEFAULT_ZERO_AREA_EPSILON,
        "huge_coord_threshold": geometry.DEFAULT_HUGE_COORD_THRESHOLD,
        "extent_tolerance": geometry.DEFAULT_EXTENT_TOLERANCE,
    }

    captured = {}
    monkeypatch.setattr(sys, "argv", ["usd-geometry-audit", str(stage_path("static_mesh_clean.usda"))])
    with mock.patch.object(geometry, "analyze", side_effect=lambda path, **kwargs: captured.update(kwargs) or report):
        geometry.main()

    assert captured["zero_area_epsilon"] == geometry.DEFAULT_ZERO_AREA_EPSILON
    assert captured["huge_coord_threshold"] == geometry.DEFAULT_HUGE_COORD_THRESHOLD
    assert captured["extent_tolerance"] == geometry.DEFAULT_EXTENT_TOLERANCE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"audit_mode": "thorough"}, "Unsupported audit mode"),
        ({"face_analysis_engine": "auto"}, "resolve 'auto'"),
    ],
)
def test_settings_reject_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        MeshCheckSettings(**kwargs)


def test_mesh_record_defaults_read_transforms_at_the_attribute_time(tmp_path: Path) -> None:
    """With no xform_cache, mesh_record() evaluates transforms at the same time code as attributes."""
    path = tmp_path / "flip.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetStartTimeCode(2)
    stage.SetEndTimeCode(2)
    parent = UsdGeom.Xform.Define(stage, "/World")
    scale = parent.AddScaleOp()
    scale.Set(Gf.Vec3f(1, 1, 1))
    scale.Set(Gf.Vec3f(-1, 1, 1), 2)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    record = mesh_record(stage.GetPrimAtPath("/World/Mesh"))

    assert record["issues"] == geometry.analyze(path)["summary_counts"] == {"negative_transform_determinant": 1}
