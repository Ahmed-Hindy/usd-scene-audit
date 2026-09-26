"""Regression tests for mesh attributes authored with the wrong value type.

``GetNormalsAttr().Get()`` and friends return whatever value type the layer
authored, not the schema type. A ``float[] normals`` or a scalar
``int faceVertexCounts`` used to raise inside the vectorized checks and abort
the whole audit. They must be reported as findings instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Usd, UsdGeom

from usd_scene_audit import geometry

GOOD_ATTRIBUTES = {
    "points": "point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]",
    "counts": "int[] faceVertexCounts = [3]",
    "indices": "int[] faceVertexIndices = [0, 1, 2]",
}


def _write_mesh(path: Path, name: str = "M", **overrides: str) -> None:
    lines = {**GOOD_ATTRIBUTES, **overrides}
    body = "\n    ".join(lines.values())
    path.write_text(f'#usda 1.0\ndef Mesh "{name}"\n{{\n    {body}\n}}\n', encoding="utf-8")


def _audit(path: Path) -> dict:
    return geometry.analyze(path)


@pytest.mark.parametrize(
    ("override", "finding", "type_name", "problem"),
    [
        ({"normals": "float[] normals = [0, 0, 1]"}, "normals_bad_shape", "float[]", {"shape": [3]}),
        ({"normals": "float3 normals = (0, 0, 1)"}, "normals_bad_shape", "float3", {"shape": [3]}),
        ({"points": "float[] points = [0, 0, 0, 1, 0, 0, 0, 1, 0]"}, "points_bad_shape", "float[]", {"shape": [9]}),
        ({"points": "float2[] points = [(0,0), (1,0), (0,1)]"}, "points_bad_shape", "float2[]", {"shape": [3, 2]}),
        ({"points": "point3f points = (0, 0, 0)"}, "points_bad_shape", "point3f", {"shape": [3]}),
        ({"counts": "int faceVertexCounts = 3"}, "face_vertex_counts_bad_shape", "int", {"shape": []}),
        (
            {"indices": "int2[] faceVertexIndices = [(0, 1), (2, 0)]"},
            "face_vertex_indices_bad_shape",
            "int2[]",
            {"shape": [2, 2]},
        ),
    ],
    ids=[
        "normals-float-array",
        "normals-scalar",
        "points-float-array",
        "points-float2",
        "points-scalar",
        "counts-scalar",
        "indices-int2",
    ],
)
def test_mistyped_attribute_is_a_finding_not_a_crash(tmp_path: Path, override, finding, type_name, problem) -> None:
    """Every one of these raised on main; each must now report a *_bad_shape finding."""
    path = tmp_path / "mesh.usda"
    _write_mesh(path, **override)

    report = _audit(path)

    assert report["summary_counts"][finding] == 1
    assert report["check_errors"]["count"] == 0
    record = report["worst_meshes"][0]
    detail = record["details"][finding]
    detail = detail[0] if isinstance(detail, list) else detail
    assert detail["authored_type"] == type_name
    assert detail["expected_type"] != detail["authored_type"]
    assert {key: detail[key] for key in problem} == problem


@pytest.mark.parametrize(
    "finding", ["points_bad_shape", "face_vertex_counts_bad_shape", "face_vertex_indices_bad_shape"]
)
def test_unusable_topology_counts_as_a_serious_failure(finding: str) -> None:
    """Unusable topology ranks with missing topology, not with the unregistered weight of 1."""
    record = {"issues": {finding: 1}}

    assert geometry.seriousness_score(record) == geometry.seriousness_score({"issues": {"missing_points": 1}})


def test_empty_arrays_are_not_bad_shapes(tmp_path: Path) -> None:
    """Empty arrays keep reporting as empty, whatever their authored type."""
    path = tmp_path / "mesh.usda"
    _write_mesh(path, points="float[] points = []", normals="normal3f[] normals = []")

    issues = _audit(path)["summary_counts"]

    assert "points_bad_shape" not in issues
    assert "normals_bad_shape" not in issues
    assert issues["empty_points"] == 1


def test_one_failing_mesh_does_not_abort_the_audit(tmp_path: Path, monkeypatch) -> None:
    """An unanticipated error in one mesh is recorded, and every other mesh is still audited."""
    path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name in ("Good", "Bad"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    real_validate = geometry.validate_primvars

    def explode_on_bad(prim, *args, **kwargs):
        if prim.GetName() == "Bad":
            raise RuntimeError("unexpected authored data")
        return real_validate(prim, *args, **kwargs)

    monkeypatch.setattr(geometry, "validate_primvars", explode_on_bad)

    report = _audit(path)

    assert report["mesh_count"] == 2
    assert [record["path"] for record in report["worst_meshes"]] == ["/World/Good"]
    assert report["check_errors"] == {
        "count": 1,
        "examples": [
            {"check": "mesh_record", "subject": "/World/Bad", "error": "RuntimeError: unexpected authored data"}
        ],
    }
