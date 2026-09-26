"""Regression tests for mesh attributes authored with the wrong value type.

``GetNormalsAttr().Get()`` and friends return whatever value type the layer
authored, not the schema type. A ``float[] normals`` or a scalar
``int faceVertexCounts`` used to raise inside the vectorized checks and abort
the whole audit; a ``float[] faceVertexIndices`` was silently truncated to
integers. They must be reported as findings instead.
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
TOPOLOGY_CODES = ("points_wrong_type", "face_vertex_counts_wrong_type", "face_vertex_indices_wrong_type")


def _write_mesh(path: Path, name: str = "M", **overrides: str) -> None:
    lines = {**GOOD_ATTRIBUTES, **overrides}
    body = "\n    ".join(lines.values())
    path.write_text(f'#usda 1.0\ndef Mesh "{name}"\n{{\n    {body}\n}}\n', encoding="utf-8")


def _audit(path: Path) -> dict:
    return geometry.analyze(path, 1e-12, 1e6, 1e-4)


MISTYPED_CASES = [
    ({"normals": "float[] normals = [0, 0, 1]"}, "normals_wrong_type", "float[]"),
    ({"normals": "float3 normals = (0, 0, 1)"}, "normals_wrong_type", "float3"),
    ({"points": "float[] points = [0, 0, 0, 1, 0, 0, 0, 1, 0]"}, "points_wrong_type", "float[]"),
    ({"points": "float2[] points = [(0,0), (1,0), (0,1)]"}, "points_wrong_type", "float2[]"),
    ({"points": "point3f points = (0, 0, 0)"}, "points_wrong_type", "float3"),
    ({"points": 'string[] points = ["a", "b", "c"]'}, "points_wrong_type", "string[]"),
    ({"counts": "int faceVertexCounts = 3"}, "face_vertex_counts_wrong_type", "int"),
    ({"counts": "float[] faceVertexCounts = [3.9]"}, "face_vertex_counts_wrong_type", "float[]"),
    ({"indices": "int2[] faceVertexIndices = [(0, 1), (2, 0)]"}, "face_vertex_indices_wrong_type", "int2[]"),
    ({"indices": "float[] faceVertexIndices = [0.7, 1.2, 2.9]"}, "face_vertex_indices_wrong_type", "float[]"),
    ({"indices": "bool[] faceVertexIndices = [0, 1, 1]"}, "face_vertex_indices_wrong_type", "bool[]"),
    ({"indices": 'string[] faceVertexIndices = ["0", "1", "2"]'}, "face_vertex_indices_wrong_type", "string[]"),
    ({"indices": "int3 faceVertexIndices = (0, 1, 2)"}, "face_vertex_indices_wrong_type", "int3"),
    ({"extent": "float[] extent = [0, 0, 0, 1, 1, 0]"}, "extent_wrong_type", "float[]"),
    ({"extent": 'string[] extent = ["a", "b"]'}, "extent_wrong_type", "string[]"),
]


@pytest.mark.parametrize(
    ("override", "finding", "authored_type"),
    MISTYPED_CASES,
    ids=[f"{next(iter(o))}-{t}" for o, _f, t in MISTYPED_CASES],
)
def test_mistyped_attribute_is_a_finding_not_a_crash(tmp_path: Path, override, finding, authored_type) -> None:
    """Each of these raised or was silently misread on main; each must report a *_wrong_type finding."""
    path = tmp_path / "mesh.usda"
    _write_mesh(path, **override)

    report = _audit(path)

    assert report["summary_counts"][finding] == 1
    assert report["check_errors"]["count"] == 0
    assert report["unaudited_mesh_count"] == 0
    [detail] = report["worst_meshes"][0]["details"][finding]
    assert detail["authored_type"] == authored_type
    assert detail["expected_type"] != authored_type
    assert detail["reason"]
    expected_serious = 1 if finding in TOPOLOGY_CODES else 0
    assert report["serious_geometry_failures"] >= expected_serious


@pytest.mark.parametrize("finding", TOPOLOGY_CODES)
def test_unusable_topology_counts_as_a_serious_failure(finding: str) -> None:
    """Unusable topology ranks and counts with missing topology."""
    assert geometry.seriousness_score({"issues": {finding: 1}}) == geometry.seriousness_score(
        {"issues": {"missing_points": 1}}
    )


@pytest.mark.parametrize(
    "override",
    [
        {"points": "double3[] points = [(0,0,0), (1,0,0), (0,1,0)]"},
        {"points": "half3[] points = [(0,0,0), (1,0,0), (0,1,0)]"},
        {"points": "int3[] points = [(0,0,0), (1,0,0), (0,1,0)]"},
        {"normals": 'normal3d[] normals = [(0,0,1), (0,0,1), (0,0,1)] (interpolation = "vertex")'},
        {"indices": "int64[] faceVertexIndices = [0, 1, 2]"},
    ],
    ids=["double3-points", "half3-points", "int3-points", "normal3d-normals", "int64-indices"],
)
def test_usable_numeric_variants_are_audited_normally(tmp_path: Path, override) -> None:
    """Numeric arrays of the right shape are usable data, not type findings."""
    path = tmp_path / "mesh.usda"
    _write_mesh(path, **override)

    assert _audit(path)["summary_counts"] == {}


def test_empty_arrays_are_not_wrong_types(tmp_path: Path) -> None:
    """Empty arrays keep reporting as empty, whatever their authored type."""
    path = tmp_path / "mesh.usda"
    _write_mesh(path, points="float[] points = []", normals="normal3f[] normals = []")

    issues = _audit(path)["summary_counts"]

    assert "points_wrong_type" not in issues
    assert "normals_wrong_type" not in issues
    assert issues["empty_points"] == 1


def test_authored_type_comes_from_the_value_not_the_strongest_spec(tmp_path: Path) -> None:
    """A stronger layer that only overrides interpolation must not hide the weaker layer's value type."""
    weak = tmp_path / "weak.usda"
    _write_mesh(weak, normals="float[] normals = [0, 0, 1]")
    root = tmp_path / "root.usda"
    root.write_text(
        "#usda 1.0\n(\n    subLayers = [@./weak.usda@]\n)\n"
        'over "M"\n{\n    normal3f[] normals (\n        interpolation = "constant"\n    )\n}\n',
        encoding="utf-8",
    )

    [detail] = _audit(root)["worst_meshes"][0]["details"]["normals_wrong_type"]

    assert detail["authored_type"] == "float[]"
    assert detail["expected_type"] == "normal3f[]"


def _two_triangle_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    for name in ("Good", "Bad"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (float("nan"), 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()


@pytest.mark.expects_check_errors
def test_failing_check_phase_keeps_the_rest_of_the_record(tmp_path: Path, monkeypatch) -> None:
    """A check that raises costs only its own findings; the mesh's other findings still count."""
    path = tmp_path / "stage.usda"
    _two_triangle_stage(path)
    real_validate = geometry.validate_primvars

    def explode_on_bad(prim, *args, **kwargs):
        if prim.GetName() == "Bad":
            raise RuntimeError("unexpected authored data")
        return real_validate(prim, *args, **kwargs)

    monkeypatch.setattr(geometry, "validate_primvars", explode_on_bad)

    report = _audit(path)

    assert report["mesh_count"] == 2
    assert report["unaudited_mesh_count"] == 0
    assert report["summary_counts"]["non_finite_points"] == 2
    assert report["serious_geometry_failures"] == 2
    assert report["check_errors"] == {
        "count": 1,
        "examples": [{"check": "primvars", "subject": "/World/Bad", "error": "RuntimeError: unexpected authored data"}],
    }


def test_failing_check_phase_raises_for_direct_callers_without_a_log(tmp_path: Path, monkeypatch) -> None:
    """With no error_log, mesh_record() must not swallow the exception."""
    path = tmp_path / "stage.usda"
    _two_triangle_stage(path)
    monkeypatch.setattr(geometry, "validate_primvars", lambda *args, **kwargs: 1 / 0)
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/World/Bad")

    with pytest.raises(ZeroDivisionError):
        geometry.mesh_record(
            prim,
            1e-12,
            1e6,
            1e-4,
            UsdGeom.XformCache(),
            "numpy",
            "exhaustive",
            geometry.PhaseTimer(),
            geometry.FaceAnalysisCache(False),
        )


@pytest.mark.expects_check_errors
def test_mesh_that_escapes_every_phase_guard_is_counted_unaudited(tmp_path: Path, monkeypatch, capsys) -> None:
    """The last-resort guard in analyze() records the mesh and reports it as having no record."""
    path = tmp_path / "stage.usda"
    _two_triangle_stage(path)
    real_classify = geometry.classify_mesh

    def explode_on_bad(mesh_path, name):
        if name == "Bad":
            raise RuntimeError("classification failed")
        return real_classify(mesh_path, name)

    monkeypatch.setattr(geometry, "classify_mesh", explode_on_bad)

    report = _audit(path)

    assert report["mesh_count"] == 2
    assert report["unaudited_mesh_count"] == 1
    assert [record["path"] for record in report["worst_meshes"]] == ["/World/Good"]
    assert report["check_errors"]["examples"][0]["check"] == "mesh_record"
    geometry.print_summary(report)
    assert "Meshes with no record: 1" in capsys.readouterr().out
