"""Regression tests for geometry audit helpers."""

from __future__ import annotations

import pytest
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from usd_scene_audit import geometry
from usd_scene_audit.geometry import (
    FaceAnalysisCache,
    PhaseTimer,
    analyze_face_geometry,
    mesh_record,
    resolve_face_analysis_engine,
    transform_determinant,
    validate_normals,
    validate_primvars,
)


def _triangle_mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    return mesh


def test_validate_primvars_uses_logical_element_count() -> None:
    """Primvars with elementSize describe grouped values, not multiplied values."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle_mesh(stage, "/World/Mesh")
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "weights",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.vertex,
        elementSize=2,
    )
    primvar.Set(Vt.FloatArray([1, 2, 3, 4, 5, 6]))

    issues = validate_primvars(mesh.GetPrim(), point_count=3, face_count=1, face_vertex_count=3)

    assert issues == []


def test_validate_primvars_reports_bad_element_grouping() -> None:
    """Invalid grouped primvars should report the grouping issue explicitly."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle_mesh(stage, "/World/Mesh")
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "weights",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.vertex,
        elementSize=2,
    )
    primvar.Set(Vt.FloatArray([1, 2, 3, 4, 5]))

    issues = validate_primvars(mesh.GetPrim(), point_count=3, face_count=1, face_vertex_count=3)

    assert any(issue["issue"] == "primvar_element_size_mismatch" for issue in issues)


def test_transform_determinant_includes_scale() -> None:
    """Negative scale should be visible to the determinant audit."""
    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World/Scaled")
    xform.AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
    mesh = _triangle_mesh(stage, "/World/Scaled/Mesh")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    determinant = transform_determinant(mesh.GetPrim(), cache)

    assert determinant == -1.0


def test_validate_normals_handles_empty_authored_array() -> None:
    """Vectorized normal validation should preserve the old empty-array behavior."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle_mesh(stage, "/World/Mesh")
    mesh.CreateNormalsAttr(Vt.Vec3fArray([]))

    issues = validate_normals(mesh, point_count=3, face_count=1, face_vertex_count=3)

    assert issues == [
        {
            "issue": "normals_length_mismatch",
            "interpolation": "vertex",
            "expected": 3,
            "actual": 0,
        }
    ]


def test_face_geometry_engines_report_same_issues() -> None:
    """NumPy and Numba face checks should agree on exact issue counts."""
    try:
        resolve_face_analysis_engine("numba")
    except RuntimeError:
        pytest.skip("Numba extra is not installed")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    counts = np.array([3, 4, 3], dtype=np.int64)
    indices = np.array([0, 1, 2, 0, 1, 1, 3, 0, 3, 4], dtype=np.int64)

    numpy_issues, _ = analyze_face_geometry(counts, indices, points, 5, 1e-12, "numpy")
    numba_issues, _ = analyze_face_geometry(counts, indices, points, 5, 1e-12, "numba")

    assert dict(numba_issues) == dict(numpy_issues)
    assert numba_issues["faces_with_repeated_vertices"] == 1
    assert numba_issues["zero_area_triangles"] == 2


def test_auto_face_engine_uses_numba_when_available() -> None:
    """Auto should use the faster real-stage face engine when the extra is installed."""
    try:
        resolve_face_analysis_engine("numba")
    except RuntimeError:
        pytest.skip("Numba extra is not installed")

    assert resolve_face_analysis_engine("auto") == resolve_face_analysis_engine("numba")


def test_mesh_record_skips_deep_face_checks_when_index_lengths_mismatch() -> None:
    """Mismatched face arrays should be reported without indexing past available data."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/BrokenMesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1]))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    record = mesh_record(mesh.GetPrim(), 1e-12, 1e6, 1e-4, cache, "numpy", "exhaustive", PhaseTimer(), FaceAnalysisCache(False))

    assert record["issues"]["face_vertex_count_index_length_mismatch"] == 1


def test_fast_face_mode_skips_deep_face_checks() -> None:
    """Fast mode should keep cheap topology checks but skip exact deep face scans."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    counts = np.array([3, 2], dtype=np.int64)
    indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)

    issues, _ = analyze_face_geometry(
        counts,
        indices,
        points,
        3,
        1e-12,
        "numpy",
        check_repeated_vertices=False,
        check_zero_area=False,
    )

    assert issues["one_or_two_vertex_faces"] == 1
    assert "zero_area_triangles" not in issues


def test_face_analysis_cache_reuses_duplicate_array_results() -> None:
    """Face cache should reuse results for identical arrays and thresholds."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    counts = np.array([3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)
    face_cache = FaceAnalysisCache(True)

    first, _ = analyze_face_geometry(counts, indices, points, 3, 1e-12, "numpy", face_cache=face_cache)
    second, _ = analyze_face_geometry(counts.copy(), indices.copy(), points.copy(), 3, 1e-12, "numpy", face_cache=face_cache)

    assert dict(first) == dict(second)
    assert face_cache.stats()["hits"] == 1


@pytest.mark.parametrize("face_cache", [None, FaceAnalysisCache(False)], ids=["no-cache", "disabled-cache"])
def test_disabled_face_cache_does_not_hash_arrays(monkeypatch, face_cache) -> None:
    """With the cache off, building a cache key is pure overhead: a full hash pass per mesh."""

    def fail_digest(array):
        raise AssertionError("array_digest called while the face cache is disabled")

    monkeypatch.setattr(geometry, "array_digest", fail_digest)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    counts = np.array([3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)

    issues, _ = analyze_face_geometry(counts, indices, points, 3, 1e-12, "numpy", face_cache=face_cache)

    assert dict(issues) == {}


def test_analyze_with_mesh_cache_off_does_not_hash_arrays(monkeypatch, stage_path) -> None:
    """The default --mesh-cache off must not pay the hashing cost of face-hash."""
    calls = []
    real_digest = geometry.array_digest
    monkeypatch.setattr(geometry, "array_digest", lambda array: calls.append(1) or real_digest(array))

    geometry.analyze(stage_path("static_mesh_clean.usda"), 1e-12, 1e6, 1e-4, mesh_cache_mode="off")
    assert calls == []

    geometry.analyze(stage_path("static_mesh_clean.usda"), 1e-12, 1e6, 1e-4, mesh_cache_mode="face-hash")
    assert calls, "face-hash mode should still hash arrays to build cache keys"
