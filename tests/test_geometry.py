"""Regression tests for geometry audit helpers."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from usd_asset_audit.geometry import transform_determinant, validate_primvars


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
