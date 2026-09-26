"""Regression tests for scene-audit material binding counts."""

from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade, Vt

from usd_scene_audit import scene


def _mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 1, 3, 2]))
    return mesh


def test_mesh_and_subset_material_counts(tmp_path: Path) -> None:
    """Meshes and GeomSubsets are bound, counted, and cross-referenced independently."""
    stage_path = tmp_path / "materials.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/mtl/Bound")
    UsdShade.Material.Define(stage, "/World/mtl/Unused")

    UsdShade.MaterialBindingAPI.Apply(_mesh(stage, "/World/geo/MeshBound").GetPrim()).Bind(material)
    _mesh(stage, "/World/geo/Unbound")
    _mesh(stage, "/World/geo/SubsetOnly")
    bound_subset = UsdGeom.Subset.Define(stage, "/World/geo/SubsetOnly/face0")
    UsdShade.MaterialBindingAPI.Apply(bound_subset.GetPrim()).Bind(material)
    UsdGeom.Subset.Define(stage, "/World/geo/SubsetOnly/face1")
    stage.GetRootLayer().Save()

    materials = scene.analyze(stage_path)["materials"]

    assert materials["mesh_count"] == 3
    assert materials["mesh_with_computed_material"] == 1
    assert materials["mesh_without_computed_material"] == 2
    assert materials["mesh_without_material_examples"] == ["/World/geo/Unbound", "/World/geo/SubsetOnly"]
    assert materials["geom_subset_count"] == 2
    assert materials["geom_subset_with_computed_material"] == 1
    assert materials["geom_subset_without_computed_material"] == 1
    assert materials["geom_subset_without_material_examples"] == ["/World/geo/SubsetOnly/face1"]
    # SubsetOnly is covered through its subset, so only Unbound has no material at all.
    assert materials["mesh_without_mesh_or_subset_material_examples"] == ["/World/geo/Unbound"]
    assert materials["bound_materials_used_by_mesh_count"] == 1
    assert materials["unbound_material_prim_count"] == 1
