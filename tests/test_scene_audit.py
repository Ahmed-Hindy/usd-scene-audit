"""Regression tests for the scene audit's material binding and naming checks."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade

from usd_scene_audit import scene


def test_mesh_and_subset_material_counts(tmp_path: Path) -> None:
    """Meshes and GeomSubsets are bound, counted, and cross-referenced independently.

    The mesh and the subset bind different materials, so dropping either side's
    contribution to the used-material tally changes the counts below.
    """
    stage_path = tmp_path / "materials.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh_material = UsdShade.Material.Define(stage, "/World/mtl/MeshMat")
    subset_material = UsdShade.Material.Define(stage, "/World/mtl/SubsetMat")
    UsdShade.Material.Define(stage, "/World/mtl/Unused")

    UsdShade.MaterialBindingAPI.Apply(UsdGeom.Mesh.Define(stage, "/World/geo/MeshBound").GetPrim()).Bind(mesh_material)
    UsdGeom.Mesh.Define(stage, "/World/geo/Unbound")
    UsdGeom.Mesh.Define(stage, "/World/geo/SubsetOnly")
    bound_subset = UsdGeom.Subset.Define(stage, "/World/geo/SubsetOnly/face0")
    UsdShade.MaterialBindingAPI.Apply(bound_subset.GetPrim()).Bind(subset_material)
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
    assert materials["bound_materials_used_by_mesh_count"] == 2
    assert materials["unbound_material_prim_count"] == 1


@pytest.mark.parametrize("name", ["a b", "a\tb", "a\u00a0b", "a\u3000b", "trailing "])
def test_usd_rejects_whitespace_prim_names(name: str) -> None:
    """scene.py has no whitespace name check because USD cannot represent such a prim.

    If a future USD release relaxes identifier rules, this fails and the check
    must come back.
    """
    assert not Sdf.Path.IsValidIdentifier(name)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    with pytest.raises(Tf.ErrorException):
        Sdf.PrimSpec(layer, name, Sdf.SpecifierDef)
