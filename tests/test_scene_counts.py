"""The scene audit's capped example lists must each carry an exact count.

Example lists stop at ``scene.MAX_EXAMPLES``. Without a separate count, the
only number a consumer could read was the list length, which silently stopped
at the cap on large stages.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

from usd_scene_audit import scene

OVER_CAP = scene.MAX_EXAMPLES + 10


def test_case_collision_count_is_exact_beyond_example_cap(tmp_path: Path) -> None:
    """One count per colliding sibling group, even for three spellings."""
    path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for index in range(OVER_CAP - 1):
        UsdGeom.Xform.Define(stage, f"/Group_{index}/part")
        UsdGeom.Xform.Define(stage, f"/Group_{index}/Part")
    for spelling in ("part", "Part", "PART"):
        UsdGeom.Xform.Define(stage, f"/Triple/{spelling}")
    stage.GetRootLayer().Save()

    naming = scene.analyze(path)["naming"]

    assert naming["case_collision_count"] == OVER_CAP
    assert len(naming["case_collision_names"]) == scene.MAX_EXAMPLES
    assert naming["duplicate_sibling_count"] == 0


def test_duplicate_sibling_count_is_exact_beyond_example_cap(tmp_path: Path, monkeypatch) -> None:
    """USD cannot author duplicate siblings, so feed each prim twice to reach the branch."""
    path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for index in range(OVER_CAP):
        UsdGeom.Xform.Define(stage, f"/Group_{index}/child")
    stage.GetRootLayer().Save()
    real_traversal = scene.prims_with_prototypes

    def doubled(stage):
        prims, prototype_count = real_traversal(stage)
        return prims * 2, prototype_count

    monkeypatch.setattr(scene, "prims_with_prototypes", doubled)

    naming = scene.analyze(path)["naming"]

    # Each doubled /Group_i is one group under "/", and each doubled child one more.
    assert naming["duplicate_sibling_count"] == 2 * OVER_CAP
    assert len(naming["duplicate_sibling_names"]) == scene.MAX_EXAMPLES


def test_direct_binding_target_counts_are_exact_beyond_example_cap(tmp_path: Path) -> None:
    """Missing and non-material binding targets are counted per target, not per example."""
    path = tmp_path / "stage.usda"
    stage = Usd.Stage.CreateNew(str(path))
    not_a_material = UsdGeom.Xform.Define(stage, "/World/NotAMaterial").GetPrim().GetPath()
    for index in range(OVER_CAP):
        missing = UsdGeom.Mesh.Define(stage, f"/World/Missing_{index}").GetPrim()
        missing.CreateRelationship("material:binding").SetTargets([Sdf.Path(f"/World/NoSuchMaterial_{index}")])
        wrong = UsdGeom.Mesh.Define(stage, f"/World/Wrong_{index}").GetPrim()
        wrong.CreateRelationship("material:binding").SetTargets([not_a_material])
    stage.GetRootLayer().Save()

    materials = scene.analyze(path)["materials"]

    assert materials["direct_binding_relation_count"] == 2 * OVER_CAP
    assert materials["direct_binding_targets_missing_count"] == OVER_CAP
    assert len(materials["direct_binding_targets_missing"]) == scene.MAX_EXAMPLES
    assert materials["direct_binding_targets_not_material_count"] == OVER_CAP
    assert len(materials["direct_binding_targets_not_material"]) == scene.MAX_EXAMPLES


def test_counts_are_zero_on_a_clean_stage(stage_path) -> None:
    """The new keys exist and read zero when nothing is wrong."""
    report = scene.analyze(stage_path("static_mesh_clean.usda"))

    assert report["naming"]["case_collision_count"] == 0
    assert report["naming"]["duplicate_sibling_count"] == 0
    assert report["materials"]["direct_binding_targets_missing_count"] == 0
    assert report["materials"]["direct_binding_targets_not_material_count"] == 0
