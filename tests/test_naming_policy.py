"""Regression tests for optional naming policy checks."""

from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdGeom

from usd_scene_audit import names_hierarchy, scene


def _write_named_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/PlainName")
    UsdGeom.Xform.Define(stage, "/World/USD_ASSET_PrefixName")
    stage.GetRootLayer().Save()


def test_scene_audit_prefix_policy_is_off_by_default(tmp_path: Path) -> None:
    """Generic USD scene audits should not require vendor-prefix naming by default."""
    stage_path = tmp_path / "scene.usda"
    _write_named_stage(stage_path)

    report = scene.analyze(stage_path)

    assert report["naming_policy"]["prefix_style"] == {"enabled": False, "pattern": None}
    assert report["naming"]["non_prefix_style_count"] == 0
    assert "non_prefix_style" not in report["naming"]["examples"]


def test_scene_audit_prefix_policy_can_be_enabled(tmp_path: Path) -> None:
    """Prefix-style naming can be opted into with a custom regex."""
    stage_path = tmp_path / "scene.usda"
    _write_named_stage(stage_path)

    report = scene.analyze(stage_path, prefix_style_pattern=r"USD_[A-Z]+_[A-Za-z0-9]+")

    assert report["naming_policy"]["prefix_style"] == {
        "enabled": True,
        "pattern": r"USD_[A-Z]+_[A-Za-z0-9]+",
    }
    assert report["naming"]["non_prefix_style_count"] == 2
    assert report["naming"]["examples"]["non_prefix_style"] == ["/World", "/World/PlainName"]


def test_names_hierarchy_prefix_policy_is_off_by_default(tmp_path: Path) -> None:
    """Naming/hierarchy audits should avoid prefix-style oddities unless requested."""
    stage_path = tmp_path / "scene.usda"
    _write_named_stage(stage_path)

    report = names_hierarchy.analyze(stage_path)

    assert report["naming_policy"]["prefix_style"] == {"enabled": False, "pattern": None}
    assert "non_prefix_style" not in report["name_oddity_counts"]


def test_names_hierarchy_prefix_policy_can_be_enabled(tmp_path: Path) -> None:
    """Naming/hierarchy audits should report prefix mismatches when a policy is supplied."""
    stage_path = tmp_path / "scene.usda"
    _write_named_stage(stage_path)

    report = names_hierarchy.analyze(stage_path, prefix_style_pattern=r"USD_[A-Z]+_[A-Za-z0-9]+")

    assert report["naming_policy"]["prefix_style"] == {
        "enabled": True,
        "pattern": r"USD_[A-Z]+_[A-Za-z0-9]+",
    }
    assert report["name_oddity_counts"]["non_prefix_style"] == 2
    assert report["name_oddities"]["non_prefix_style"] == ["/World", "/World/PlainName"]
