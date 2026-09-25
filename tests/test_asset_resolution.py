"""Regression tests for authored asset resolution.

Asset existence used to be decided with ``os.path.join`` plus
``os.path.exists``, bypassing USD's asset resolution layer. That reported
several normal authoring patterns as missing files even when the assets were
present and resolved correctly in any USD runtime.
"""

from __future__ import annotations

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdUtils

from usd_scene_audit import scene
from usd_scene_audit.scene import (
    asset_identifier,
    classify_authored_asset,
    has_variable_tokens,
)


def missing_paths(report: dict) -> str:
    """Render the missing-asset list for readable assertion failures."""
    return "\n".join(report["assets"]["missing_authored_assets"]) or "<none>"


# --------------------------------------------------------------------- UDIM


def test_udim_texture_with_tiles_present_is_not_missing(stage_path) -> None:
    """A UDIM pattern is not a filename; present tiles must not read as missing."""
    report = scene.analyze(stage_path("assets_udim_texture.usda"))

    assert report["assets"]["missing_authored_asset_count"] == 0, missing_paths(report)
    assert report["assets"]["missing_authored_assets"] == []


def test_udim_texture_is_counted_as_resolved_via_tile_probe(stage_path) -> None:
    """Probing a tile proves the set exists, so the asset counts as resolved."""
    report = scene.analyze(stage_path("assets_udim_texture.usda"))

    assert report["assets"]["resolved_asset_count"] >= 1
    assert report["assets"]["unverifiable_asset_count"] == 0


def test_udim_pattern_without_tiles_is_unverifiable_not_missing(tmp_path) -> None:
    """With no tiles on disk the pattern is unverifiable, never a false missing."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "root.usda"))
    stage = Usd.Stage.Open(layer)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    prim = UsdGeom.Mesh.Define(stage, "/World/Card").GetPrim()
    prim.CreateAttribute("userProperties:tex", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./nowhere/color.<UDIM>.exr")
    )
    layer.Save()

    report = scene.analyze(tmp_path / "root.usda")

    assert report["assets"]["missing_authored_asset_count"] == 0, missing_paths(report)
    assert report["assets"]["unverifiable_asset_count"] == 1


# ------------------------------------------------------------------ missing


def test_genuinely_missing_texture_is_still_reported(stage_path) -> None:
    """Routing through Ar must not stop real missing assets from being found."""
    report = scene.analyze(stage_path("assets_missing_texture.usda"))

    assert report["assets"]["missing_authored_asset_count"] == 1
    assert "absent.exr" in missing_paths(report)


def test_missing_asset_is_reported_as_an_absolute_path(stage_path) -> None:
    """An explicitly relative path anchors, so the report names where it should be."""
    report = scene.analyze(stage_path("assets_missing_texture.usda"))

    reported = missing_paths(report)
    assert "tests/fixtures/tex/absent.exr" in reported.replace("\\", "/")


# ---------------------------------------------------------------- packages


def _build_package_with_internal_texture(tmp_path):
    """Build a .usdz whose layer references a texture stored inside the package."""
    source = tmp_path / "source"
    (source / "tex").mkdir(parents=True)
    (source / "tex" / "color.png").write_bytes(b"x")

    inner = source / "asset.usda"
    stage = Usd.Stage.CreateNew(str(inner))
    mesh = UsdGeom.Mesh.Define(stage, "/Asset")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.GetPrim().CreateAttribute("userProperties:tex", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./tex/color.png"))
    stage.GetRootLayer().Save()

    package = tmp_path / "packed.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(inner)), str(package))
    return package


def test_asset_authored_inside_a_usdz_is_not_missing(tmp_path) -> None:
    """An asset stored inside a package must resolve to a package-relative path.

    This is the case os.path.exists cannot see at all, and it is also the case
    that anchoring on layer.realPath gets wrong: realPath for a packaged layer is
    the path of the .usdz itself, so a layer-relative asset anchors to a sibling
    of the archive rather than into it.
    """
    package = _build_package_with_internal_texture(tmp_path)

    report = scene.analyze(package)

    assert report["assets"]["missing_authored_asset_count"] == 0, missing_paths(report)
    assert report["assets"]["resolved_asset_count"] >= 1


def test_package_internal_asset_identifier_is_package_relative(tmp_path) -> None:
    """The identifier must keep the package context, not point beside the archive."""
    package = _build_package_with_internal_texture(tmp_path)
    stage = Usd.Stage.Open(str(package))
    layer = next(layer for layer in stage.GetUsedLayers() if not layer.anonymous)

    identifier = asset_identifier(layer, "./tex/color.png")

    assert identifier.endswith("packed.usdz[tex/color.png]"), identifier


# ------------------------------------------------------------------- units


@pytest.mark.parametrize(
    "asset_path",
    [
        "./tex/color.<UDIM>.exr",
        "./tex/COLOR.<udim>.exr",
        "./render/beauty.<f4>.exr",
        "./render/beauty.####.exr",
        "./render/beauty.$F4.exr",
        "./cache/points.<frame>.bgeo",
    ],
)
def test_variable_token_paths_are_detected(asset_path: str) -> None:
    """Patterns standing for a family of files must be recognised as such."""
    assert has_variable_tokens(asset_path)


@pytest.mark.parametrize(
    "asset_path",
    ["./tex/color.exr", "./tex/color.1001.exr", "geo/model.usd", "/abs/tex.png", "a#b.exr"],
)
def test_concrete_paths_are_not_treated_as_patterns(asset_path: str) -> None:
    """A single-file path must not be mistaken for a pattern."""
    assert not has_variable_tokens(asset_path)


@pytest.mark.parametrize(
    "asset_path",
    [
        "$FX/cache/points.bgeo",
        "$FOOTAGE/plate.exr",
        "./$FONTS/glyph.png",
        "$HIP/tex/color.exr",
        "./v###/color.exr",
        "./notes##draft/color.exr",
    ],
)
def test_variable_names_are_not_frame_sequences(asset_path: str) -> None:
    """Environment-variable and directory names must not read as frame tokens.

    These are ordinary in Houdini and Nuke pipelines. Misreading them as patterns
    silently moves the reference into 'unverifiable', where no counter treats it
    as a problem, so it stops being audited at all.
    """
    assert not has_variable_tokens(asset_path)


def test_udim_probe_works_for_search_paths(tmp_path) -> None:
    """A UDIM set must be found whether or not the authored path starts with ./."""
    (tmp_path / "tex").mkdir()
    (tmp_path / "tex" / "color.1001.exr").write_bytes(b"x")
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))
    layer.Save()

    assert classify_authored_asset(layer, "./tex/color.<UDIM>.exr")[0] == "resolved"
    assert classify_authored_asset(layer, "tex/color.<UDIM>.exr")[0] == "resolved"


@pytest.mark.parametrize(
    "uri",
    ["https://example.com/tex.exr", "http://example.com/tex.exr", "omniverse://host/a.usd", "s3://bucket/a.usd"],
)
def test_uri_asset_paths_are_left_intact(tmp_path, uri: str) -> None:
    """Anchoring would collapse the // in a URI, so URIs pass through unchanged."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))

    assert asset_identifier(layer, uri) == uri


@pytest.mark.parametrize(
    "uri",
    ["https://cdn.example.com/tex.exr", "omniverse://nucleus/proj/a.usd"],
)
def test_unclaimed_uri_is_unverifiable_not_missing(tmp_path, uri: str) -> None:
    """A URI no resolver claims cannot be judged, so it must not be called missing.

    Reporting it missing would mean any stage referencing cloud assets produces
    false findings on every machine without the matching resolver plugin.
    """
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))
    layer.Save()

    assert classify_authored_asset(layer, uri)[0] == "unverifiable"


def test_uri_references_are_not_reported_as_missing(tmp_path) -> None:
    """End to end: a stage referencing cloud assets reports no missing assets."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "root.usda"))
    stage = Usd.Stage.Open(layer)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for index, uri in enumerate(["https://cdn.example.com/t.exr", "omniverse://nucleus/proj/a.usd"]):
        world.GetPrim().CreateAttribute(f"userProperties:u{index}", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(uri))
    layer.Save()

    report = scene.analyze(tmp_path / "root.usda")

    assert report["assets"]["missing_authored_asset_count"] == 0, missing_paths(report)
    assert report["assets"]["unverifiable_asset_count"] == 2


def test_windows_drive_letter_is_not_a_uri(tmp_path) -> None:
    """A URI scheme needs 2+ characters, so C:// is a path, not a scheme."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))

    assert asset_identifier(layer, "C://tex/color.exr") != "C://tex/color.exr"


def test_explicitly_relative_path_anchors_to_the_layer(tmp_path) -> None:
    """./foo resolves against the authoring layer's directory."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))

    identifier = asset_identifier(layer, "./tex/color.exr")

    assert identifier.replace("\\", "/").endswith("/tex/color.exr")
    assert identifier != "./tex/color.exr"


def test_search_path_is_left_unanchored(tmp_path) -> None:
    """A bare relative path is a USD search path with no single expected location."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))

    assert asset_identifier(layer, "tex/color.exr") == "tex/color.exr"


def test_anonymous_layer_paths_are_left_unanchored() -> None:
    """An anonymous layer has no directory to anchor against."""
    layer = Sdf.Layer.CreateAnonymous()

    assert asset_identifier(layer, "./tex/color.exr") == "./tex/color.exr"


def test_classify_resolved_and_missing(tmp_path) -> None:
    """The three classifications are driven by the resolver, not by os.path."""
    present = tmp_path / "there.exr"
    present.write_bytes(b"x")
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))
    layer.Save()

    assert classify_authored_asset(layer, "./there.exr")[0] == "resolved"
    assert classify_authored_asset(layer, "./gone.exr")[0] == "missing"
    assert classify_authored_asset(layer, "./gone.<UDIM>.exr")[0] == "unverifiable"


def test_asset_path_is_counted_once(tmp_path) -> None:
    """A single authored reference must not be counted twice.

    Sdf.AssetPath carries both the authored path and its resolved form; both used
    to be collected, inflating authored_asset_count for every resolvable asset.
    """
    target = tmp_path / "there.exr"
    target.write_bytes(b"x")
    layer = Sdf.Layer.CreateNew(str(tmp_path / "root.usda"))
    stage = Usd.Stage.Open(layer)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().CreateAttribute("userProperties:tex", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./there.exr"))
    layer.Save()

    report = scene.analyze(tmp_path / "root.usda")

    assert report["assets"]["authored_asset_count"] == 1
    assert report["assets"]["resolved_asset_count"] == 1
