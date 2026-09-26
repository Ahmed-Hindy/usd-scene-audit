"""Audit a composed OpenUSD stage for naming and material assignment issues."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

# TODO(#22): CheckErrorLog belongs in a shared module once one exists; it is not
# geometry-specific.
from usd_scene_audit.geometry import CheckErrorLog


MAX_EXAMPLES = 40

# A URI scheme must be detected before anchoring, because anchoring a path
# collapses the "//" in "scheme://host" into "scheme:/host". The scheme requires
# two or more characters so a Windows drive letter ("C://tex/a.exr") is not
# mistaken for a URI.
ASSET_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+://")

# Asset paths that stand in for a family of files rather than one file. Their
# existence cannot be decided by resolving the authored string.
UDIM_TOKEN_PATTERN = re.compile(r"<UDIM>", re.IGNORECASE)

# Frame-sequence tokens. Both alternatives are bounded deliberately:
#   - "$F" needs a non-alphanumeric follower, so the Houdini-style "$F4" matches
#     while variable names such as "$FX", "$FONTS", and "$FOOTAGE" do not.
#   - a "####" run must sit between "." or "_" delimiters, so "beauty.####.exr"
#     matches while directory names such as "v###" or "notes##draft" do not.
SEQUENCE_TOKEN_PATTERN = re.compile(
    r"<f\d*>|<n\d*>|<frame>|\$F\d*(?![A-Za-z0-9])|[._]#{2,}(?=[._])",
    re.IGNORECASE,
)

# Tiles tried when probing whether a UDIM texture set exists at all. Probing can
# only ever upgrade an asset to "resolved"; a failed probe never reports missing,
# because tile numbering is asset-specific.
UDIM_PROBE_TILES = ("1001", "1002")

COMMON_CONTAINER_NAMES = {
    "geo",
    "mtl",
    "materials",
    "render",
    "proxy",
    "lod",
    "payload",
    "payloads",
    "variants",
}


def add_example(bucket: list[str], value: str, max_items: int = MAX_EXAMPLES) -> None:
    """Append an example while keeping report size bounded."""
    if len(bucket) < max_items:
        bucket.append(value)


def direct_material_targets(prim: Usd.Prim) -> list[str]:
    """Return authored material binding targets directly on this prim."""
    targets: list[str] = []
    for rel in prim.GetRelationships():
        name = rel.GetName()
        if name == "material:binding" or name.startswith("material:binding:"):
            for target in rel.GetTargets():
                targets.append(str(target))
    return targets


def prims_with_prototypes(stage: Usd.Stage) -> tuple[list[Usd.Prim], int]:
    """Return ordinary stage traversal plus prototype contents, and the prototype count."""
    prims = list(stage.Traverse())
    prototypes = list(stage.GetPrototypes())
    for prototype in prototypes:
        prims.extend(Usd.PrimRange(prototype))
    return prims, len(prototypes)


def computed_material_bindings(stage: Usd.Stage, prim_paths: list[str]) -> dict[str, str | None]:
    """Map each prim path to its computed bound material path, or None when unbound.

    Keys keep the order of ``prim_paths``.
    """
    bindings: dict[str, str | None] = {}
    for prim_path in prim_paths:
        binding_api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path))
        material, _relationship = binding_api.ComputeBoundMaterial()
        bindings[prim_path] = str(material.GetPath()) if material and material.GetPrim() else None
    return bindings


# Sdf.ListOp fields that can hold references, payloads, or asset paths.
LIST_OP_FIELDS = (
    "explicitItems",
    "addedItems",
    "prependedItems",
    "appendedItems",
    "deletedItems",
    "orderedItems",
)


def collect_asset_paths(value, assets: set[str]) -> None:
    """Add every authored asset path found in an Sdf field value to ``assets``."""
    if hasattr(value, "assetPath"):
        # Sdf.Reference and Sdf.Payload.
        if value.assetPath:
            assets.add(value.assetPath)
        return
    if isinstance(value, Sdf.AssetPath):
        # Only the authored path. resolvedPath is derived from it, so adding
        # both counted a single reference twice.
        if value.path:
            assets.add(value.path)
        return
    for list_attr in LIST_OP_FIELDS:
        if hasattr(value, list_attr):
            collect_asset_paths(getattr(value, list_attr), assets)
    if isinstance(value, dict):
        for key, item in value.items():
            collect_asset_paths(key, assets)
            collect_asset_paths(item, assets)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            collect_asset_paths(item, assets)


def collect_spec_asset_paths(layer: Sdf.Layer, spec, assets: set[str], error_log: CheckErrorLog | None) -> None:
    """Walk one spec's fields, then its child prims and properties."""
    for field_name in spec.ListInfoKeys():
        try:
            collect_asset_paths(spec.GetInfo(field_name), assets)
        except Exception as error:  # noqa: BLE001 - recorded below; see CheckErrorLog
            # An unreadable field could hide an asset reference, which is the
            # very thing this walk exists to find. Skipping is acceptable;
            # skipping silently is not.
            if error_log is not None:
                error_log.record("authored_asset_paths", f"{layer.identifier}:{spec.path}.{field_name}", error)
    for child_spec in getattr(spec, "nameChildren", ()):
        collect_spec_asset_paths(layer, child_spec, assets, error_log)
    for prop_spec in getattr(spec, "properties", ()):
        collect_spec_asset_paths(layer, prop_spec, assets, error_log)


def authored_asset_paths(layer: Sdf.Layer, error_log: CheckErrorLog | None = None) -> set[str]:
    """Collect authored asset paths from a layer by walking Sdf fields."""
    assets: set[str] = set()
    for root in layer.rootPrims:
        collect_spec_asset_paths(layer, root, assets, error_log)
    return assets


def has_variable_tokens(asset_path: str) -> bool:
    """Return true when an asset path names a family of files, not one file."""
    return bool(UDIM_TOKEN_PATTERN.search(asset_path) or SEQUENCE_TOKEN_PATTERN.search(asset_path))


def asset_identifier(layer: Sdf.Layer, asset_path: str) -> str:
    """Anchor an authored asset path against the layer that authored it.

    Uses ``Sdf.ComputeAssetPathRelativeToLayer``, which is the only accessor that
    preserves a layer's *package* context. Anchoring against ``layer.realPath``
    silently discards it: for a layer loaded out of a ``.usdz``, ``realPath`` is
    the path of the package file, so a layer-relative asset anchors to a sibling
    of the archive instead of into it.

        packaged layer realPath    : /show/packed.usdz
        realPath anchoring         : /show/tex/color.png              (wrong)
        ComputeAssetPathRelativeToLayer: /show/packed.usdz[tex/color.png]  (right)

    An explicitly relative path such as ``./tex/color.exr`` anchors to the layer
    directory whether or not the file exists, which is what lets a missing asset
    be reported with its expected location.

    A *bare* relative path such as ``tex/color.exr`` is a USD search path. USD
    resolves it against the resolver's search path rather than the layer, so it
    is returned unanchored when it does not resolve -- it has no single expected
    location, and which directory it resolves from can depend on the process
    working directory. That is USD's semantics, not a choice made here.
    """
    if ASSET_URI_PATTERN.match(asset_path):
        return asset_path
    if layer.anonymous:
        return asset_path
    return Sdf.ComputeAssetPathRelativeToLayer(layer, asset_path)


def classify_authored_asset(layer: Sdf.Layer, asset_path: str) -> tuple[str, str]:
    """Classify an authored asset path as resolved, missing, or unverifiable.

    Resolution goes through ``Ar`` rather than ``os.path``, so package-relative
    paths into ``.usdz`` archives and paths served by a custom resolver are
    judged correctly. ``os.path.exists`` reports both as missing.
    """
    resolver = Ar.GetResolver()
    identifier = asset_identifier(layer, asset_path)

    if has_variable_tokens(asset_path):
        # Substitute into the authored path and re-anchor, so a probe works for
        # search paths as well as for explicitly relative paths.
        for tile in UDIM_PROBE_TILES:
            probe = UDIM_TOKEN_PATTERN.sub(tile, asset_path)
            if probe != asset_path and resolver.Resolve(asset_identifier(layer, probe)):
                return "resolved", identifier
        return "unverifiable", identifier

    if resolver.Resolve(identifier):
        return "resolved", identifier

    if ASSET_URI_PATTERN.match(asset_path):
        # No registered resolver claimed this scheme, so existence cannot be
        # decided locally. Calling it missing would be a guess, and a stage that
        # legitimately references cloud assets would report false findings on any
        # machine without the matching resolver plugin installed.
        return "unverifiable", identifier

    return "missing", identifier


def is_expected_prefix_name(name: str, prefix_style_re: re.Pattern[str]) -> bool:
    """Check a loose vendor/package prefix naming convention."""
    if name in COMMON_CONTAINER_NAMES:
        return True
    if name == "__class__" or name.startswith("__Prototype_"):
        return True
    return bool(prefix_style_re.fullmatch(name))


@dataclass
class PrimInventory:
    """Prims of interest collected in one traversal, in traversal order."""

    type_counts: Counter[str] = field(default_factory=Counter)
    child_names_by_parent: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # A list, not a set: materials_without_surface_output is reported in this
    # order, and set order changes with the per-process string hash seed.
    material_paths: list[str] = field(default_factory=list)
    mesh_paths: list[str] = field(default_factory=list)
    geom_subset_paths: list[str] = field(default_factory=list)
    subsets_by_parent_mesh: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def new_report(stage_path: Path, stage: Usd.Stage, used_layers: list[Sdf.Layer], naming_policy: dict) -> dict:
    """Return the report skeleton, with every count at zero and every list empty."""
    return {
        "stage": str(stage_path),
        "root_layer": stage.GetRootLayer().identifier,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "layer_count": len(used_layers),
        "prims": {},
        "prototype_count": 0,
        "naming_policy": naming_policy,
        "naming": {
            "suspicious_count": 0,
            "non_prefix_style_count": 0,
            "duplicate_sibling_count": 0,
            "duplicate_sibling_names": [],
            "case_collision_count": 0,
            "case_collision_names": [],
            "examples": defaultdict(list),
        },
        "materials": {
            "material_prim_count": 0,
            "shader_prim_count": 0,
            "mesh_count": 0,
            "geom_subset_count": 0,
            "mesh_with_computed_material": 0,
            "mesh_without_computed_material": 0,
            "geom_subset_with_computed_material": 0,
            "geom_subset_without_computed_material": 0,
            "mesh_without_mesh_or_subset_material": 0,
            "bound_materials_used_by_mesh_count": 0,
            "unbound_material_prim_count": 0,
            "direct_binding_relation_count": 0,
            "direct_binding_targets_missing_count": 0,
            "direct_binding_targets_missing": [],
            "direct_binding_targets_not_material_count": 0,
            "direct_binding_targets_not_material": [],
            "mesh_without_material_examples": [],
            "mesh_without_mesh_or_subset_material_examples": [],
            "geom_subset_without_material_examples": [],
            "materials_without_surface_output": [],
            "materials_without_surface_output_count": 0,
        },
        "assets": {
            "authored_asset_count": 0,
            "resolved_asset_count": 0,
            "missing_authored_asset_count": 0,
            "missing_authored_assets": [],
            "unverifiable_asset_count": 0,
            "unverifiable_assets": [],
        },
        "check_errors": {"count": 0, "examples": []},
        "elapsed_seconds": None,
    }


def check_prim_name(name: str, path: str, naming: dict, prefix_style_re: re.Pattern[str] | None) -> None:
    """Record suspicious characters and prefix-policy violations in one prim name."""
    if name == "__class__" or name.startswith("__Prototype_"):
        return
    # No separate whitespace check: SdfPath rejects prim names containing
    # whitespace on authoring and on read (.usda and .usdc), so a composed
    # stage cannot hold one -- and the identifier-char check below would
    # flag it regardless. tests/test_scene_audit.py pins that assumption.
    if re.search(r"[^A-Za-z0-9_]", name):
        naming["suspicious_count"] += 1
        add_example(naming["examples"]["non_ascii_identifier_chars"], path)
    if "__" in name:
        naming["suspicious_count"] += 1
        add_example(naming["examples"]["double_underscore"], path)
    if prefix_style_re and not is_expected_prefix_name(name, prefix_style_re):
        naming["non_prefix_style_count"] += 1
        add_example(naming["examples"]["non_prefix_style"], path)


def check_direct_bindings(stage: Usd.Stage, prim: Usd.Prim, path: str, materials: dict) -> None:
    """Record direct material bindings whose target is missing or is not a Material."""
    targets = direct_material_targets(prim)
    materials["direct_binding_relation_count"] += len(targets)
    for target in targets:
        target_prim = stage.GetPrimAtPath(target)
        if not target_prim:
            materials["direct_binding_targets_missing_count"] += 1
            add_example(materials["direct_binding_targets_missing"], f"{path} -> {target}")
        elif not target_prim.IsA(UsdShade.Material):
            materials["direct_binding_targets_not_material_count"] += 1
            add_example(
                materials["direct_binding_targets_not_material"],
                f"{path} -> {target} ({target_prim.GetTypeName() or '<untyped>'})",
            )


def enclosing_mesh(prim: Usd.Prim) -> Usd.Prim | None:
    """Return the nearest Mesh ancestor of a prim, or None."""
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsA(UsdGeom.Mesh):
        parent = parent.GetParent()
    return parent if parent and parent.IsValid() and parent.IsA(UsdGeom.Mesh) else None


def inventory_prim(prim: Usd.Prim, path: str, inventory: PrimInventory, materials: dict) -> None:
    """Record a prim's type and file it under the categories later passes need."""
    inventory.type_counts[prim.GetTypeName() or "<untyped>"] += 1
    parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else "/"
    inventory.child_names_by_parent[parent_path].append(prim.GetName())
    if prim.IsA(UsdShade.Material):
        inventory.material_paths.append(path)
    if prim.IsA(UsdShade.Shader):
        materials["shader_prim_count"] += 1
    if prim.IsA(UsdGeom.Mesh):
        inventory.mesh_paths.append(path)
    if prim.IsA(UsdGeom.Subset):
        inventory.geom_subset_paths.append(path)
        mesh = enclosing_mesh(prim)
        if mesh:
            inventory.subsets_by_parent_mesh[str(mesh.GetPath())].append(path)


def scan_prims(
    stage: Usd.Stage, prims: list[Usd.Prim], report: dict, prefix_style_re: re.Pattern[str] | None
) -> PrimInventory:
    """Run the per-prim checks and collect the inventory the later passes need."""
    inventory = PrimInventory()
    for prim in prims:
        path = str(prim.GetPath())
        inventory_prim(prim, path, inventory, report["materials"])
        check_prim_name(prim.GetName(), path, report["naming"], prefix_style_re)
        check_direct_bindings(stage, prim, path, report["materials"])
    return inventory


def check_sibling_names(child_names_by_parent: dict[str, list[str]], naming: dict) -> None:
    """Record duplicate and case-colliding sibling names.

    Counts are one per colliding group, matching usd-names-hierarchy-audit.
    """
    for parent, names in child_names_by_parent.items():
        for name, count in Counter(names).items():
            if count > 1:
                naming["duplicate_sibling_count"] += 1
                add_example(naming["duplicate_sibling_names"], f"{parent}/{name} x{count}")
        by_lower: dict[str, set[str]] = defaultdict(set)
        for name in names:
            by_lower[name.lower()].add(name)
        for originals in by_lower.values():
            if len(originals) > 1:
                naming["case_collision_count"] += 1
                add_example(naming["case_collision_names"], f"{parent}: {', '.join(sorted(originals))}")


def check_material_bindings(stage: Usd.Stage, inventory: PrimInventory, materials: dict) -> None:
    """Record meshes and subsets without a computed material, and unused or incomplete materials."""
    mesh_bindings = computed_material_bindings(stage, inventory.mesh_paths)
    subset_bindings = computed_material_bindings(stage, inventory.geom_subset_paths)
    computed_materials: Counter[str] = Counter(m for m in mesh_bindings.values() if m)
    computed_materials.update(m for m in subset_bindings.values() if m)
    mesh_has_material = {path for path, m in mesh_bindings.items() if m}
    subset_has_material = {path for path, m in subset_bindings.items() if m}

    materials["mesh_with_computed_material"] = len(mesh_has_material)
    materials["mesh_without_computed_material"] = len(inventory.mesh_paths) - len(mesh_has_material)
    for mesh_path, material_path in mesh_bindings.items():
        if material_path is None:
            add_example(materials["mesh_without_material_examples"], mesh_path)

    materials["geom_subset_with_computed_material"] = len(subset_has_material)
    materials["geom_subset_without_computed_material"] = len(inventory.geom_subset_paths) - len(subset_has_material)
    for subset_path, material_path in subset_bindings.items():
        if material_path is None:
            add_example(materials["geom_subset_without_material_examples"], subset_path)

    for mesh_path in inventory.mesh_paths:
        subsets = inventory.subsets_by_parent_mesh.get(mesh_path, [])
        if mesh_path in mesh_has_material or any(subset in subset_has_material for subset in subsets):
            continue
        materials["mesh_without_mesh_or_subset_material"] += 1
        add_example(materials["mesh_without_mesh_or_subset_material_examples"], mesh_path)

    for material_path in inventory.material_paths:
        surface = UsdShade.Material(stage.GetPrimAtPath(material_path)).GetSurfaceOutput()
        if not surface or not surface.HasConnectedSource():
            materials["materials_without_surface_output_count"] += 1
            add_example(materials["materials_without_surface_output"], material_path)

    materials["material_prim_count"] = len(inventory.material_paths)
    materials["mesh_count"] = len(inventory.mesh_paths)
    materials["geom_subset_count"] = len(inventory.geom_subset_paths)
    materials["bound_materials_used_by_mesh_count"] = len(computed_materials)
    materials["unbound_material_prim_count"] = len(set(inventory.material_paths) - set(computed_materials))


def check_assets(used_layers: list[Sdf.Layer], assets_report: dict, error_log: CheckErrorLog) -> None:
    """Classify every authored asset path in the used layers as resolved, missing, or unverifiable."""
    missing_assets: dict[str, set[str]] = defaultdict(set)
    unverifiable_assets: dict[str, set[str]] = defaultdict(set)
    for layer in used_layers:
        for asset in authored_asset_paths(layer, error_log):
            status, identifier = classify_authored_asset(layer, asset)
            assets_report["authored_asset_count"] += 1
            if status == "resolved":
                assets_report["resolved_asset_count"] += 1
            elif status == "missing":
                missing_assets[identifier].add(layer.identifier)
            else:
                unverifiable_assets[identifier].add(layer.identifier)

    for key, found in (("missing_authored_assets", missing_assets), ("unverifiable_assets", unverifiable_assets)):
        for asset, layers in sorted(found.items()):
            add_example(assets_report[key], f"{asset} (authored from {len(layers)} layer(s))", max_items=120)
    assets_report["missing_authored_asset_count"] = len(missing_assets)
    assets_report["unverifiable_asset_count"] = len(unverifiable_assets)


def analyze(stage_path: Path, *, prefix_style_pattern: str | None = None) -> dict:
    """Audit naming, material bindings, and authored asset references on a composed stage."""
    start = time.perf_counter()
    error_log = CheckErrorLog()
    prefix_style_re = re.compile(prefix_style_pattern) if prefix_style_pattern else None
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    # GetUsedLayers() has no stable order between runs. Sorting keeps the order
    # of recorded check_errors, which is capped, identical across runs.
    used_layers = sorted(stage.GetUsedLayers(), key=lambda layer: layer.identifier)
    naming_policy = {"prefix_style": {"enabled": prefix_style_re is not None, "pattern": prefix_style_pattern}}
    report = new_report(stage_path, stage, used_layers, naming_policy)

    prims, report["prototype_count"] = prims_with_prototypes(stage)
    inventory = scan_prims(stage, prims, report, prefix_style_re)
    check_sibling_names(inventory.child_names_by_parent, report["naming"])
    check_material_bindings(stage, inventory, report["materials"])
    check_assets(used_layers, report["assets"], error_log)

    report["prims"] = {
        "total": sum(inventory.type_counts.values()),
        "by_type": dict(inventory.type_counts.most_common()),
    }
    report["check_errors"] = error_log.as_report()
    report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    # Convert defaultdicts for JSON stability.
    report["naming"]["examples"] = dict(report["naming"]["examples"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--prefix-style-pattern",
        help="Optional regex for vendor/package prefix-style names. Disabled by default.",
    )
    args = parser.parse_args()

    if args.prefix_style_pattern:
        try:
            re.compile(args.prefix_style_pattern)
        except re.error as exc:
            parser.error(f"invalid --prefix-style-pattern: {exc}")

    report = analyze(args.stage, prefix_style_pattern=args.prefix_style_pattern)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
