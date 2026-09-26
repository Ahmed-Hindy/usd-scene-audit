"""Audit a composed OpenUSD stage for naming and material assignment issues."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade

# TODO(#22): CheckErrorLog belongs in a shared module once one exists; it is not
# geometry-specific.
from usd_scene_audit.geometry import CheckErrorLog, PrototypePaths


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


def authored_asset_paths(layer: Sdf.Layer, error_log: CheckErrorLog | None = None) -> set[str]:
    """Collect authored asset paths from a layer by walking Sdf fields."""
    assets: set[str] = set()

    def visit(value) -> None:
        if hasattr(value, "assetPath"):
            asset_path = getattr(value, "assetPath", "")
            if asset_path:
                assets.add(asset_path)
            prim_path = getattr(value, "primPath", None)
            if prim_path:
                visit(prim_path)
            return
        if isinstance(value, Sdf.AssetPath):
            # Only the authored path. resolvedPath is derived from it, so adding
            # both counted a single reference twice.
            if value.path:
                assets.add(value.path)
            return
        for list_attr in (
            "explicitItems",
            "addedItems",
            "prependedItems",
            "appendedItems",
            "deletedItems",
            "orderedItems",
        ):
            if hasattr(value, list_attr):
                visit(getattr(value, list_attr))
        if isinstance(value, dict):
            for k, v in value.items():
                visit(k)
                visit(v)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    def visit_spec(spec) -> None:
        for field in spec.ListInfoKeys():
            try:
                visit(spec.GetInfo(field))
            except Exception as error:  # noqa: BLE001 - recorded below; see CheckErrorLog
                # An unreadable field could hide an asset reference, which is the
                # very thing this walk exists to find. Skipping is acceptable;
                # skipping silently is not.
                if error_log is not None:
                    error_log.record("authored_asset_paths", f"{layer.identifier}:{spec.path}.{field}", error)
                continue
        for child_spec in getattr(spec, "nameChildren", ()):
            visit_spec(child_spec)
        for prop_spec in getattr(spec, "properties", ()):
            visit_spec(prop_spec)

    for root in layer.rootPrims:
        visit_spec(root)
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


def analyze(stage_path: Path, prefix_style_pattern: str | None = None) -> dict:
    start = time.perf_counter()
    error_log = CheckErrorLog()
    prefix_style_re = re.compile(prefix_style_pattern) if prefix_style_pattern else None
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    root_layer = stage.GetRootLayer()
    # GetUsedLayers() order follows memory addresses, so it changes between
    # runs. Sorting keeps the order of recorded check_errors, which is capped,
    # identical across runs. Anonymous identifiers embed an address, so those
    # layers sort after file-backed ones and by their display name.
    used_layers = sorted(
        stage.GetUsedLayers(),
        key=lambda layer: (layer.anonymous, layer.GetDisplayName() if layer.anonymous else layer.identifier),
    )
    naming_policy = {
        "prefix_style": {
            "enabled": prefix_style_re is not None,
            "pattern": prefix_style_pattern,
        },
    }

    report = {
        "stage": str(stage_path),
        "root_layer": root_layer.identifier,
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
            "duplicate_sibling_names": [],
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
            "direct_binding_targets_missing": [],
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

    type_counts: Counter[str] = Counter()
    child_names_by_parent: dict[str, list[str]] = defaultdict(list)
    # A list, not a set: materials_without_surface_output is reported in this
    # order, and set order changes with the per-process string hash seed.
    material_paths: list[str] = []
    mesh_paths: list[str] = []
    geom_subset_paths: list[str] = []
    subsets_by_parent_mesh: dict[str, list[str]] = defaultdict(list)
    computed_materials: Counter[str] = Counter()

    # Prototypes are walked in a stable order and reported under stable names;
    # lookups keep using the real prim paths. See PrototypePaths.
    prototype_paths = PrototypePaths(stage)
    shown = prototype_paths.stable
    prims_to_scan = list(stage.Traverse())
    prototypes = prototype_paths.ordered()
    for prototype in prototypes:
        prims_to_scan.extend(Usd.PrimRange(prototype))
    report["prototype_count"] = len(prototypes)

    for prim in prims_to_scan:
        path = str(prim.GetPath())
        name = prim.GetName()
        type_name = prim.GetTypeName() or "<untyped>"
        type_counts[type_name] += 1
        child_names_by_parent[str(prim.GetParent().GetPath()) if prim.GetParent() else "/"].append(name)

        internal_generated = name == "__class__" or name.startswith("__Prototype_")
        if not internal_generated:
            # No separate whitespace check: SdfPath rejects prim names containing
            # whitespace on authoring and on read (.usda and .usdc), so a composed
            # stage cannot hold one -- and the identifier-char check below would
            # flag it regardless. tests/test_scene_audit.py pins that assumption.
            if re.search(r"[^A-Za-z0-9_]", name):
                report["naming"]["suspicious_count"] += 1
                add_example(report["naming"]["examples"]["non_ascii_identifier_chars"], shown(path))
            if "__" in name:
                report["naming"]["suspicious_count"] += 1
                add_example(report["naming"]["examples"]["double_underscore"], shown(path))
            if prefix_style_re and not is_expected_prefix_name(name, prefix_style_re):
                report["naming"]["non_prefix_style_count"] += 1
                add_example(report["naming"]["examples"]["non_prefix_style"], shown(path))

        if prim.IsA(UsdShade.Material):
            material_paths.append(path)
        if prim.IsA(UsdShade.Shader):
            report["materials"]["shader_prim_count"] += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(path)
        if prim.IsA(UsdGeom.Subset):
            geom_subset_paths.append(path)
            parent = prim.GetParent()
            while parent and parent.IsValid() and not parent.IsA(UsdGeom.Mesh):
                parent = parent.GetParent()
            if parent and parent.IsValid() and parent.IsA(UsdGeom.Mesh):
                subsets_by_parent_mesh[str(parent.GetPath())].append(path)

        targets = direct_material_targets(prim)
        if targets:
            report["materials"]["direct_binding_relation_count"] += len(targets)
            for target in targets:
                target_prim = stage.GetPrimAtPath(target)
                if not target_prim:
                    add_example(
                        report["materials"]["direct_binding_targets_missing"], f"{shown(path)} -> {shown(target)}"
                    )
                elif not target_prim.IsA(UsdShade.Material):
                    add_example(
                        report["materials"]["direct_binding_targets_not_material"],
                        f"{shown(path)} -> {shown(target)} ({target_prim.GetTypeName() or '<untyped>'})",
                    )

    for parent, names in child_names_by_parent.items():
        counts = Counter(names)
        for name, count in counts.items():
            if count > 1:
                add_example(report["naming"]["duplicate_sibling_names"], f"{shown(parent)}/{name} x{count}")
        by_lower: dict[str, set[str]] = defaultdict(set)
        for name in names:
            by_lower[name.lower()].add(name)
        for originals in by_lower.values():
            if len(originals) > 1:
                add_example(
                    report["naming"]["case_collision_names"],
                    f"{shown(parent)}: {', '.join(sorted(originals))}",
                )

    mesh_bindings = computed_material_bindings(stage, mesh_paths)
    subset_bindings = computed_material_bindings(stage, geom_subset_paths)
    computed_materials.update(m for m in mesh_bindings.values() if m)
    computed_materials.update(m for m in subset_bindings.values() if m)
    mesh_has_material = {path for path, m in mesh_bindings.items() if m}
    subset_has_material = {path for path, m in subset_bindings.items() if m}

    report["materials"]["mesh_with_computed_material"] = len(mesh_has_material)
    report["materials"]["mesh_without_computed_material"] = len(mesh_paths) - len(mesh_has_material)
    for mesh_path, material_path in mesh_bindings.items():
        if material_path is None:
            add_example(report["materials"]["mesh_without_material_examples"], shown(mesh_path))

    report["materials"]["geom_subset_with_computed_material"] = len(subset_has_material)
    report["materials"]["geom_subset_without_computed_material"] = len(geom_subset_paths) - len(subset_has_material)
    for subset_path, material_path in subset_bindings.items():
        if material_path is None:
            add_example(report["materials"]["geom_subset_without_material_examples"], shown(subset_path))
    for mesh_path in mesh_paths:
        if mesh_path in mesh_has_material:
            continue
        if any(subset_path in subset_has_material for subset_path in subsets_by_parent_mesh.get(mesh_path, [])):
            continue
        report["materials"]["mesh_without_mesh_or_subset_material"] += 1
        add_example(report["materials"]["mesh_without_mesh_or_subset_material_examples"], shown(mesh_path))

    for material_path in material_paths:
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        surface = material.GetSurfaceOutput()
        if not surface or not surface.HasConnectedSource():
            report["materials"]["materials_without_surface_output_count"] += 1
            add_example(report["materials"]["materials_without_surface_output"], shown(material_path))

    missing_assets: dict[str, set[str]] = defaultdict(set)
    unverifiable_assets: dict[str, set[str]] = defaultdict(set)
    for layer in used_layers:
        for asset in authored_asset_paths(layer, error_log):
            status, identifier = classify_authored_asset(layer, asset)
            report["assets"]["authored_asset_count"] += 1
            if status == "resolved":
                report["assets"]["resolved_asset_count"] += 1
            elif status == "missing":
                missing_assets[identifier].add(layer.identifier)
            else:
                unverifiable_assets[identifier].add(layer.identifier)

    for asset, layers in sorted(missing_assets.items()):
        add_example(
            report["assets"]["missing_authored_assets"],
            f"{asset} (authored from {len(layers)} layer(s))",
            max_items=120,
        )

    for asset, layers in sorted(unverifiable_assets.items()):
        add_example(
            report["assets"]["unverifiable_assets"],
            f"{asset} (authored from {len(layers)} layer(s))",
            max_items=120,
        )

    report["prims"] = {
        "total": sum(type_counts.values()),
        "by_type": dict(type_counts.most_common()),
    }
    report["materials"]["material_prim_count"] = len(material_paths)
    report["materials"]["mesh_count"] = len(mesh_paths)
    report["materials"]["geom_subset_count"] = len(geom_subset_paths)
    report["materials"]["bound_materials_used_by_mesh_count"] = len(computed_materials)
    report["materials"]["unbound_material_prim_count"] = len(set(material_paths) - set(computed_materials))
    report["assets"]["missing_authored_asset_count"] = len(missing_assets)
    report["assets"]["unverifiable_asset_count"] = len(unverifiable_assets)
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
