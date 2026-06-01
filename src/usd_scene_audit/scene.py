"""Audit a composed OpenUSD stage for naming and material assignment issues."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdShade


MAX_EXAMPLES = 40

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


def authored_asset_paths(layer: Sdf.Layer) -> set[str]:
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
            if value.path:
                assets.add(value.path)
            if value.resolvedPath:
                assets.add(value.resolvedPath)
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
            except Exception:
                continue
        for child_spec in getattr(spec, "nameChildren", ()):
            visit_spec(child_spec)
        for prop_spec in getattr(spec, "properties", ()):
            visit_spec(prop_spec)

    for root in layer.rootPrims:
        visit_spec(root)
    return assets


def resolve_authored_asset(layer: Sdf.Layer, asset_path: str) -> str | None:
    """Resolve a path authored in a USD layer into an absolute filesystem path."""
    if not asset_path or asset_path.startswith(("http://", "https://", "omniverse://")):
        return None
    if os.path.isabs(asset_path):
        return os.path.normpath(asset_path)
    layer_path = layer.realPath or layer.identifier
    if not layer_path or layer_path.startswith("anon:"):
        return None
    return os.path.normpath(os.path.join(os.path.dirname(layer_path), asset_path))


def is_expected_prefix_name(name: str, prefix_style_re: re.Pattern[str]) -> bool:
    """Check a loose vendor/package prefix naming convention."""
    if name in COMMON_CONTAINER_NAMES:
        return True
    if name == "__class__" or name.startswith("__Prototype_"):
        return True
    return bool(prefix_style_re.fullmatch(name))


def analyze(stage_path: Path, prefix_style_pattern: str | None = None) -> dict:
    start = time.perf_counter()
    prefix_style_re = re.compile(prefix_style_pattern) if prefix_style_pattern else None
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    root_layer = stage.GetRootLayer()
    used_layers = stage.GetUsedLayers()
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
            "missing_authored_asset_count": 0,
            "missing_authored_assets": [],
        },
        "elapsed_seconds": None,
    }

    type_counts: Counter[str] = Counter()
    child_names_by_parent: dict[str, list[str]] = defaultdict(list)
    material_paths: set[str] = set()
    mesh_paths: list[str] = []
    geom_subset_paths: list[str] = []
    subsets_by_parent_mesh: dict[str, list[str]] = defaultdict(list)
    computed_materials: Counter[str] = Counter()
    binding_api_cache: dict[str, UsdShade.MaterialBindingAPI] = {}

    prims_to_scan = list(stage.Traverse())
    prototypes = list(stage.GetPrototypes())
    for prototype in prototypes:
        prims_to_scan.extend(list(Usd.PrimRange(prototype)))
    report["prototype_count"] = len(prototypes)

    for prim in prims_to_scan:
        path = str(prim.GetPath())
        name = prim.GetName()
        type_name = prim.GetTypeName() or "<untyped>"
        type_counts[type_name] += 1
        child_names_by_parent[str(prim.GetParent().GetPath()) if prim.GetParent() else "/"].append(name)

        internal_generated = name == "__class__" or name.startswith("__Prototype_")
        if not internal_generated:
            if re.search(r"\s", name):
                report["naming"]["suspicious_count"] += 1
                add_example(report["naming"]["examples"]["contains_whitespace"], path)
            if re.search(r"[^A-Za-z0-9_]", name):
                report["naming"]["suspicious_count"] += 1
                add_example(report["naming"]["examples"]["non_ascii_identifier_chars"], path)
            if "__" in name:
                report["naming"]["suspicious_count"] += 1
                add_example(report["naming"]["examples"]["double_underscore"], path)
            if prefix_style_re and not is_expected_prefix_name(name, prefix_style_re):
                report["naming"]["non_prefix_style_count"] += 1
                add_example(report["naming"]["examples"]["non_prefix_style"], path)

        if prim.IsA(UsdShade.Material):
            material_paths.add(path)
        if prim.IsA(UsdShade.Shader):
            report["materials"]["shader_prim_count"] += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(path)
        if prim.GetTypeName() == "GeomSubset":
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
                    add_example(report["materials"]["direct_binding_targets_missing"], f"{path} -> {target}")
                elif not target_prim.IsA(UsdShade.Material):
                    add_example(
                        report["materials"]["direct_binding_targets_not_material"],
                        f"{path} -> {target} ({target_prim.GetTypeName() or '<untyped>'})",
                    )

    for parent, names in child_names_by_parent.items():
        counts = Counter(names)
        for name, count in counts.items():
            if count > 1:
                add_example(report["naming"]["duplicate_sibling_names"], f"{parent}/{name} x{count}")
        by_lower: dict[str, set[str]] = defaultdict(set)
        for name in names:
            by_lower[name.lower()].add(name)
        for lower_name, originals in by_lower.items():
            if len(originals) > 1:
                add_example(
                    report["naming"]["case_collision_names"],
                    f"{parent}: {', '.join(sorted(originals))}",
                )

    mesh_has_material: set[str] = set()
    subset_has_material: set[str] = set()
    for mesh_path in mesh_paths:
        prim = stage.GetPrimAtPath(mesh_path)
        binding_api = binding_api_cache.setdefault(mesh_path, UsdShade.MaterialBindingAPI(prim))
        material, _relationship = binding_api.ComputeBoundMaterial()
        if material and material.GetPrim():
            report["materials"]["mesh_with_computed_material"] += 1
            mesh_has_material.add(mesh_path)
            computed_materials[str(material.GetPath())] += 1
        else:
            report["materials"]["mesh_without_computed_material"] += 1
            add_example(report["materials"]["mesh_without_material_examples"], mesh_path)

    for subset_path in geom_subset_paths:
        prim = stage.GetPrimAtPath(subset_path)
        binding_api = binding_api_cache.setdefault(subset_path, UsdShade.MaterialBindingAPI(prim))
        material, _relationship = binding_api.ComputeBoundMaterial()
        if material and material.GetPrim():
            report["materials"]["geom_subset_with_computed_material"] += 1
            subset_has_material.add(subset_path)
            computed_materials[str(material.GetPath())] += 1
        else:
            report["materials"]["geom_subset_without_computed_material"] += 1
            add_example(report["materials"]["geom_subset_without_material_examples"], subset_path)

    for mesh_path in mesh_paths:
        if mesh_path in mesh_has_material:
            continue
        if any(subset_path in subset_has_material for subset_path in subsets_by_parent_mesh.get(mesh_path, [])):
            continue
        report["materials"]["mesh_without_mesh_or_subset_material"] += 1
        add_example(report["materials"]["mesh_without_mesh_or_subset_material_examples"], mesh_path)

    for material_path in material_paths:
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        surface = material.GetSurfaceOutput()
        if not surface or not surface.HasConnectedSource():
            report["materials"]["materials_without_surface_output_count"] += 1
            add_example(report["materials"]["materials_without_surface_output"], material_path)

    missing_assets: dict[str, set[str]] = defaultdict(set)
    for layer in used_layers:
        for asset in authored_asset_paths(layer):
            resolved = resolve_authored_asset(layer, asset)
            if not resolved:
                continue
            report["assets"]["authored_asset_count"] += 1
            if not os.path.exists(resolved):
                missing_assets[resolved].add(layer.identifier)

    for asset, layers in sorted(missing_assets.items()):
        add_example(
            report["assets"]["missing_authored_assets"],
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
    report["materials"]["unbound_material_prim_count"] = len(material_paths - set(computed_materials))
    report["assets"]["missing_authored_asset_count"] = len(missing_assets)
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
