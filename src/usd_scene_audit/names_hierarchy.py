"""Focused OpenUSD naming and hierarchy audit for a composed stage."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade

# TODO(#22): CheckErrorLog belongs in a shared module once one exists; it is not
# geometry-specific.
from usd_scene_audit.geometry import CheckErrorLog, PrototypePaths


MAX_EXAMPLES = 80
CONTAINER_NAMES = {"geo", "mtl", "materials", "render", "proxy", "lod", "payload"}
KNOWN_ALLOWED = {"__class__"} | CONTAINER_NAMES


def add_example(report: dict, key: str, value: str, limit: int = MAX_EXAMPLES) -> None:
    """Append a bounded example to a list entry."""
    report.setdefault(key, [])
    if len(report[key]) < limit:
        report[key].append(value)


def note_oddity(report: dict, oddity_counts: Counter[str], key: str, value: str) -> None:
    """Count an oddity and keep a bounded example."""
    oddity_counts[key] += 1
    add_example(report["name_oddities"], key, value)


def normalized_path(path: str) -> str:
    """Strip OpenUSD generated prototype roots for easier reading."""
    return re.sub(r"^/__Prototype_\d+", "/<prototype>", path)


def is_internal_generated(name: str) -> bool:
    """Return true for OpenUSD-generated prototype/class prims."""
    return name.startswith("__Prototype_") or name == "__class__"


def is_prefix_style_name(name: str, prefix_style_re: re.Pattern[str]) -> bool:
    """Return true for a loose vendor/package prefix naming convention."""
    if name in KNOWN_ALLOWED or is_internal_generated(name):
        return True
    return bool(prefix_style_re.fullmatch(name))


def prims_with_prototypes(stage: Usd.Stage) -> list[Usd.Prim]:
    """Return normal traversal plus prototype contents, prototypes in stable order.

    Paths are reported with prototype roots normalized to ``/<prototype>``, but
    their order still followed OpenUSD's per-run prototype numbering.
    """
    prims = list(stage.Traverse())
    for prototype in PrototypePaths(stage).ordered():
        prims.extend(Usd.PrimRange(prototype))
    return prims


def new_report(
    stage_path: Path, stage: Usd.Stage, prims: list[Usd.Prim], prefix_style_pattern: str | None, prefix_enabled: bool
) -> dict:
    """Return the report skeleton, with every count at zero and every list empty."""
    return {
        "stage": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prototype_count": len(stage.GetPrototypes()),
        "total_prims_including_prototypes": len(prims),
        "type_counts": {},
        "name_oddities": defaultdict(list),
        "naming_policy": {
            "prefix_style": {
                "enabled": prefix_enabled,
                "pattern": prefix_style_pattern,
            },
        },
        "name_counts": {},
        "hierarchy": {
            "max_depth": 0,
            "max_depth_examples": [],
            "single_child_chain_count": 0,
            "single_child_chain_examples": [],
            "leaf_scope_count": 0,
            "leaf_scope_examples": [],
            "empty_scope_count": 0,
            "empty_scope_examples": [],
            "same_name_parent_child_count": 0,
            "same_name_parent_child_examples": [],
            "xform_mesh_same_name_count": 0,
            "xform_mesh_same_name_examples": [],
            "mesh_with_mesh_child_count": 0,
            "mesh_with_mesh_child_examples": [],
            "mesh_parent_type_counts": {},
            "deep_collision_like_mesh_count": 0,
            "deep_collision_like_mesh_examples": [],
        },
        "check_errors": {"count": 0, "examples": []},
        "elapsed_seconds": None,
    }


def note_hierarchy(hierarchy: dict, key: str, example: str) -> None:
    """Count a hierarchy finding and keep a bounded example."""
    hierarchy[f"{key}_count"] += 1
    add_example(hierarchy, f"{key}_examples", example)


def valid_parent(prim: Usd.Prim) -> Usd.Prim | None:
    """Return the prim's parent, or None at the pseudo-root boundary."""
    parent = prim.GetParent()
    return parent if parent and parent.IsValid() else None


def check_name(
    name: str,
    parent_name: str,
    norm_path: str,
    note,
    prefix_style_re: re.Pattern[str] | None,
) -> None:
    """Note naming-convention oddities for one prim name."""
    if prefix_style_re and not is_prefix_style_name(name, prefix_style_re):
        note("non_prefix_style", norm_path)
    if name.endswith("_"):
        note("trailing_underscore", norm_path)
    if re.search(r"_COL_$", name):
        note("trailing_col_underscore", norm_path)
    if (
        parent_name
        and name.startswith(parent_name + "_")
        and (re.search(r"_C(?:_\d+)?$", name) or re.search(r"_CO$", name))
    ):
        note("truncated_collision_suffix", norm_path)
    if re.search(r"tunel", name, re.IGNORECASE):
        note("possible_tunnel_typo", norm_path)
    if re.search(r"foiliage", name, re.IGNORECASE):
        note("possible_foliage_typo", norm_path)
    if re.search(r"exterior", name):
        note("lowercase_token_in_name", norm_path)


def record_depth(hierarchy: dict, depth: int, norm_path: str) -> None:
    """Track the deepest prim paths seen so far."""
    if depth > hierarchy["max_depth"]:
        hierarchy["max_depth"] = depth
        hierarchy["max_depth_examples"] = [norm_path]
    elif depth == hierarchy["max_depth"]:
        add_example(hierarchy, "max_depth_examples", norm_path, 10)


def check_scope_and_chain(prim: Usd.Prim, name: str, norm_path: str, hierarchy: dict) -> None:
    """Note empty or leaf Scopes and single-child chains that repeat the parent name."""
    children = list(prim.GetChildren())
    if prim.GetTypeName() == "Scope" and not children:
        note_hierarchy(hierarchy, "leaf_scope", norm_path)
        if not prim.HasAuthoredReferences() and not prim.HasPayload():
            note_hierarchy(hierarchy, "empty_scope", norm_path)

    if len(children) == 1 and not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdShade.Material):
        child_name = children[0].GetName()
        if child_name == name or child_name.startswith(name):
            note_hierarchy(hierarchy, "single_child_chain", f"{norm_path} -> {child_name}")


def check_mesh_placement(
    prim: Usd.Prim, name: str, norm_path: str, parent: Usd.Prim | None, depth: int, hierarchy: dict
) -> None:
    """Note meshes whose parent or depth suggests a hierarchy mistake."""
    parent_type = (parent.GetTypeName() if parent else "<none>") or "<untyped>"
    parent_types = hierarchy["mesh_parent_type_counts"]
    parent_types[parent_type] = parent_types.get(parent_type, 0) + 1
    if parent and parent.GetName() == name and parent.GetTypeName() == "Xform":
        note_hierarchy(hierarchy, "xform_mesh_same_name", norm_path)
    if parent and parent.IsA(UsdGeom.Mesh):
        note_hierarchy(hierarchy, "mesh_with_mesh_child", norm_path)
    if "_COL" in name and depth >= 5:
        note_hierarchy(hierarchy, "deep_collision_like_mesh", norm_path)


def check_prim(prim: Usd.Prim, report: dict, note, prefix_style_re: re.Pattern[str] | None) -> None:
    """Run every per-prim naming and hierarchy check."""
    path = str(prim.GetPath())
    norm_path = normalized_path(path)
    name = prim.GetName()
    parent = valid_parent(prim)
    hierarchy = report["hierarchy"]

    depth = len([part for part in path.split("/") if part])
    record_depth(hierarchy, depth, norm_path)
    check_name(name, parent.GetName() if parent else "", norm_path, note, prefix_style_re)
    if parent and parent.GetName() == name and not is_internal_generated(parent.GetName()):
        note_hierarchy(hierarchy, "same_name_parent_child", norm_path)
    check_scope_and_chain(prim, name, norm_path, hierarchy)
    if prim.IsA(UsdGeom.Mesh):
        check_mesh_placement(prim, name, norm_path, parent, depth, hierarchy)


def sibling_oddities(parent_child_names: dict[str, list[str]], oddity_counts: Counter[str]) -> tuple[list, list]:
    """Return (duplicate sibling examples, case collision examples), counting every group.

    These two counts are one per colliding sibling group, not one per prim
    (unlike note_oddity()), so each count pairs with one example string. Counts
    are exact; only the example lists are capped.
    """
    duplicate_sibling_examples: list[str] = []
    case_collision_examples: list[str] = []
    for parent_path, child_names in parent_child_names.items():
        by_lower: dict[str, set[str]] = defaultdict(set)
        for child_name in child_names:
            by_lower[child_name.lower()].add(child_name)
        for variants in by_lower.values():
            if len(variants) > 1:
                oddity_counts["case_collision_names"] += 1
                if len(case_collision_examples) < MAX_EXAMPLES:
                    case_collision_examples.append(f"{normalized_path(parent_path)}: {', '.join(sorted(variants))}")
        # Sdf/Usd disallows true duplicate siblings, but keep the check for completeness.
        for child_name, count in Counter(child_names).items():
            if count > 1:
                oddity_counts["duplicate_sibling_names"] += 1
                if len(duplicate_sibling_examples) < MAX_EXAMPLES:
                    duplicate_sibling_examples.append(f"{normalized_path(parent_path)}/{child_name} x{count}")
    return duplicate_sibling_examples, case_collision_examples


def analyze(stage_path: Path, *, prefix_style_pattern: str | None = None) -> dict:
    """Audit names and hierarchy while ignoring material reference resolution."""
    started = time.perf_counter()
    # No check in this module catches broadly today. The block is still reported
    # so all three audits expose the same contract and a gate can read one key.
    error_log = CheckErrorLog()
    prefix_style_re = re.compile(prefix_style_pattern) if prefix_style_pattern else None
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    prims = prims_with_prototypes(stage)
    report = new_report(stage_path, stage, prims, prefix_style_pattern, prefix_style_re is not None)
    type_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    oddity_counts: Counter[str] = Counter()
    parent_child_names: dict[str, list[str]] = defaultdict(list)

    def note(key: str, value: str) -> None:
        note_oddity(report, oddity_counts, key, value)

    for prim in prims:
        name = prim.GetName()
        type_counts[prim.GetTypeName() or "<untyped>"] += 1
        name_counts[name] += 1
        parent = valid_parent(prim)
        if parent:
            parent_child_names[str(parent.GetPath())].append(name)
        if not is_internal_generated(name):
            check_prim(prim, report, note, prefix_style_re)

    duplicate_sibling_examples, case_collision_examples = sibling_oddities(parent_child_names, oddity_counts)
    report["name_oddities"]["duplicate_sibling_names"] = duplicate_sibling_examples
    report["name_oddities"]["case_collision_names"] = case_collision_examples
    report["name_oddity_counts"] = dict(oddity_counts)
    report["type_counts"] = dict(type_counts.most_common())
    report["name_counts"] = {
        name: count
        for name, count in name_counts.most_common()
        if count > 20 and not is_internal_generated(name) and name not in CONTAINER_NAMES
    }
    report["name_oddities"] = dict(report["name_oddities"])
    report["check_errors"] = error_log.as_report()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
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
