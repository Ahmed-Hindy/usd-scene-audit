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


def analyze(stage_path: Path, prefix_style_pattern: str | None = None) -> dict:
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
    report = {
        "stage": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prototype_count": len(stage.GetPrototypes()),
        "total_prims_including_prototypes": len(prims),
        "type_counts": {},
        "name_oddities": defaultdict(list),
        "naming_policy": {
            "prefix_style": {
                "enabled": prefix_style_re is not None,
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
        "check_errors": error_log.as_report(),
        "elapsed_seconds": None,
    }

    type_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    oddity_counts: Counter[str] = Counter()
    parent_child_names: dict[str, list[str]] = defaultdict(list)

    for prim in prims:
        path = str(prim.GetPath())
        norm_path = normalized_path(path)
        name = prim.GetName()
        type_name = prim.GetTypeName() or "<untyped>"
        type_counts[type_name] += 1
        name_counts[name] += 1
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_child_names[str(parent.GetPath())].append(name)

        if is_internal_generated(name):
            continue

        depth = len([part for part in path.split("/") if part])
        if depth > report["hierarchy"]["max_depth"]:
            report["hierarchy"]["max_depth"] = depth
            report["hierarchy"]["max_depth_examples"] = [norm_path]
        elif depth == report["hierarchy"]["max_depth"]:
            add_example(report["hierarchy"], "max_depth_examples", norm_path, 10)

        if prefix_style_re and not is_prefix_style_name(name, prefix_style_re):
            note_oddity(report, oddity_counts, "non_prefix_style", norm_path)
        if name.endswith("_"):
            note_oddity(report, oddity_counts, "trailing_underscore", norm_path)
        if re.search(r"_COL_$", name):
            note_oddity(report, oddity_counts, "trailing_col_underscore", norm_path)
        parent_name = parent.GetName() if parent and parent.IsValid() else ""
        if (
            parent_name
            and name.startswith(parent_name + "_")
            and (re.search(r"_C(?:_\d+)?$", name) or re.search(r"_CO$", name))
        ):
            note_oddity(report, oddity_counts, "truncated_collision_suffix", norm_path)
        if re.search(r"tunel", name, re.IGNORECASE):
            note_oddity(report, oddity_counts, "possible_tunnel_typo", norm_path)
        if re.search(r"foiliage", name, re.IGNORECASE):
            note_oddity(report, oddity_counts, "possible_foliage_typo", norm_path)
        if re.search(r"exterior", name):
            note_oddity(report, oddity_counts, "lowercase_token_in_name", norm_path)

        if parent and parent.IsValid() and parent.GetName() == name and not is_internal_generated(parent.GetName()):
            report["hierarchy"]["same_name_parent_child_count"] += 1
            add_example(report["hierarchy"], "same_name_parent_child_examples", norm_path)

        children = list(prim.GetChildren())
        if prim.GetTypeName() == "Scope":
            if not children:
                report["hierarchy"]["leaf_scope_count"] += 1
                add_example(report["hierarchy"], "leaf_scope_examples", norm_path)
            if not children and not prim.HasAuthoredReferences() and not prim.HasPayload():
                report["hierarchy"]["empty_scope_count"] += 1
                add_example(report["hierarchy"], "empty_scope_examples", norm_path)

        if len(children) == 1 and not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdShade.Material):
            child = children[0]
            if child.GetName() == name or child.GetName().startswith(name):
                report["hierarchy"]["single_child_chain_count"] += 1
                add_example(
                    report["hierarchy"],
                    "single_child_chain_examples",
                    f"{norm_path} -> {child.GetName()}",
                )

        if prim.IsA(UsdGeom.Mesh):
            parent_type = parent.GetTypeName() if parent and parent.IsValid() else "<none>"
            report["hierarchy"]["mesh_parent_type_counts"][parent_type or "<untyped>"] = (
                report["hierarchy"]["mesh_parent_type_counts"].get(parent_type or "<untyped>", 0) + 1
            )
            if parent and parent.IsValid() and parent.GetName() == name and parent.GetTypeName() == "Xform":
                report["hierarchy"]["xform_mesh_same_name_count"] += 1
                add_example(report["hierarchy"], "xform_mesh_same_name_examples", norm_path)
            if parent and parent.IsValid() and parent.IsA(UsdGeom.Mesh):
                report["hierarchy"]["mesh_with_mesh_child_count"] += 1
                add_example(report["hierarchy"], "mesh_with_mesh_child_examples", norm_path)
            if "_COL" in name and depth >= 5:
                report["hierarchy"]["deep_collision_like_mesh_count"] += 1
                add_example(report["hierarchy"], "deep_collision_like_mesh_examples", norm_path)

    repeated_names = {
        name: count
        for name, count in name_counts.most_common()
        if count > 20 and not is_internal_generated(name) and name not in CONTAINER_NAMES
    }

    duplicate_sibling_examples = []
    case_collision_examples = []
    for parent_path, child_names in parent_child_names.items():
        by_lower: dict[str, set[str]] = defaultdict(set)
        for child_name in child_names:
            by_lower[child_name.lower()].add(child_name)
        # These two counts are one per colliding sibling group, not one per prim
        # (unlike note_oddity()), so each count pairs with one example string.
        # Counts are exact; only the example lists are capped.
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

    report["name_oddities"]["duplicate_sibling_names"] = duplicate_sibling_examples
    report["name_oddities"]["case_collision_names"] = case_collision_examples
    report["name_oddity_counts"] = dict(oddity_counts)
    report["type_counts"] = dict(type_counts.most_common())
    report["name_counts"] = dict(repeated_names)
    report["name_oddities"] = dict(report["name_oddities"])
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
