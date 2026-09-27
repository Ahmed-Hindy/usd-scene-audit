"""The per-layer asset walk finds every asset path a layer authors, and only those.

Issue #55: ``asset[]`` values and everything inside variants were never walked,
so missing textures there went unreported, while a reference a layer *deletes*
was reported as a missing asset.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Sdf

from usd_scene_audit import scene

FIXTURE = Path(__file__).parent / "fixtures" / "golden_corpus" / "assets_variants.usda"


def _walked_names() -> set[str]:
    return {Path(path).name for path in scene.authored_asset_paths(Sdf.Layer.FindOrOpen(str(FIXTURE)))}


def _missing_names(report: dict) -> list[str]:
    return sorted(Path(entry.split(" (authored from")[0]).name for entry in report["assets"]["missing_authored_assets"])


@pytest.mark.parametrize(
    "name",
    [
        "array_missing.exr",
        "sampled_missing.exr",
        "hero_missing.exr",
        "background_missing.exr",
        "background_ref_missing.usda",
        "nested_variant_missing.exr",
    ],
    ids=[
        "asset-array",
        "asset-array-time-sample",
        "selected-variant",
        "unselected-variant",
        "reference-on-variant",
        "nested-variant",
    ],
)
def test_authored_asset_paths_are_found(name: str) -> None:
    assert name in _walked_names()


@pytest.mark.parametrize("name", ["deleted_only.usda", "reordered_only.usda"], ids=["delete", "reorder"])
def test_list_op_entries_that_add_nothing_are_not_authored_assets(name: str) -> None:
    """delete removes a reference and reorder only reorders ones added elsewhere."""
    assert name not in _walked_names()


def test_scene_report_lists_every_missing_asset_and_nothing_else() -> None:
    report = scene.analyze(FIXTURE)

    assert _missing_names(report) == [
        "array_missing.exr",
        "background_missing.exr",
        "background_ref_missing.usda",
        "hero_missing.exr",
        "nested_variant_missing.exr",
        "sampled_missing.exr",
    ]
    assert report["assets"]["resolved_asset_count"] == 1  # tex/found.exr, inside the asset[] value
    assert report["assets"]["authored_asset_count"] == 7
    assert report["check_errors"]["count"] == 0


def test_a_layer_that_adds_and_deletes_the_same_reference_still_counts_it(tmp_path: Path) -> None:
    """Only the delete entry is ignored; the prepend that adds the reference is still walked."""
    layer = tmp_path / "stage.usda"
    layer.write_text(
        '#usda 1.0\ndef "Root" (\n    delete references = @./gone.usda@\n'
        "    prepend references = @./kept.usda@\n)\n{\n}\n",
        encoding="utf-8",
    )

    assert _missing_names(scene.analyze(layer)) == ["kept.usda"]
