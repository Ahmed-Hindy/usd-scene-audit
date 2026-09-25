"""Regression tests for reading mesh attributes at a usable time code.

Mesh attributes used to be read at ``Usd.TimeCode.Default()``, which resolves
only an attribute's default value. Deforming geometry authors ``points`` purely
as time samples, so those meshes resolved to ``None`` and were reported as
missing their points, which in turn made every face-vertex index look out of
range.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pxr import Usd, UsdGeom

from usd_scene_audit.geometry import (
    FaceAnalysisCache,
    PhaseTimer,
    analyze,
    describe_time_code,
    mesh_record,
    resolve_time_code,
)

CLEAN_FIXTURES = [
    "animated_points_valid.usda",
    "animated_normals_primvars.usda",
    "static_mesh_clean.usda",
]


def audit(path, **kwargs):
    """Run the geometry audit with the documented default thresholds."""
    return analyze(path, 1e-12, 1e6, 1e-4, **kwargs)


@pytest.mark.parametrize("fixture_name", CLEAN_FIXTURES)
def test_valid_stages_report_no_findings(stage_path, fixture_name: str) -> None:
    """Static and time-sampled geometry that is valid must produce zero findings."""
    report = audit(stage_path(fixture_name))

    assert report["mesh_count"] == 1
    assert report["summary_counts"] == {}
    assert report["serious_geometry_failures"] == 0


def test_time_sampled_points_resolve_to_real_data(stage_path) -> None:
    """The deforming mesh must be seen as having its points, not as missing them."""
    report = audit(stage_path("animated_points_valid.usda"))

    record = report["worst_meshes"][0]
    assert record["point_count"] == 3
    assert record["face_count"] == 1
    assert record["issues"] == {}


def test_out_of_range_indices_are_still_reported(stage_path) -> None:
    """Widening the time-code read must not suppress genuine index corruption."""
    report = audit(stage_path("static_mesh_out_of_range_indices.usda"))

    assert report["summary_counts"]["out_of_range_face_vertex_indices"] == 1
    assert report["serious_geometry_failures"] >= 1


def test_missing_points_are_still_reported(stage_path) -> None:
    """A mesh with no authored points at any time is still a real defect."""
    report = audit(stage_path("static_mesh_missing_points.usda"))

    assert report["summary_counts"]["missing_points"] == 1
    assert report["serious_geometry_failures"] >= 1


def test_topology_change_across_time_reports_no_false_findings(stage_path) -> None:
    """Time-varying defects are out of scope here, but must not fire falsely.

    This stage is valid at its earliest time sample and corrupt later on. Until
    time-sampled auditing lands, the correct behaviour is to report nothing
    rather than to report the wrong thing.
    """
    report = audit(stage_path("animated_topology_change.usda"))

    assert report["summary_counts"] == {}


# --------------------------------------------------------------- time code plumbing


def test_resolve_time_code_falls_back_to_earliest() -> None:
    """With no frame and no stage time range, resolve at the earliest sample."""
    assert resolve_time_code(None).IsEarliestTime()
    assert not resolve_time_code(None).IsDefault()


def test_resolve_time_code_prefers_authored_start_time(stage_path) -> None:
    """A stage that declares its time range is audited at the start of that range.

    One concrete time code keeps every attribute on the same frame. EarliestTime
    resolves each attribute at its own first sample, which makes checks that
    compare two attributes report mismatches that exist at no real frame.
    """
    stage = Usd.Stage.Open(str(stage_path("animated_preroll_extent.usda")))

    time_code = resolve_time_code(None, stage)

    assert not time_code.IsEarliestTime()
    assert time_code.GetValue() == 2.0


def test_resolve_time_code_ignores_stage_when_frame_is_explicit(stage_path) -> None:
    """An explicit frame always wins over the stage's authored range."""
    stage = Usd.Stage.Open(str(stage_path("animated_preroll_extent.usda")))

    assert resolve_time_code(9.0, stage).GetValue() == 9.0


def test_stage_without_authored_range_uses_earliest(stage_path) -> None:
    """A static stage with no time metadata still falls back to earliest."""
    stage = Usd.Stage.Open(str(stage_path("static_mesh_clean.usda")))

    assert resolve_time_code(None, stage).IsEarliestTime()


def test_preroll_sampling_does_not_create_false_findings(stage_path) -> None:
    """Attributes sampled over different frame ranges must not be cross-compared."""
    report = audit(stage_path("animated_preroll_extent.usda"))

    assert report["time_code"] == 2.0
    assert report["summary_counts"] == {}


def test_time_sampled_authoring_defects_are_caught(stage_path) -> None:
    """Defects in time-sampled normals and extent must be reported.

    The clean animated fixtures assert an absence of findings, which a regressed
    normals or extent read satisfies for the wrong reason: at default time both
    resolve to None and their checks exit quietly. This fixture fails if either
    read stops resolving real data.
    """
    report = audit(stage_path("animated_mesh_authoring_defects.usda"))

    assert report["summary_counts"]["normals_length_mismatch"] == 1
    assert report["summary_counts"]["authored_extent_mismatch"] == 1


def test_resolve_time_code_honours_explicit_frame() -> None:
    """An explicit frame produces a numeric time code."""
    time_code = resolve_time_code(7.0)

    assert not time_code.IsEarliestTime()
    assert time_code.GetValue() == 7.0


def test_describe_time_code_is_json_friendly() -> None:
    """Reports need a serializable description of the evaluated time code."""
    assert describe_time_code(resolve_time_code(None)) == "earliest"
    assert describe_time_code(resolve_time_code(2.0)) == 2.0
    assert describe_time_code(Usd.TimeCode.Default()) == "default"


def test_report_records_evaluated_time_code(stage_path) -> None:
    """The evaluated time code belongs in the report for reproducibility."""
    # This fixture declares startTimeCode = 1, so the default run reports it.
    default_run = audit(stage_path("animated_points_valid.usda"))
    framed_run = audit(stage_path("animated_points_valid.usda"), frame=3.0)
    static_run = audit(stage_path("static_mesh_clean.usda"))

    assert default_run["requested_frame"] is None
    assert default_run["time_code"] == 1.0
    assert framed_run["requested_frame"] == 3.0
    assert framed_run["time_code"] == 3.0
    assert static_run["time_code"] == "earliest"


def test_frame_selection_changes_the_data_read(stage_path) -> None:
    """A requested frame must actually drive attribute resolution."""
    stage = Usd.Stage.Open(str(stage_path("animated_points_valid.usda")))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))

    def bounds_at(frame: float):
        time_code = resolve_time_code(frame)
        record = mesh_record(
            prim,
            1e-12,
            1e6,
            1e-4,
            UsdGeom.XformCache(time_code),
            "numpy",
            "exhaustive",
            PhaseTimer(),
            FaceAnalysisCache(False),
            time_code,
        )
        assert record["point_count"] == 3
        return record["bounds"]

    # The fixture translates along +Z by one unit per frame.
    assert bounds_at(1.0)[0][2] == 0.0
    assert bounds_at(3.0)[0][2] == 2.0


def test_cli_accepts_and_records_frame(stage_path, tmp_path) -> None:
    """The --frame flag round-trips through the CLI into the JSON report."""
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "usd_scene_audit.geometry",
            str(stage_path("animated_points_valid.usda")),
            "--frame",
            "2",
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["requested_frame"] == 2.0
    assert report["time_code"] == 2.0
    assert report["summary_counts"] == {}
