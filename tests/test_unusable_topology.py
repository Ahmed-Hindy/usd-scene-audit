"""Missing or unusable topology is reported once, not once per index and per primvar.

When ``points`` was missing or mistyped the audit used a point count of zero, so
every face-vertex index was reported out of range and every vertex-interpolated
normal or primvar as the wrong length. One defect became thousands of findings
and buried everything else in the report. See issue #54.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from usd_scene_audit import geometry

CASCADE_CODES = {
    "out_of_range_face_vertex_indices",
    "normals_length_mismatch",
    "primvar_length_mismatch",
    "face_vertex_count_index_length_mismatch",
}


def _write(path: Path, *meshes: str) -> Path:
    body = "".join(f'def Mesh "{name}"\n{{\n    ' + "\n    ".join(lines) + "\n}\n" for name, *lines in meshes)
    path.write_text("#usda 1.0\n" + body, encoding="utf-8")
    return path


def _issues(report: dict) -> dict[str, dict[str, int]]:
    return {record["path"]: record["issues"] for record in report["worst_meshes"]}


def test_issue_54_stage_reports_only_root_causes(tmp_path: Path) -> None:
    """The reproduction from #54: two meshes, one defect each, two findings."""
    stage = _write(
        tmp_path / "stage.usda",
        (
            "Mistyped",
            "float[] points = [0, 0, 0, 1, 0, 0, 0, 1, 0]",
            "int[] faceVertexCounts = [3]",
            "int[] faceVertexIndices = [0, 1, 2]",
            'normal3f[] normals = [(0,0,1), (0,0,1), (0,0,1)] (interpolation = "vertex")',
            'texCoord2f[] primvars:st = [(0,0), (1,0), (0,1)] (interpolation = "vertex")',
        ),
        ("Missing", "int[] faceVertexCounts = [3, 3]", "int[] faceVertexIndices = [0, 1, 2, 2, 1, 3]"),
    )

    report = geometry.analyze(stage)

    assert _issues(report) == {"/Missing": {"missing_points": 1}, "/Mistyped": {"points_wrong_type": 1}}
    assert report["serious_geometry_failures"] == 2


def test_empty_points_do_not_cascade_into_index_findings(tmp_path: Path) -> None:
    stage = _write(
        tmp_path / "stage.usda",
        ("Empty", "point3f[] points = []", "int[] faceVertexCounts = [3]", "int[] faceVertexIndices = [0, 1, 2]"),
    )

    assert _issues(geometry.analyze(stage)) == {"/Empty": {"empty_points": 1}}


def test_negative_indices_are_still_reported_without_points(tmp_path: Path) -> None:
    """A negative index is wrong whatever the points are, so it is not part of the cascade."""
    stage = _write(
        tmp_path / "stage.usda", ("M", "int[] faceVertexCounts = [3]", "int[] faceVertexIndices = [0, -1, 2]")
    )

    issues = _issues(geometry.analyze(stage))["/M"]

    assert issues == {"missing_points": 1, "negative_face_vertex_indices": 1}


@pytest.mark.parametrize(
    ("override", "root_cause"),
    [
        ({"counts": None}, "missing_face_vertex_counts"),
        ({"counts": "float[] faceVertexCounts = [3.5]"}, "face_vertex_counts_wrong_type"),
        ({"indices": None}, "missing_face_vertex_indices"),
        ({"indices": "float[] faceVertexIndices = [0, 1, 2]"}, "face_vertex_indices_wrong_type"),
    ],
    ids=["counts-missing", "counts-mistyped", "indices-missing", "indices-mistyped"],
)
def test_unusable_face_arrays_do_not_cascade(tmp_path: Path, override, root_cause) -> None:
    """Uniform and faceVarying primvars, and the count/index length check, need usable face arrays."""
    lines = {
        "points": "point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]",
        "counts": "int[] faceVertexCounts = [3]",
        "indices": "int[] faceVertexIndices = [0, 1, 2]",
        "uniform": 'float[] primvars:faceTag = [1] (interpolation = "uniform")',
        "faceVarying": 'texCoord2f[] primvars:st = [(0,0), (1,0), (0,1)] (interpolation = "faceVarying")',
    }
    for key, value in override.items():
        if value is None:
            del lines[key]
        else:
            lines[key] = value
    stage = _write(tmp_path / "stage.usda", ("M", *lines.values()))

    issues = _issues(geometry.analyze(stage))["/M"]

    assert issues == {root_cause: 1}
    assert not CASCADE_CODES & set(issues)


def test_genuine_mismatches_are_still_reported(tmp_path: Path) -> None:
    """With usable topology, every check that this change can skip still runs."""
    stage = _write(
        tmp_path / "stage.usda",
        (
            "M",
            "point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]",
            "int[] faceVertexCounts = [3, 3]",
            "int[] faceVertexIndices = [0, 1, 5]",
            'normal3f[] normals = [(0,0,1)] (interpolation = "vertex")',
            'float[] primvars:faceTag = [1] (interpolation = "uniform")',
            'texCoord2f[] primvars:st = [(0,0)] (interpolation = "faceVarying")',
        ),
    )

    issues = _issues(geometry.analyze(stage))["/M"]

    assert issues["out_of_range_face_vertex_indices"] == 1
    assert issues["face_vertex_count_index_length_mismatch"] == 1
    assert issues["normals_length_mismatch"] == 1
    assert issues["primvar_length_mismatch"] == 2


def test_record_counts_stay_integers(tmp_path: Path) -> None:
    """Unknown counts only gate checks; the report still shows 0 for an absent attribute."""
    stage = _write(
        tmp_path / "stage.usda", ("M", "int[] faceVertexCounts = [3]", "int[] faceVertexIndices = [0, 1, 2]")
    )

    [record] = geometry.analyze(stage)["worst_meshes"]

    assert (record["point_count"], record["face_count"], record["face_vertex_index_count"]) == (0, 1, 3)


def test_expected_primvar_length_is_unknown_when_its_count_is_unknown() -> None:
    assert geometry.expected_primvar_length("vertex", None, 1, 3) is None
    assert geometry.expected_primvar_length("varying", None, 1, 3) is None
    assert geometry.expected_primvar_length("uniform", 3, None, 3) is None
    assert geometry.expected_primvar_length("faceVarying", 3, 1, None) is None
    assert geometry.expected_primvar_length("constant", None, None, None) == 1
