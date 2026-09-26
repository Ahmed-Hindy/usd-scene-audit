"""The geometry audit's Python API: keyword options, deprecation shims, and validation.

``geometry.analyze()`` used to take three adjacent float thresholds
positionally, so swapping two was a silent behaviour change. Options are now
keyword-only; positional calls still work for one release with a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from usd_scene_audit import geometry, names_hierarchy, scene
from usd_scene_audit.geometry import FaceAnalysisCache, MeshCheckSettings, analyze_face_geometry, mesh_record

DEPRECATED = "positionally is deprecated"


def _mesh(stage_path, name: str = "static_mesh_clean.usda") -> tuple[Usd.Stage, Usd.Prim]:
    """Return the stage with its first mesh; the caller must keep the stage alive."""
    stage = Usd.Stage.Open(str(stage_path(name)))
    return stage, next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))


# ------------------------------------------------------------ deprecation shims


def test_positional_thresholds_still_work_with_a_warning(stage_path) -> None:
    """Old positional calls map onto the same keyword options and warn."""
    path = stage_path("static_mesh_clean.usda")

    with pytest.warns(DeprecationWarning, match=DEPRECATED) as warned:
        legacy = geometry.analyze(path, 1e-9, 5e5, 1e-3, "numpy", "fast", "face-hash")

    assert "zero_area_epsilon, huge_coord_threshold, extent_tolerance" in str(warned[0].message)
    assert warned[0].filename == __file__
    modern = geometry.analyze(
        path,
        zero_area_epsilon=1e-9,
        huge_coord_threshold=5e5,
        extent_tolerance=1e-3,
        geometry_engine="numpy",
        audit_mode="fast",
        mesh_cache_mode="face-hash",
    )
    for key in ("thresholds", "audit_mode", "geometry_engine", "mesh_cache", "summary_counts"):
        assert legacy[key] == modern[key]


def test_positional_and_keyword_for_the_same_option_is_an_error(stage_path) -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="multiple values for argument"):
        geometry.analyze(stage_path("static_mesh_clean.usda"), 1e-9, zero_area_epsilon=1e-9)


def test_too_many_positional_arguments_is_an_error(stage_path) -> None:
    with pytest.raises(TypeError, match="takes at most 8 positional arguments"):
        geometry.analyze(stage_path("static_mesh_clean.usda"), *range(8))


def test_legacy_mesh_record_call_folds_thresholds_into_settings(stage_path) -> None:
    """mesh_record()'s old eleven-argument form still works and matches the keyword form."""
    _stage, prim = _mesh(stage_path)
    xform_cache = UsdGeom.XformCache(geometry.default_time_code(prim))

    with pytest.warns(DeprecationWarning, match=DEPRECATED) as warned:
        legacy = mesh_record(
            prim, 1e-12, 1e6, 1e-4, xform_cache, "numpy", "exhaustive", geometry.PhaseTimer(), FaceAnalysisCache(False)
        )

    assert len(warned) == 1
    assert legacy == mesh_record(prim, settings=MeshCheckSettings(), xform_cache=xform_cache)


def test_loose_thresholds_passed_by_keyword_also_warn(stage_path) -> None:
    """Old keyword spellings fold into settings too, with their own warning."""
    _stage, prim = _mesh(stage_path)

    with pytest.warns(DeprecationWarning, match="pass settings=MeshCheckSettings") as warned:
        record = mesh_record(prim, zero_area_epsilon=1e-12, audit_mode="fast")

    assert len(warned) == 1
    assert record == mesh_record(prim, settings=MeshCheckSettings(audit_mode="fast"))


def test_legacy_mesh_record_rejects_settings_and_loose_thresholds_together(stage_path) -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="both settings and loose threshold"):
        mesh_record(_mesh(stage_path)[1], 1e-12, settings=MeshCheckSettings())


def test_legacy_analyze_face_geometry_call_still_works() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    counts = np.array([3], dtype=np.int64)
    indices = np.array([0, 1, 2], dtype=np.int64)

    with pytest.warns(DeprecationWarning, match=DEPRECATED):
        legacy, _ = analyze_face_geometry(counts, indices, points, 3, 1e-12, "numpy")

    assert legacy == analyze_face_geometry(counts, indices, points, 3, face_analysis_engine="numpy")[0]
    assert legacy["zero_area_triangles"] == 1


@pytest.mark.parametrize("module", [names_hierarchy, scene])
def test_prefix_pattern_is_still_accepted_positionally(module, stage_path) -> None:
    """prefix_style_pattern is the only option, so there is no swap hazard and no deprecation."""
    report = module.analyze(stage_path("static_mesh_clean.usda"), r"[A-Z]\w*")

    assert report["naming_policy"]["prefix_style"] == {"enabled": True, "pattern": r"[A-Z]\w*"}


# ------------------------------------------------------------------ defaults


def test_library_and_cli_share_every_default(stage_path, monkeypatch) -> None:
    """analyze() with no options must audit exactly like the CLI with no flags."""
    report = geometry.analyze(stage_path("static_mesh_clean.usda"))
    assert report["thresholds"] == {
        "zero_area_epsilon": geometry.DEFAULT_ZERO_AREA_EPSILON,
        "huge_coord_threshold": geometry.DEFAULT_HUGE_COORD_THRESHOLD,
        "extent_tolerance": geometry.DEFAULT_EXTENT_TOLERANCE,
    }
    assert report["geometry_engine"] == geometry.DEFAULT_GEOMETRY_ENGINE
    assert report["audit_mode"] == geometry.DEFAULT_AUDIT_MODE
    assert report["mesh_cache"]["mode"] == geometry.DEFAULT_MESH_CACHE_MODE

    captured = {}
    monkeypatch.setattr(sys, "argv", ["usd-geometry-audit", str(stage_path("static_mesh_clean.usda"))])
    with mock.patch.object(geometry, "analyze", side_effect=lambda path, **kwargs: captured.update(kwargs) or report):
        geometry.main()

    assert captured == {
        "zero_area_epsilon": geometry.DEFAULT_ZERO_AREA_EPSILON,
        "huge_coord_threshold": geometry.DEFAULT_HUGE_COORD_THRESHOLD,
        "extent_tolerance": geometry.DEFAULT_EXTENT_TOLERANCE,
        "geometry_engine": geometry.DEFAULT_GEOMETRY_ENGINE,
        "audit_mode": geometry.DEFAULT_AUDIT_MODE,
        "mesh_cache_mode": geometry.DEFAULT_MESH_CACHE_MODE,
        "frame": None,
    }


def test_mesh_record_defaults_read_transforms_at_the_attribute_time(tmp_path: Path) -> None:
    """With no xform_cache, mesh_record() evaluates transforms at the same time code as attributes."""
    path = tmp_path / "flip.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetStartTimeCode(2)
    stage.SetEndTimeCode(2)
    scale = UsdGeom.Xform.Define(stage, "/World").AddScaleOp()
    scale.Set(Gf.Vec3f(1, 1, 1))
    scale.Set(Gf.Vec3f(-1, 1, 1), 2)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()

    record = mesh_record(stage.GetPrimAtPath("/World/Mesh"))

    assert record["issues"] == geometry.analyze(path)["summary_counts"] == {"negative_transform_determinant": 1}


def test_collision_like_category_is_always_reported(stage_path) -> None:
    """category_issue_counts always carries a collision_like entry, so consumers need no key check."""
    report = geometry.analyze(stage_path("static_mesh_clean.usda"))

    assert report["category_issue_counts"] == {"collision_like": {}}
    assert report["likely_benign_collision_helper_warnings"] == 0


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"audit_mode": "thorough"}, "Unsupported audit mode"),
        ({"face_analysis_engine": "auto"}, "resolve 'auto'"),
        ({"zero_area_epsilon": -1.0}, "zero_area_epsilon must be a finite, non-negative number"),
        ({"huge_coord_threshold": float("nan")}, "huge_coord_threshold must be a finite"),
        ({"extent_tolerance": float("inf")}, "extent_tolerance must be a finite"),
        ({"extent_tolerance": "0.1"}, "extent_tolerance must be a finite"),
        ({"zero_area_epsilon": True}, "zero_area_epsilon must be a finite"),
    ],
)
def test_settings_reject_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        MeshCheckSettings(**kwargs)


def test_analyze_validates_thresholds_before_opening_the_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero_area_epsilon"):
        geometry.analyze(tmp_path / "does-not-exist.usda", zero_area_epsilon=-1.0)


def test_numba_engine_without_numba_fails_loudly(monkeypatch) -> None:
    """Without Numba, the numba engine must raise rather than report no deep-face findings."""
    monkeypatch.setattr(geometry, "get_numba_face_kernel", lambda: None)

    with pytest.raises(RuntimeError, match="Numba acceleration is not installed"):
        MeshCheckSettings(face_analysis_engine="numba")

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="Numba acceleration is not installed"):
        analyze_face_geometry(
            np.array([3], dtype=np.int64), np.array([0, 1, 1], dtype=np.int64), points, 3, face_analysis_engine="numba"
        )


def test_face_analysis_rejects_unknown_engines() -> None:
    with pytest.raises(ValueError, match="Unsupported face analysis engine"):
        analyze_face_geometry(np.array([3]), np.array([0, 1, 2]), np.zeros((3, 3)), 3, face_analysis_engine="auto")


# ------------------------------------------------------ face cache bookkeeping


def _triangle_arrays(indices):
    return np.array([3], dtype=np.int64), np.array(indices, dtype=np.int64), np.zeros((3, 3))


def test_inconsistent_face_arrays_count_a_miss_but_are_never_cached() -> None:
    cache = FaceAnalysisCache(True)
    counts, indices, points = _triangle_arrays([0, 1, 9])

    for _ in range(2):
        issues, _ = analyze_face_geometry(counts, indices, points, 3, face_cache=cache)

    assert issues["out_of_range_face_vertex_indices"] == 1
    assert cache.stats() == {"mode": "face-hash", "entries": 0, "hits": 0, "misses": 2}


def test_consistent_arrays_are_cached_even_when_no_deep_check_runs() -> None:
    """Fast mode switches the deep checks off, and still stores the topology result."""
    cache = FaceAnalysisCache(True)
    counts, indices, points = _triangle_arrays([0, 1, 2])

    for _ in range(2):
        analyze_face_geometry(
            counts, indices, points, 3, check_repeated_vertices=False, check_zero_area=False, face_cache=cache
        )

    assert cache.stats() == {"mode": "face-hash", "entries": 1, "hits": 1, "misses": 1}


# ----------------------------------------------------------- detail ordering


def test_normals_findings_are_recorded_before_primvar_findings(tmp_path: Path) -> None:
    """Details keep normals before primvars; archived report diffs depend on the order."""
    path = tmp_path / "mesh.usda"
    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (0, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([(0, 0, 1)]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([(0, 0)]))
    stage.GetRootLayer().Save()

    details = geometry.analyze(path)["worst_meshes"][0]["details"]

    assert list(details) == ["normals_length_mismatch", "primvar_length_mismatch"]
