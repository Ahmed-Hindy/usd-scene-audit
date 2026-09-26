"""Regression tests for surfacing checks that failed to run.

Two checks used to catch bare ``Exception`` and return a value indistinguishable
from "nothing wrong here". A crashed check and a passed check produced the same
output, so a report reading "0 issues" could not be trusted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import usd_scene_audit
from usd_scene_audit import geometry, names_hierarchy, scene  # noqa: F401 - resolved via getattr
from usd_scene_audit.geometry import (
    CheckErrorLog,
    analyze,
    mesh_record,
    transform_determinant,
)


class Exploding:
    """Stand-in for an XformCache whose transform computation fails."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def GetLocalToWorldTransform(self, prim):
        """Named to mirror UsdGeom.XformCache, hence the non-PEP8 casing."""
        raise self.error


def _triangle(stage: Usd.Stage, path: str = "/World/Mesh") -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    return mesh


# ------------------------------------------------------------------ the log


def test_error_log_starts_empty() -> None:
    """A clean run must report no failed checks."""
    log = CheckErrorLog()

    assert log.as_report() == {"count": 0, "examples": []}


@pytest.mark.expects_check_errors
def test_error_log_records_type_and_message() -> None:
    """Recorded entries must identify the check, the subject, and the error."""
    log = CheckErrorLog()

    log.record("some_check", "/World/Thing", ValueError("bad value"))

    report = log.as_report()
    assert report["count"] == 1
    assert report["examples"] == [{"check": "some_check", "subject": "/World/Thing", "error": "ValueError: bad value"}]


@pytest.mark.expects_check_errors
def test_error_log_counts_beyond_its_example_limit() -> None:
    """Examples are bounded, but the count must stay exact."""
    log = CheckErrorLog(limit=3)

    for index in range(10):
        log.record("some_check", f"/World/Thing{index}", RuntimeError("boom"))

    report = log.as_report()
    assert report["count"] == 10
    assert len(report["examples"]) == 3


# ------------------------------------------------------- transform_determinant


@pytest.mark.expects_check_errors
def test_failed_transform_is_recorded_not_swallowed() -> None:
    """A transform that cannot be computed must leave a trace."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle(stage)
    log = CheckErrorLog()

    determinant = transform_determinant(mesh.GetPrim(), Exploding(RuntimeError("no xform")), log)

    assert determinant is None
    assert log.count == 1
    assert log.entries[0]["check"] == "transform_determinant"
    assert log.entries[0]["subject"] == "/World/Mesh"
    assert "no xform" in log.entries[0]["error"]


def test_failed_transform_without_a_log_still_degrades_gracefully() -> None:
    """The error log is optional; omitting it must not raise."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle(stage)

    assert transform_determinant(mesh.GetPrim(), Exploding(RuntimeError("no xform"))) is None


def test_successful_transform_records_nothing() -> None:
    """A working check must not add noise to the error log."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle(stage)
    log = CheckErrorLog()

    determinant = transform_determinant(mesh.GetPrim(), UsdGeom.XformCache(Usd.TimeCode.EarliestTime()), log)

    assert determinant == 1.0
    assert log.count == 0


@pytest.mark.expects_check_errors
def test_mesh_record_propagates_the_error_log() -> None:
    """A failure deep in a mesh record must reach the caller's log."""
    stage = Usd.Stage.CreateInMemory()
    mesh = _triangle(stage)
    log = CheckErrorLog()

    record = mesh_record(mesh.GetPrim(), xform_cache=Exploding(RuntimeError("no xform")), error_log=log)

    assert record["transform_determinant"] is None
    assert log.count == 1


# ----------------------------------------------------------------- reports


def test_geometry_report_exposes_check_errors(stage_path) -> None:
    """The geometry report must always carry a check_errors block."""
    report = analyze(stage_path("static_mesh_clean.usda"))

    assert report["check_errors"] == {"count": 0, "examples": []}


def test_scene_report_exposes_check_errors(stage_path) -> None:
    """The scene report must always carry a check_errors block."""
    report = scene.analyze(stage_path("static_mesh_clean.usda"))

    assert report["check_errors"] == {"count": 0, "examples": []}


@pytest.mark.expects_check_errors
def test_scene_records_unreadable_layer_fields(monkeypatch, stage_path) -> None:
    """An unreadable Sdf field could hide an asset reference, so it must be logged."""

    def exploding_get_info(self, field):
        raise RuntimeError(f"cannot read {field}")

    monkeypatch.setattr(Sdf.Spec, "GetInfo", exploding_get_info, raising=True)

    report = scene.analyze(stage_path("assets_missing_texture.usda"))

    assert report["check_errors"]["count"] > 0
    checks = {entry["check"] for entry in report["check_errors"]["examples"]}
    assert checks == {"authored_asset_paths"}
    assert "cannot read" in report["check_errors"]["examples"][0]["error"]


def test_geometry_summary_mentions_failed_checks(capsys, stage_path) -> None:
    """A non-zero failure count must be visible without opening the JSON."""
    from usd_scene_audit.geometry import print_summary

    report = analyze(stage_path("static_mesh_clean.usda"))
    report["check_errors"] = {"count": 3, "examples": []}
    print_summary(report)

    assert "Checks that failed to run: 3" in capsys.readouterr().out


def test_geometry_summary_stays_quiet_when_all_checks_ran(capsys, stage_path) -> None:
    """A clean run must not print a zero-failure line."""
    from usd_scene_audit.geometry import print_summary

    print_summary(analyze(stage_path("static_mesh_clean.usda")))

    assert "Checks that failed to run" not in capsys.readouterr().out


BROAD_EXCEPTION_NAMES = {"Exception", "BaseException"}


def _catches_broadly(caught: ast.expr | None) -> bool:
    """Return true for `except:`, `except Exception`, or a tuple containing one."""
    if caught is None:
        return True
    if isinstance(caught, ast.Name):
        return caught.id in BROAD_EXCEPTION_NAMES
    if isinstance(caught, ast.Tuple):
        return any(_catches_broadly(element) for element in caught.elts)
    return False


def _broad_except_handlers(tree: ast.AST) -> list[ast.ExceptHandler]:
    """Return handlers that catch broadly: bare except, Exception, BaseException."""
    return [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and _catches_broadly(node.type)]


def _records_a_failure(handler: ast.ExceptHandler) -> bool:
    """Return true when the handler body calls error_log.record."""
    return any(
        isinstance(node, ast.Attribute) and node.attr == "record"
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
    )


@pytest.mark.parametrize("module_name", ["geometry", "scene", "names_hierarchy"])
def test_no_silent_excepts_remain(module_name: str) -> None:
    """Guard against reintroducing a swallowed exception.

    Every broad exception handler in these modules must record the failure. The
    check parses the AST rather than matching source text, so ``except:``,
    ``except BaseException``, and tuple handlers containing a broad type are all
    caught, and a guarded handler cannot vouch for an unguarded neighbour.

    This inspects source rather than behaviour on purpose: the regression being
    guarded against is invisible at runtime, which is the entire bug.
    """
    source = Path(usd_scene_audit.__file__).parent / f"{module_name}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    unguarded = [handler.lineno for handler in _broad_except_handlers(tree) if not _records_a_failure(handler)]

    assert not unguarded, f"{module_name}.py: broad except without a recorded failure at line(s) {unguarded}"


def test_guard_detects_an_unguarded_handler() -> None:
    """The guard must actually fail on a swallowed exception.

    Without this, the guard above could pass vacuously if its detection stopped
    working.
    """
    swallowed = ast.parse("try:\n    x()\nexcept Exception:\n    pass\n")
    recorded = ast.parse("try:\n    x()\nexcept Exception as e:\n    error_log.record('c', 's', e)\n")

    assert [h.lineno for h in _broad_except_handlers(swallowed) if not _records_a_failure(h)] == [3]
    assert [h.lineno for h in _broad_except_handlers(recorded) if not _records_a_failure(h)] == []


@pytest.mark.parametrize(
    "source",
    [
        "try:\n    x()\nexcept:\n    pass\n",
        "try:\n    x()\nexcept BaseException:\n    pass\n",
        "try:\n    x()\nexcept (RuntimeError, Exception):\n    pass\n",
    ],
)
def test_guard_recognises_all_broad_handler_forms(source: str) -> None:
    """Bare except, BaseException, and tuple handlers must all be detected."""
    assert _broad_except_handlers(ast.parse(source))


@pytest.mark.parametrize("module_name", ["geometry", "scene", "names_hierarchy"])
def test_every_report_exposes_check_errors(module_name: str, stage_path) -> None:
    """All three audits must expose the same check_errors contract."""
    module = getattr(usd_scene_audit, module_name)
    path = stage_path("static_mesh_clean.usda")
    report = module.analyze(path)

    assert report["check_errors"] == {"count": 0, "examples": []}
