"""Audit mesh geometry in a composed USD stage, including prototypes."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import inspect
import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

MAX_EXAMPLES = 80
FACE_CHUNK_SIZE = 100_000
FACE_ANALYSIS_ENGINES = ("auto", "numpy", "numba")
AUDIT_MODES = ("fast", "standard", "exhaustive")
MESH_CACHE_MODES = ("off", "face-hash")

# Defaults shared by analyze() and the CLI, so a library call with no options
# audits exactly like the command with no flags.
DEFAULT_ZERO_AREA_EPSILON = 1e-12
DEFAULT_HUGE_COORD_THRESHOLD = 1e6
DEFAULT_EXTENT_TOLERANCE = 1e-4
DEFAULT_GEOMETRY_ENGINE = "auto"
DEFAULT_AUDIT_MODE = "exhaustive"
DEFAULT_MESH_CACHE_MODE = "off"
_NUMBA_FACE_KERNEL = None
_NUMBA_IMPORT_ERROR: Exception | None = None


def accepts_legacy_positional(*legacy_names: str, fold=None):
    """Keep accepting options positionally for one release, with a DeprecationWarning.

    The decorated function takes its options keyword-only. Positional values
    beyond its own positional parameters are mapped, in order, onto
    ``legacy_names`` -- the old positional signature. ``fold``, when given,
    rewrites the resulting keyword arguments into the new signature (for
    example, grouping loose thresholds into a settings object) and is also
    applied when a caller passes those legacy names by keyword. It is called as
    ``fold(kwargs, warned)`` so that one legacy call emits one warning.
    """

    def decorate(function):
        positional = [
            parameter.name
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            legacy_values = args[len(positional) :]
            warned = bool(legacy_values)
            if legacy_values:
                if len(legacy_values) > len(legacy_names):
                    raise TypeError(
                        f"{function.__name__}() takes at most {len(positional) + len(legacy_names)} "
                        f"positional arguments ({len(args)} given)"
                    )
                names = legacy_names[: len(legacy_values)]
                warnings.warn(
                    f"passing {', '.join(names)} to {function.__name__}() positionally is deprecated "
                    "and will stop working in the next major release; pass them by keyword",
                    DeprecationWarning,
                    stacklevel=2,
                )
                for name, value in zip(names, legacy_values):
                    if name in kwargs:
                        raise TypeError(f"{function.__name__}() got multiple values for argument '{name}'")
                    kwargs[name] = value
                args = args[: len(positional)]
            if fold is not None:
                kwargs = fold(kwargs, warned)
            return function(*args, **kwargs)

        return wrapper

    return decorate


class PhaseTimer:
    """Collect wall-clock timings for broad audit phases."""

    def __init__(self) -> None:
        self.timings: defaultdict[str, float] = defaultdict(float)

    @contextmanager
    def phase(self, name: str):
        """Accumulate elapsed seconds under a named phase."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] += time.perf_counter() - started

    def rounded(self) -> dict[str, float]:
        """Return timings rounded for stable JSON output."""
        return {key: round(value, 3) for key, value in sorted(self.timings.items())}


class CheckErrorLog:
    """Record checks that failed to run.

    A check that raised and a check that passed used to produce identical
    output: nothing. For an audit tool that is the worst failure mode, because
    "0 issues" cannot be distinguished from "the analysis partially failed".

    Failures are recorded rather than narrowed away on purpose. One unusual prim
    must not abort a multi-minute audit of a large stage, so the broad catch
    stays -- what changes is that it is no longer silent. The recorded exception
    types are what will tell us which excepts can safely be narrowed later.
    """

    def __init__(self, limit: int = 40) -> None:
        self.entries: list[dict[str, str]] = []
        self.count = 0
        self.limit = limit

    def record(self, check: str, subject: str, error: BaseException) -> None:
        """Record one failed check, keeping the example list bounded."""
        self.count += 1
        if len(self.entries) < self.limit:
            self.entries.append(
                {
                    "check": check,
                    "subject": subject,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    def as_report(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of failed checks."""
        return {"count": self.count, "examples": self.entries}


class FaceAnalysisCache:
    """Cache exact face-analysis results for duplicate mesh arrays."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.entries: dict[tuple[Any, ...], tuple[Counter[str], dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...] | None) -> tuple[Counter[str], dict[str, Any]] | None:
        """Return a defensive copy of cached face results."""
        if not self.enabled or key is None:
            return None
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return Counter(value[0]), copy.deepcopy(value[1])

    def set(self, key: tuple[Any, ...] | None, issues: Counter[str], details: dict[str, Any]) -> None:
        """Store a defensive copy of face results."""
        if self.enabled and key is not None:
            self.entries[key] = (Counter(issues), copy.deepcopy(details))

    def stats(self) -> dict[str, int | str]:
        """Return cache statistics for reports."""
        return {
            "mode": "face-hash" if self.enabled else "off",
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
        }


def add_example(examples: dict[str, list[Any]], key: str, value: Any, limit: int = MAX_EXAMPLES) -> None:
    """Append a bounded example for an issue type."""
    examples.setdefault(key, [])
    if len(examples[key]) < limit:
        examples[key].append(value)


def normalized_path(path: str) -> str:
    """Collapse generated prototype numbers for easier cross-run comparison."""
    return re.sub(r"^/__Prototype_\d+", "/<prototype>", path)


def resolve_time_code(frame: float | None, stage: Usd.Stage | None = None) -> Usd.TimeCode:
    """Return the time code used to read geometry attributes.

    Reading at ``Usd.TimeCode.Default()`` resolves only an attribute's default
    value. Deforming geometry normally authors ``points`` purely as time samples
    with no default, so a default-time read returns ``None`` and the mesh looks
    like it is missing its points entirely.

    Preference order:

    1. An explicitly requested ``frame``.
    2. The stage's authored ``startTimeCode``, when it has one.
    3. ``EarliestTime()``.

    Options 1 and 2 are a single concrete moment, which matters because several
    checks compare two attributes against each other. ``EarliestTime()`` resolves
    *each attribute* at its own first authored sample, so a mesh whose extent is
    authored only over the shot range while its points also carry pre-roll
    samples would be compared across two different moments and report a
    mismatch that does not exist at any real frame. Preferring the authored
    ``startTimeCode`` keeps every attribute on the same frame and makes the
    default audit describe the shot rather than whatever pre-roll sample happens
    to sort earliest.

    ``EarliestTime()`` remains the fallback for stages with no authored time
    range: it resolves to the first authored sample when one exists and falls
    back to the default value otherwise, so it is safe for static geometry too.
    """
    if frame is not None:
        return Usd.TimeCode(float(frame))
    if stage is not None and stage.HasAuthoredTimeCodeRange():
        return Usd.TimeCode(stage.GetStartTimeCode())
    return Usd.TimeCode.EarliestTime()


def default_time_code(prim: Usd.Prim) -> Usd.TimeCode:
    """Return the time code a helper uses when the caller does not pass one.

    Matches the time code ``analyze()`` evaluates with no ``--frame``, so a
    helper called directly reads attributes at the same moment the full audit
    does. Falling back to ``EarliestTime()`` here instead would reintroduce the
    pre-roll false positives that ``resolve_time_code`` exists to prevent.

    This only covers attribute reads. ``mesh_record`` also takes an
    ``XformCache``, which evaluates transforms at whatever time it was built
    with; see its docstring.
    """
    return resolve_time_code(None, prim.GetStage())


def describe_time_code(time_code: Usd.TimeCode) -> str | float:
    """Return a JSON-friendly description of an evaluated time code."""
    if time_code.IsEarliestTime():
        return "earliest"
    if time_code.IsDefault():
        return "default"
    return float(time_code.GetValue())


PROTOTYPE_ROOT_PATTERN = re.compile(r"^/__Prototype_\d+")


class PrototypePaths:
    """Stable names and a stable order for a stage's instancing prototypes.

    OpenUSD numbers prototypes ``/__Prototype_1``, ``/__Prototype_2``, ... in an
    order that changes between opens of the same stage, even in one process
    with one hash seed. A report that iterates ``GetPrototypes()`` or prints a
    raw prototype path therefore differs from run to run.

    Each prototype is named after its first instance in sorted path order, with
    any enclosing prototype resolved the same way, so ``/__Prototype_7/Body``
    is reported as ``/World/Asset_0/Body`` -- the instance-proxy path of the
    first instance that shares it. Prototypes are visited in that order.

    Only report output uses these names. Lookups such as material binding
    resolution keep the real prototype path, because an instance proxy can
    resolve differently.
    """

    def __init__(self, stage: Usd.Stage) -> None:
        self._prototypes = {str(prototype.GetPath()): prototype for prototype in stage.GetPrototypes()}
        self._labels: dict[str, str] = {}

    def label(self, prototype_root: str) -> str:
        """Return the stable name for a prototype root path."""
        if prototype_root not in self._labels:
            instances = self._prototypes[prototype_root].GetInstances()
            # Nesting is acyclic, so resolving enclosing prototypes terminates.
            self._labels[prototype_root] = min(
                (self.stable(str(instance.GetPath())) for instance in instances), default=prototype_root
            )
        return self._labels[prototype_root]

    def stable(self, path: str) -> str:
        """Return ``path`` with a leading prototype root replaced by its stable name."""
        match = PROTOTYPE_ROOT_PATTERN.match(path)
        if match is None or match.group(0) not in self._prototypes:
            return path
        return self.label(match.group(0)) + path[match.end() :]

    def ordered(self) -> list[Usd.Prim]:
        """Return the prototypes sorted by stable name."""
        return [self._prototypes[root] for root in sorted(self._prototypes, key=self.label)]

    def __len__(self) -> int:
        return len(self._prototypes)


def prims_with_prototypes(stage: Usd.Stage) -> tuple[list[Usd.Prim], int]:
    """Return ordinary stage traversal plus prototype contents, prototypes in stable order."""
    prims = list(stage.Traverse())
    prototypes = PrototypePaths(stage).ordered()
    for prototype in prototypes:
        prims.extend(Usd.PrimRange(prototype))
    return prims, len(prototypes)


def classify_mesh(path: str, name: str) -> str:
    """Classify mesh names without excluding anything from analysis."""
    collision_patterns = (
        "_COL" in name
        or name.endswith("_C")
        or name.endswith("_CO")
        or bool(re.search(r"_C_\d+$", name))
        or bool(re.search(r"_C\d+$", name))
        or "collision" in name.lower()
    )
    if collision_patterns:
        return "collision_like"
    if any(token in path.lower() for token in ("/proxy", "/helper", "/helpers", "/guide", "/guides")):
        return "unknown_helper_like"
    return "render_like"


def vec3_tuple(value) -> tuple[float, float, float]:
    """Convert a Gf vec-like object to a Python tuple."""
    return (float(value[0]), float(value[1]), float(value[2]))


def bbox_from_points(points) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute min and max bounds from finite points."""
    points_np = np.asarray(points)
    if points_np.size == 0:
        return None
    finite_mask = np.isfinite(points_np).all(axis=1)
    if not np.any(finite_mask):
        return None
    finite_points = points_np[finite_mask]
    return (
        tuple(float(v) for v in np.min(finite_points, axis=0)),
        tuple(float(v) for v in np.max(finite_points, axis=0)),
    )


def bbox_diagonal(bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None) -> float:
    """Return the diagonal length of a bounds tuple."""
    if bounds is None:
        return 0.0
    lo, hi = bounds
    return math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))


def triangle_area(a, b, c) -> float:
    """Return triangle area for three point-like values."""
    ab = Gf.Vec3d(b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = Gf.Vec3d(c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return 0.5 * Gf.Cross(ab, ac).GetLength()


def authored_extent_bounds(mesh: UsdGeom.Mesh, time_code: Usd.TimeCode | None = None):
    """Return authored extent as a bounds tuple, or None.

    ``time_code`` defaults to ``default_time_code(mesh.GetPrim())``.
    """
    if time_code is None:
        time_code = default_time_code(mesh.GetPrim())
    bounds, _problem = read_extent(mesh, time_code)
    return bounds


def read_extent(mesh: UsdGeom.Mesh, time_code: Usd.TimeCode):
    """Return ``(bounds, problem)`` for the authored extent.

    ``bounds`` is None when extent is absent, empty, or unusable. ``problem``
    describes an extent that is authored but is not two finite-typed rows of
    three values -- a ``float[]`` or ``string[]`` extent used to be skipped or
    to raise.
    """
    attr = mesh.GetExtentAttr()
    extent = attr.Get(time_code)
    extent_np, problem = shaped_array(extent, 3)
    if problem is None and extent_np is not None and extent_np.size and len(extent_np) != 2:
        problem = {"reason": "extent must have exactly two rows", "shape": [int(v) for v in extent_np.shape]}
    if problem is not None:
        return None, {**attribute_types(attr, extent), **problem}
    if extent_np is None or not extent_np.size:
        return None, None
    return (vec3_tuple(extent_np[0]), vec3_tuple(extent_np[1])), None


def bounds_mismatch(
    authored: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    computed: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    tolerance: float,
) -> float | None:
    """Return max component bounds delta when it exceeds tolerance."""
    if authored is None or computed is None:
        return None
    max_delta = 0.0
    for corner in range(2):
        for axis in range(3):
            max_delta = max(max_delta, abs(authored[corner][axis] - computed[corner][axis]))
    return max_delta if max_delta > tolerance else None


def expected_primvar_length(
    interpolation: str, point_count: int | None, face_count: int | None, face_vertex_count: int | None
) -> int | None:
    """Return expected primvar element count for a USD interpolation.

    Returns None when the interpolation is unknown, or when the count it depends
    on is None -- unknown because the topology attribute it comes from is
    missing or unusable. That defect is reported once, as ``missing_*`` or
    ``*_wrong_type``; comparing against a stand-in count of zero would repeat
    it as a length mismatch on every primvar.
    """
    if interpolation == "constant":
        return 1
    if interpolation == "uniform":
        return face_count
    if interpolation in {"vertex", "varying"}:
        return point_count
    if interpolation == "faceVarying":
        return face_vertex_count
    return None


def attr_length(value) -> int:
    """Return length for array-like values, or zero for absent scalars."""
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 1


def issue_details(details: dict[str, Any], key: str, value: Any, limit: int = 10) -> None:
    """Append bounded issue detail on a mesh record."""
    details.setdefault(key, [])
    if len(details[key]) < limit:
        details[key].append(value)


def value_type_name(value) -> str:
    """Return the Sdf type name of a resolved value, e.g. ``float[]`` for a Vt.FloatArray."""
    type_name = str(Sdf.GetValueTypeNameForValue(value))
    return type_name or type(value).__name__


def attribute_types(attr: Usd.Attribute, value) -> dict[str, str]:
    """Return the type the schema expects and the type of the value that resolved.

    ``GetTypeName()`` reports the schema type, so a ``float[] normals`` opinion
    still reads as ``normal3f[]`` there. The authored type is taken from the
    resolved value itself rather than from a property spec: the strongest spec
    may carry only metadata (an interpolation override), and value clips never
    appear in the property stack at all. Roles do not survive into the value,
    so a scalar ``point3f`` reads back as ``float3``.
    """
    return {"authored_type": value_type_name(value), "expected_type": str(attr.GetTypeName())}


def shaped_array(value, row_width: int | None, *, integer: bool = False) -> tuple[Any, dict[str, Any] | None]:
    """Convert an authored value to NumPy, rejecting values the checks cannot use.

    ``row_width=None`` expects a flat array (face-vertex counts and indices); an
    integer expects ``N x row_width`` rows (points, normals, extent).
    ``integer=True`` additionally requires integer elements and returns int64.

    An attribute authored with the wrong value type resolves to a value of that
    type: ``float[] normals`` is flat, ``int faceVertexCounts`` is a scalar,
    ``float[] faceVertexIndices`` holds floats. Feeding those to the vectorized
    checks either raised or -- for float or bool indices cast to int64 --
    silently truncated them into plausible-looking topology.

    Returns ``(array, None)`` for a usable value, ``(None, problem)`` for an
    unusable one, and ``(None, None)`` when the value is absent. Empty arrays of
    any type are usable; they are reported as empty elsewhere.
    """
    if value is None:
        return None, None
    value_type = Sdf.GetValueTypeNameForValue(value)
    if value_type and not value_type.isArray:
        return None, {"reason": "authored as a scalar, not an array"}
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        return None, {"reason": f"not convertible to an array ({type(error).__name__}: {error})"}
    shape = [int(v) for v in raw.shape]
    if raw.size == 0:
        return (raw.astype(np.int64) if integer else raw).reshape(raw.shape), None
    allowed_kinds = "iu" if integer else "fiu"
    if raw.dtype.kind not in allowed_kinds:
        expected = "integer" if integer else "numeric"
        return None, {"reason": f"{raw.dtype} elements are not {expected}", "shape": shape}
    expected_ndim = 1 if row_width is None else 2
    if raw.ndim != expected_ndim or (row_width is not None and raw.shape[1] != row_width):
        expected = "a flat array" if row_width is None else f"rows of {row_width} values"
        return None, {"reason": f"shape is not {expected}", "shape": shape}
    if integer:
        if raw.dtype.kind == "u" and raw.max() > np.iinfo(np.int64).max:
            return None, {"reason": "values exceed the int64 range", "shape": shape}
        return raw.astype(np.int64, copy=False), None
    if raw.dtype.kind == "f" and raw.dtype.itemsize < 4:
        # half3[] points: float16 cannot represent the default 1e6 huge-coordinate
        # threshold, so comparisons against it would overflow.
        return raw.astype(np.float32), None
    return raw, None


def array_digest(array) -> tuple[str, tuple[int, ...], str] | None:
    """Return a stable digest for a NumPy array without changing audit results."""
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous.view(np.uint8)))
    return digest.hexdigest(), tuple(int(v) for v in contiguous.shape), str(contiguous.dtype)


def face_cache_key(
    counts_np,
    indices_np,
    points_np,
    point_count: int,
    zero_area_epsilon: float,
    face_analysis_engine: str,
    check_repeated_vertices: bool,
    check_zero_area: bool,
) -> tuple[Any, ...] | None:
    """Build a cache key for checks that depend only on face arrays and points."""
    if counts_np is None or indices_np is None:
        return None
    points_key = array_digest(points_np) if check_zero_area else None
    return (
        array_digest(counts_np),
        array_digest(indices_np),
        points_key,
        point_count,
        zero_area_epsilon,
        face_analysis_engine,
        check_repeated_vertices,
        check_zero_area,
    )


def face_checks_for_mode(audit_mode: str, category: str) -> tuple[bool, bool]:
    """Return repeated-vertex and zero-area check switches for an audit mode."""
    if audit_mode not in AUDIT_MODES:
        raise ValueError(f"Unsupported audit mode: {audit_mode}")
    if audit_mode == "fast":
        return False, False
    if audit_mode == "standard":
        return True, category != "collision_like"
    return True, True


def _face_geometry_numba_kernel_impl(
    counts,
    indices,
    points,
    zero_area_epsilon,
    check_repeated_vertices,
    check_zero_area,
):
    """Count repeated-vertex faces and zero-area fan triangles in one streaming pass."""
    repeated_total = 0
    zero_area_total = 0
    repeated_example_faces = np.full(10, -1, dtype=np.int64)
    zero_example_indices = np.full((10, 3), -1, dtype=np.int64)
    zero_example_areas = np.zeros(10, dtype=np.float64)
    repeated_example_count = 0
    zero_example_count = 0
    threshold = 2.0 * zero_area_epsilon
    cursor = 0

    for face_index in range(counts.size):
        width = counts[face_index]
        repeated = False
        if check_repeated_vertices:
            for left in range(width):
                left_index = indices[cursor + left]
                for right in range(left + 1, width):
                    if left_index == indices[cursor + right]:
                        repeated = True
                        break
                if repeated:
                    break

        if repeated:
            repeated_total += 1
            if repeated_example_count < repeated_example_faces.size:
                repeated_example_faces[repeated_example_count] = face_index
                repeated_example_count += 1

        if check_zero_area and width >= 3 and points.size:
            anchor_index = indices[cursor]
            ax = points[anchor_index, 0]
            ay = points[anchor_index, 1]
            az = points[anchor_index, 2]
            for column in range(1, width - 1):
                b_index = indices[cursor + column]
                c_index = indices[cursor + column + 1]

                abx = points[b_index, 0] - ax
                aby = points[b_index, 1] - ay
                abz = points[b_index, 2] - az
                acx = points[c_index, 0] - ax
                acy = points[c_index, 1] - ay
                acz = points[c_index, 2] - az

                cross_x = aby * acz - abz * acy
                cross_y = abz * acx - abx * acz
                cross_z = abx * acy - aby * acx
                norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
                if norm <= threshold:
                    zero_area_total += 1
                    if zero_example_count < zero_example_areas.size:
                        zero_example_indices[zero_example_count, 0] = anchor_index
                        zero_example_indices[zero_example_count, 1] = b_index
                        zero_example_indices[zero_example_count, 2] = c_index
                        zero_example_areas[zero_example_count] = norm * 0.5
                        zero_example_count += 1

        cursor += width

    return (
        repeated_total,
        repeated_example_faces,
        repeated_example_count,
        zero_area_total,
        zero_example_indices,
        zero_example_areas,
        zero_example_count,
    )


def get_numba_face_kernel():
    """Return the optional Numba face-analysis kernel, or None when unavailable."""
    global _NUMBA_FACE_KERNEL, _NUMBA_IMPORT_ERROR
    if _NUMBA_FACE_KERNEL is not None:
        return _NUMBA_FACE_KERNEL
    if _NUMBA_IMPORT_ERROR is not None:
        return None
    try:
        from numba import njit
    except ImportError as exc:
        _NUMBA_IMPORT_ERROR = exc
        return None

    _NUMBA_FACE_KERNEL = njit(cache=True)(_face_geometry_numba_kernel_impl)
    return _NUMBA_FACE_KERNEL


def resolve_face_analysis_engine(requested_engine: str) -> str:
    """Resolve auto/numpy/numba into the engine used for face geometry checks."""
    if requested_engine not in FACE_ANALYSIS_ENGINES:
        raise ValueError(f"Unsupported geometry engine: {requested_engine}")
    if requested_engine == "numpy":
        return "numpy"
    kernel = get_numba_face_kernel()
    if kernel is not None:
        return "numba"
    if requested_engine == "auto":
        return "numpy"
    raise RuntimeError("Numba acceleration is not installed. Install with: uv sync --extra numba")


def count_repeated_rows(index_rows) -> tuple[int, list[list[int]]]:
    """Count rows that contain repeated vertex indices."""
    if index_rows.size == 0:
        return 0, []
    repeated = np.zeros(index_rows.shape[0], dtype=bool)
    width = index_rows.shape[1]
    for left in range(width):
        for right in range(left + 1, width):
            repeated |= index_rows[:, left] == index_rows[:, right]
    repeated_positions = np.flatnonzero(repeated)
    examples = [index_rows[i].astype(int).tolist() for i in repeated_positions[:10]]
    return int(repeated_positions.size), examples


def count_zero_area_fan_triangles(points_np, index_rows, epsilon: float) -> tuple[int, list[dict[str, Any]]]:
    """Count zero-area fan triangles for same-width face index rows."""
    if points_np is None or index_rows.size == 0 or index_rows.shape[1] < 3:
        return 0, []
    total = 0
    examples: list[dict[str, Any]] = []
    threshold = 2.0 * epsilon
    anchor = points_np[index_rows[:, 0]].astype("float64", copy=False)
    for column in range(1, index_rows.shape[1] - 1):
        b = points_np[index_rows[:, column]].astype("float64", copy=False)
        c = points_np[index_rows[:, column + 1]].astype("float64", copy=False)
        cross = np.cross(b - anchor, c - anchor)
        norms = np.linalg.norm(cross, axis=1)
        bad_positions = np.flatnonzero(norms <= threshold)
        total += int(bad_positions.size)
        for pos in bad_positions[: max(0, 10 - len(examples))]:
            examples.append(
                {
                    "triangle_indices": [
                        int(index_rows[pos, 0]),
                        int(index_rows[pos, column]),
                        int(index_rows[pos, column + 1]),
                    ],
                    "area": float(norms[pos] * 0.5),
                }
            )
        if len(examples) >= 10 and total:
            # Continue counting exactly, but stop collecting examples.
            continue
    return total, examples


def count_face_geometry_numba(
    counts_np,
    indices_np,
    points_np,
    offsets,
    zero_area_epsilon: float,
    check_repeated_vertices: bool,
    check_zero_area: bool,
) -> tuple[int, list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Count expensive face-geometry checks with the optional Numba kernel."""
    kernel = get_numba_face_kernel()
    if kernel is None:
        # Returning zero counts here would read as "no repeated vertices, no
        # zero-area triangles" -- a clean result for checks that never ran.
        raise RuntimeError("Numba acceleration is not installed. Install with: uv sync --extra numba")
    points_for_kernel = points_np if points_np is not None else np.empty((0, 3), dtype=np.float64)
    (
        repeated_total,
        repeated_example_faces,
        repeated_example_count,
        zero_area_total,
        zero_example_indices,
        zero_example_areas,
        zero_example_count,
    ) = kernel(
        counts_np,
        indices_np,
        points_for_kernel,
        zero_area_epsilon,
        check_repeated_vertices,
        check_zero_area,
    )

    repeated_examples: list[dict[str, Any]] = []
    for face_index in repeated_example_faces[:repeated_example_count]:
        start = offsets[int(face_index)]
        width = counts_np[int(face_index)]
        repeated_examples.append({"indices": indices_np[start : start + width].astype(int).tolist()})

    zero_area_examples = [
        {
            "triangle_indices": zero_example_indices[i].astype(int).tolist(),
            "area": float(zero_example_areas[i]),
        }
        for i in range(int(zero_example_count))
    ]
    return int(repeated_total), repeated_examples, int(zero_area_total), zero_area_examples


def count_face_geometry_numpy(
    counts_np,
    indices_np,
    points_np,
    offsets,
    zero_area_epsilon: float,
    check_repeated_vertices: bool,
    check_zero_area: bool,
) -> tuple[int, list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Count expensive face-geometry checks with NumPy batching."""
    repeated_total = 0
    zero_area_total = 0
    repeated_examples: list[dict[str, Any]] = []
    zero_area_examples: list[dict[str, Any]] = []

    for width in sorted(int(v) for v in np.unique(counts_np) if int(v) > 0):
        face_positions = np.flatnonzero(counts_np == width)
        if face_positions.size == 0:
            continue
        for start in range(0, face_positions.size, FACE_CHUNK_SIZE):
            chunk_faces = face_positions[start : start + FACE_CHUNK_SIZE]
            starts = offsets[chunk_faces]
            rows = indices_np[starts[:, None] + np.arange(width, dtype=np.int64)]

            if check_repeated_vertices:
                repeated_count, row_examples = count_repeated_rows(rows)
                repeated_total += repeated_count
                for row in row_examples:
                    if len(repeated_examples) < 10:
                        repeated_examples.append({"indices": row})

            if check_zero_area and width >= 3 and points_np is not None:
                zero_count, tri_examples = count_zero_area_fan_triangles(points_np, rows, zero_area_epsilon)
                zero_area_total += zero_count
                for example in tri_examples:
                    if len(zero_area_examples) < 10:
                        zero_area_examples.append(example)

    return repeated_total, repeated_examples, zero_area_total, zero_area_examples


def face_topology_issues(counts_np, indices_np, point_count: int | None) -> tuple[Counter[str], dict[str, Any]]:
    """Report cheap per-face and per-index defects: bad face widths and out-of-range indices.

    ``point_count=None`` means the points are missing or unusable. Negative
    indices are still reported, but "out of range" has no range to check
    against, so it is skipped rather than reported for every index.
    """
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    empty_or_negative = np.flatnonzero(counts_np <= 0)
    one_or_two = np.flatnonzero((counts_np == 1) | (counts_np == 2))
    if empty_or_negative.size:
        issues["empty_or_negative_faces"] += int(empty_or_negative.size)
        details["empty_or_negative_face_examples"] = empty_or_negative[:10].astype(int).tolist()
    if one_or_two.size:
        issues["one_or_two_vertex_faces"] += int(one_or_two.size)
        details["one_or_two_vertex_face_examples"] = one_or_two[:10].astype(int).tolist()

    if indices_np.size:
        negative_mask = indices_np < 0
        if np.any(negative_mask):
            bad = indices_np[negative_mask]
            issues["negative_face_vertex_indices"] += int(bad.size)
            details["negative_index_examples"] = bad[:10].astype(int).tolist()
        out_of_range_mask = indices_np >= point_count if point_count is not None else None
        if out_of_range_mask is not None and np.any(out_of_range_mask):
            bad = indices_np[out_of_range_mask]
            issues["out_of_range_face_vertex_indices"] += int(bad.size)
            details["out_of_range_index_examples"] = bad[:10].astype(int).tolist()
    return issues, details


def deep_face_checks_are_safe(counts_np, indices_np, point_count: int | None) -> bool:
    """Return true when face arrays are consistent enough to walk face by face.

    The per-face checks index ``points`` through ``faceVertexIndices`` using
    offsets derived from ``faceVertexCounts``; any inconsistency here would make
    them read out of bounds. The defects themselves are reported by
    ``face_topology_issues`` and the caller.
    """
    if int(counts_np.sum()) != indices_np.size:
        return False
    if not point_count or indices_np.size == 0 or np.any(counts_np < 0):
        return False
    return not (np.any(indices_np < 0) or np.any(indices_np >= point_count))


def face_offsets(counts_np):
    """Return the start index of each face in ``faceVertexIndices``."""
    offsets = np.empty(counts_np.size, dtype=np.int64)
    offsets[0] = 0
    if counts_np.size > 1:
        offsets[1:] = np.cumsum(counts_np[:-1], dtype=np.int64)
    return offsets


def deep_face_issues(
    counts_np,
    indices_np,
    points_np,
    *,
    zero_area_epsilon: float,
    face_analysis_engine: str,
    check_repeated_vertices: bool,
    check_zero_area: bool,
) -> tuple[Counter[str], dict[str, Any]]:
    """Run the exact per-face checks: repeated vertices and zero-area fan triangles."""
    count_face_geometry = count_face_geometry_numba if face_analysis_engine == "numba" else count_face_geometry_numpy
    repeated_total, repeated_examples, zero_area_total, zero_area_examples = count_face_geometry(
        counts_np,
        indices_np,
        points_np,
        face_offsets(counts_np),
        zero_area_epsilon,
        check_repeated_vertices,
        check_zero_area,
    )
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    if repeated_total:
        issues["faces_with_repeated_vertices"] += repeated_total
        details["repeated_vertex_face_examples"] = repeated_examples
    if zero_area_total:
        issues["zero_area_triangles"] += zero_area_total
        details["zero_area_triangle_examples"] = zero_area_examples
    return issues, details


@accepts_legacy_positional(
    "zero_area_epsilon", "face_analysis_engine", "check_repeated_vertices", "check_zero_area", "face_cache"
)
def analyze_face_geometry(
    counts_np,
    indices_np,
    points_np,
    point_count: int | None,
    *,
    zero_area_epsilon: float = DEFAULT_ZERO_AREA_EPSILON,
    face_analysis_engine: str = "numpy",
    check_repeated_vertices: bool = True,
    check_zero_area: bool = True,
    face_cache: FaceAnalysisCache | None = None,
) -> tuple[Counter[str], dict[str, Any]]:
    """Validate face arrays with exact checks.

    ``face_analysis_engine`` is a resolved engine, ``"numpy"`` or ``"numba"``;
    resolve ``"auto"`` with ``resolve_face_analysis_engine()`` first.
    ``point_count=None`` means the points are missing or unusable; see
    ``face_topology_issues``.
    ``face_cache=None`` disables caching.
    """
    if face_analysis_engine not in ("numpy", "numba"):
        raise ValueError(f"Unsupported face analysis engine: {face_analysis_engine}")
    # Building the key hashes every byte of the face arrays (and the points,
    # when the zero-area check is on), so it must only happen when the cache
    # can actually use it. analyze() always passes a cache object, enabled or not.
    use_cache = face_cache is not None and face_cache.enabled
    cache_key = (
        face_cache_key(
            counts_np,
            indices_np,
            points_np,
            point_count,
            zero_area_epsilon,
            face_analysis_engine,
            check_repeated_vertices,
            check_zero_area,
        )
        if use_cache
        else None
    )
    if use_cache:
        cached = face_cache.get(cache_key)
        if cached is not None:
            return cached

    if counts_np is None or indices_np is None or counts_np.size == 0:
        return Counter(), {}

    issues, details = face_topology_issues(counts_np, indices_np, point_count)
    # Only consistent arrays are cached; inconsistent ones return here after a
    # counted miss. Their topology issues are cheap to recompute.
    if not deep_face_checks_are_safe(counts_np, indices_np, point_count):
        return issues, details

    if check_repeated_vertices or check_zero_area:
        deep_issues, deep_details = deep_face_issues(
            counts_np,
            indices_np,
            points_np,
            zero_area_epsilon=zero_area_epsilon,
            face_analysis_engine=face_analysis_engine,
            check_repeated_vertices=check_repeated_vertices,
            check_zero_area=check_zero_area,
        )
        issues.update(deep_issues)
        details.update(deep_details)

    if use_cache:
        face_cache.set(cache_key, issues, details)
    return issues, details


def validate_primvars(
    prim: Usd.Prim,
    point_count: int | None,
    face_count: int | None,
    face_vertex_count: int | None,
    time_code: Usd.TimeCode | None = None,
) -> list[dict[str, Any]]:
    """Validate authored primvar lengths, with special attention to UV-like primvars.

    ``time_code`` defaults to ``default_time_code(prim)``.
    """
    if time_code is None:
        time_code = default_time_code(prim)
    issues: list[dict[str, Any]] = []
    primvars = UsdGeom.PrimvarsAPI(prim).GetPrimvars()
    for primvar in primvars:
        attr = primvar.GetAttr()
        indices_attr = primvar.GetIndicesAttr()
        if not attr.HasAuthoredValueOpinion() and not indices_attr.HasAuthoredValueOpinion():
            continue
        name = primvar.GetPrimvarName()
        interpolation = primvar.GetInterpolation() or ""
        expected = expected_primvar_length(interpolation, point_count, face_count, face_vertex_count)
        value = primvar.Get(time_code)
        value_len = attr_length(value)
        indices = primvar.GetIndices(time_code)
        indices_len = attr_length(indices)
        is_indexed = indices is not None and indices_len > 0
        element_size = max(1, int(primvar.GetElementSize() or 1))
        value_element_count = value_len // element_size if element_size else value_len
        if value_len and value_len % element_size:
            issues.append(
                {
                    "primvar": name,
                    "issue": "primvar_element_size_mismatch",
                    "value_count": value_len,
                    "element_size": element_size,
                }
            )

        if expected is not None:
            authored_len = indices_len if is_indexed else value_element_count
            if authored_len != expected:
                issues.append(
                    {
                        "primvar": name,
                        "issue": "primvar_length_mismatch",
                        "interpolation": interpolation,
                        "expected": expected,
                        "actual": authored_len,
                        "value_count": value_len,
                        "index_count": indices_len,
                        "element_size": element_size,
                    }
                )

        if is_indexed and value_len:
            indices_np = np.asarray(indices, dtype=np.int64)
            bad_positions = np.flatnonzero((indices_np < 0) | (indices_np >= value_element_count))
            bad_indices = indices_np[bad_positions[:10]].astype(int).tolist()
            if bad_indices:
                issues.append(
                    {
                        "primvar": name,
                        "issue": "primvar_index_out_of_range",
                        "value_count": value_element_count,
                        "bad_index_examples": bad_indices,
                    }
                )

        lower_name = name.lower()
        if lower_name in {"st", "uv", "uv0", "map1"} or "uv" in lower_name:
            if interpolation not in {"faceVarying", "vertex", "varying"}:
                issues.append(
                    {
                        "primvar": name,
                        "issue": "uv_unusual_interpolation",
                        "interpolation": interpolation,
                    }
                )
            if value_len == 0:
                issues.append({"primvar": name, "issue": "uv_empty"})

    return issues


def validate_normals(
    mesh: UsdGeom.Mesh,
    point_count: int | None,
    face_count: int | None,
    face_vertex_count: int | None,
    time_code: Usd.TimeCode | None = None,
) -> list[dict[str, Any]]:
    """Validate authored normals length and finite values.

    ``time_code`` defaults to ``default_time_code(mesh.GetPrim())``.
    """
    if time_code is None:
        time_code = default_time_code(mesh.GetPrim())
    issues: list[dict[str, Any]] = []
    normals = mesh.GetNormalsAttr().Get(time_code)
    if normals is None:
        return issues
    normals_np, problem = shaped_array(normals, 3)
    if problem is not None:
        return [{"issue": "normals_wrong_type", **attribute_types(mesh.GetNormalsAttr(), normals), **problem}]
    interpolation = mesh.GetNormalsInterpolation() or ""
    normal_count = len(normals_np)
    expected = expected_primvar_length(interpolation, point_count, face_count, face_vertex_count)
    if expected is not None and normal_count != expected:
        issues.append(
            {
                "issue": "normals_length_mismatch",
                "interpolation": interpolation,
                "expected": expected,
                "actual": normal_count,
            }
        )
    non_finite = (
        np.flatnonzero(~np.isfinite(normals_np).all(axis=1))[:10].astype(int).tolist() if normals_np.size else []
    )
    if non_finite:
        issues.append({"issue": "normals_non_finite", "index_examples": non_finite})
    return issues


def transform_determinant(
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
    error_log: CheckErrorLog | None = None,
    subject: str | None = None,
) -> float | None:
    """Return local-to-world transform determinant, if computable.

    A failure is recorded in ``error_log`` rather than silently returning None,
    which the caller would otherwise read as "this transform is fine".
    """
    try:
        transform = xform_cache.GetLocalToWorldTransform(prim)
        return float(transform.GetDeterminant())
    except Exception as error:  # noqa: BLE001 - recorded below; see CheckErrorLog
        if error_log is not None:
            error_log.record("transform_determinant", subject or str(prim.GetPath()), error)
        return None


@contextmanager
def recorded_failure(check: str, subject: str, error_log: CheckErrorLog | None):
    """Record an exception raised by one check phase, then carry on with the mesh.

    Without an ``error_log`` the exception propagates, so direct callers of
    ``mesh_record()`` still see it. With one, a failing phase costs only its own
    findings: everything the mesh's other phases found still reaches the report.
    """
    if error_log is None:
        yield
        return
    try:
        yield
    except Exception as error:  # noqa: BLE001 - recorded below; see CheckErrorLog
        error_log.record(check, subject, error)


@dataclass(frozen=True)
class MeshCheckSettings:
    """Thresholds and switches for per-mesh checks, validated on construction.

    Grouped so they travel by name: three adjacent float thresholds passed
    positionally can be swapped without any error.
    """

    zero_area_epsilon: float = DEFAULT_ZERO_AREA_EPSILON
    huge_coord_threshold: float = DEFAULT_HUGE_COORD_THRESHOLD
    extent_tolerance: float = DEFAULT_EXTENT_TOLERANCE
    face_analysis_engine: str = "numpy"
    audit_mode: str = DEFAULT_AUDIT_MODE

    def __post_init__(self) -> None:
        for name in ("zero_area_epsilon", "huge_coord_threshold", "extent_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
        if self.audit_mode not in AUDIT_MODES:
            raise ValueError(f"Unsupported audit mode: {self.audit_mode}")
        if self.face_analysis_engine not in ("numpy", "numba"):
            raise ValueError(
                f"Unsupported face analysis engine: {self.face_analysis_engine}; "
                "resolve 'auto' with resolve_face_analysis_engine() first"
            )
        if self.face_analysis_engine == "numba" and get_numba_face_kernel() is None:
            # Checked here, not just in resolve_face_analysis_engine(), so a
            # direct mesh_record() caller cannot get a silently empty result.
            raise RuntimeError("Numba acceleration is not installed. Install with: uv sync --extra numba")


def read_topology(
    mesh: UsdGeom.Mesh, time_code: Usd.TimeCode, phase_timer: PhaseTimer
) -> tuple[Any, Any, Any, Counter[str], dict[str, Any]]:
    """Read points, face-vertex counts, and indices as usable NumPy arrays.

    Returns the three arrays -- None when absent or unusable -- plus the issues
    found while reading them. A wrongly typed attribute is reported as
    ``*_wrong_type`` and then treated as unusable, which cascades the same way a
    missing attribute does.
    """
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    points_attr = mesh.GetPointsAttr()
    counts_attr = mesh.GetFaceVertexCountsAttr()
    indices_attr = mesh.GetFaceVertexIndicesAttr()
    with phase_timer.phase("mesh.read_attributes"):
        points = points_attr.Get(time_code)
        counts = counts_attr.Get(time_code)
        indices = indices_attr.Get(time_code)

    points_np, points_problem = shaped_array(points, 3)
    counts_np, counts_problem = shaped_array(counts, None, integer=True)
    indices_np, indices_problem = shaped_array(indices, None, integer=True)
    for issue, attr, value, problem in (
        ("points_wrong_type", points_attr, points, points_problem),
        ("face_vertex_counts_wrong_type", counts_attr, counts, counts_problem),
        ("face_vertex_indices_wrong_type", indices_attr, indices, indices_problem),
    ):
        if problem is not None:
            issues[issue] += 1
            add_example(details, issue, {**attribute_types(attr, value), **problem}, 10)

    if points is None:
        issues["missing_points"] += 1
    elif points_np is not None and len(points_np) == 0:
        issues["empty_points"] += 1
    if counts is None:
        issues["missing_face_vertex_counts"] += 1
    if indices is None:
        issues["missing_face_vertex_indices"] += 1
    return points_np, counts_np, indices_np, issues, details


def point_issues(points_np, huge_coord_threshold: float) -> tuple[Counter[str], dict[str, Any]]:
    """Report non-finite points and points beyond the huge-coordinate threshold."""
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    if points_np is None or not points_np.size:
        return issues, details
    finite_mask = np.isfinite(points_np).all(axis=1)
    non_finite_indices = np.flatnonzero(~finite_mask)
    if non_finite_indices.size:
        issues["non_finite_points"] += int(non_finite_indices.size)
        details["non_finite_point_examples"] = non_finite_indices[:10].astype(int).tolist()
    finite_points = points_np[finite_mask]
    if not finite_points.size:
        return issues, details
    max_abs = np.max(np.abs(finite_points), axis=1)
    huge_positions = np.flatnonzero(max_abs > huge_coord_threshold)
    if huge_positions.size:
        original_indices = np.flatnonzero(finite_mask)[huge_positions]
        issues["huge_coordinate_points"] += int(huge_positions.size)
        details["huge_coordinate_examples"] = [
            {
                "index": int(original_indices[i]),
                "point": finite_points[huge_positions[i]].astype(float).tolist(),
                "max_abs": float(max_abs[huge_positions[i]]),
            }
            for i in range(min(10, huge_positions.size))
        ]
    return issues, details


def extent_issues(mesh: UsdGeom.Mesh, time_code: Usd.TimeCode, bounds, extent_tolerance: float):
    """Report an unusable authored extent, or one that disagrees with the point bounds."""
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    authored_bounds, extent_problem = read_extent(mesh, time_code)
    if extent_problem is not None:
        issues["extent_wrong_type"] += 1
        add_example(details, "extent_wrong_type", extent_problem, 10)
    extent_delta = bounds_mismatch(authored_bounds, bounds, extent_tolerance)
    if extent_delta is not None:
        issues["authored_extent_mismatch"] += 1
        details["authored_extent_max_delta"] = extent_delta
    return issues, details


def determinant_issues(determinant: float | None) -> Counter[str]:
    """Report degenerate or mirroring local-to-world transforms."""
    issues: Counter[str] = Counter()
    if determinant is None:
        return issues
    if abs(determinant) <= 1e-12:
        issues["near_zero_transform_determinant"] += 1
    elif determinant < 0:
        issues["negative_transform_determinant"] += 1
    return issues


def _fold_legacy_mesh_settings(kwargs: dict[str, Any], warned: bool) -> dict[str, Any]:
    """Map mesh_record()'s old loose threshold arguments onto ``settings``."""
    loose = {name: kwargs.pop(name) for name in MeshCheckSettings.__dataclass_fields__ if name in kwargs}
    if loose:
        if kwargs.get("settings") is not None:
            raise TypeError("mesh_record() got both settings and loose threshold arguments")
    if loose and not warned:
        warnings.warn(
            f"passing {', '.join(loose)} to mesh_record() is deprecated and will stop working in the next "
            "major release; pass settings=MeshCheckSettings(...)",
            DeprecationWarning,
            stacklevel=3,
        )
    if loose:
        kwargs["settings"] = MeshCheckSettings(**loose)
    return kwargs


@accepts_legacy_positional(
    "zero_area_epsilon",
    "huge_coord_threshold",
    "extent_tolerance",
    "xform_cache",
    "face_analysis_engine",
    "audit_mode",
    "phase_timer",
    "face_cache",
    "time_code",
    "error_log",
    "prototype_paths",
    fold=_fold_legacy_mesh_settings,
)
def mesh_record(
    prim: Usd.Prim,
    *,
    settings: MeshCheckSettings | None = None,
    xform_cache: UsdGeom.XformCache | None = None,
    time_code: Usd.TimeCode | None = None,
    phase_timer: PhaseTimer | None = None,
    face_cache: FaceAnalysisCache | None = None,
    error_log: CheckErrorLog | None = None,
    prototype_paths: PrototypePaths | None = None,
) -> dict[str, Any]:
    """Analyze a single mesh prim and return a report record.

    Every option is keyword-only and defaults to what ``analyze()`` would use:

    - ``settings``: ``MeshCheckSettings()``, the CLI's default thresholds with
      the NumPy engine in exhaustive mode.
    - ``time_code``: ``default_time_code(prim)``.
    - ``xform_cache``: a new ``UsdGeom.XformCache(time_code)``, so attributes and
      transforms are read at one moment. When auditing many meshes, pass one
      cache built at ``time_code`` and reuse it -- a cache per call loses its
      ancestor caching. A default-constructed ``XformCache()`` evaluates at
      ``Default`` and would miss animated transforms.
    - ``phase_timer``: a throwaway ``PhaseTimer``.
    - ``face_cache``: None, meaning no face-analysis caching.
    - ``error_log``: None. Without a log, a failing check raises; with one, the
      failure is recorded and the rest of the record is kept.
    - ``prototype_paths``: None. When given, prims inside instancing prototypes
      are named by ``PrototypePaths`` in the record's ``path``;
      ``normalized_path`` and the category always use the real prim path.
    """
    settings = MeshCheckSettings() if settings is None else settings
    time_code = default_time_code(prim) if time_code is None else time_code
    xform_cache = UsdGeom.XformCache(time_code) if xform_cache is None else xform_cache
    phase_timer = PhaseTimer() if phase_timer is None else phase_timer
    prim_path = str(prim.GetPath())
    path = prototype_paths.stable(prim_path) if prototype_paths is not None else prim_path
    name = prim.GetName()
    mesh = UsdGeom.Mesh(prim)
    category = classify_mesh(prim_path, name)

    points_np, counts_np, indices_np, issues, details = read_topology(mesh, time_code, phase_timer)
    point_count = len(points_np) if points_np is not None else 0
    face_count = len(counts_np) if counts_np is not None else 0
    index_count = len(indices_np) if indices_np is not None else 0

    # Counts that later checks compare against. None means unknown: the
    # attribute is missing or unusable, which read_topology() already reported
    # once. Checks against an unknown count are skipped rather than run against
    # zero, which used to repeat that one defect as a finding per index and per
    # primvar. Empty points count as unknown too: empty_points is the defect.
    known_point_count = point_count if point_count else None
    known_face_count = face_count if counts_np is not None else None
    expected_index_count = int(counts_np.sum()) if counts_np is not None else None

    if counts_np is not None and indices_np is not None and expected_index_count != index_count:
        issues["face_vertex_count_index_length_mismatch"] += 1
        details["expected_index_count"] = expected_index_count

    with phase_timer.phase("mesh.point_checks"), recorded_failure("point_checks", path, error_log):
        found, found_details = point_issues(points_np, settings.huge_coord_threshold)
        issues.update(found)
        details.update(found_details)

    check_repeated_vertices, check_zero_area = face_checks_for_mode(settings.audit_mode, category)
    with phase_timer.phase("mesh.face_checks"), recorded_failure("face_checks", path, error_log):
        found, found_details = analyze_face_geometry(
            counts_np,
            indices_np,
            points_np,
            known_point_count,
            zero_area_epsilon=settings.zero_area_epsilon,
            face_analysis_engine=settings.face_analysis_engine,
            check_repeated_vertices=check_repeated_vertices,
            check_zero_area=check_zero_area,
            face_cache=face_cache,
        )
        issues.update(found)
        details.update(found_details)

    bounds = None
    with phase_timer.phase("mesh.bounds_extent"), recorded_failure("bounds_extent", path, error_log):
        bounds = bbox_from_points(points_np if points_np is not None else [])
        found, found_details = extent_issues(mesh, time_code, bounds, settings.extent_tolerance)
        issues.update(found)
        details.update(found_details)

    if settings.audit_mode != "fast":
        # Normals are recorded before primvars runs, so a failing primvar check
        # cannot cost the normals findings.
        with phase_timer.phase("mesh.normals"), recorded_failure("normals", path, error_log):
            for normal_issue in validate_normals(
                mesh, known_point_count, known_face_count, expected_index_count, time_code
            ):
                issues[normal_issue["issue"]] += 1
                add_example(details, normal_issue["issue"], normal_issue, 10)
        with phase_timer.phase("mesh.primvars"), recorded_failure("primvars", path, error_log):
            for primvar_issue in validate_primvars(
                prim, known_point_count, known_face_count, expected_index_count, time_code
            ):
                issues[primvar_issue["issue"]] += 1
                add_example(details, primvar_issue["issue"], primvar_issue, 10)

    with phase_timer.phase("mesh.transforms"):
        determinant = transform_determinant(prim, xform_cache, error_log, subject=path)
    issues.update(determinant_issues(determinant))

    return {
        "path": path,
        "normalized_path": normalized_path(prim_path),
        "name": name,
        "category": category,
        "point_count": point_count,
        "face_count": face_count,
        "face_vertex_index_count": index_count,
        "bounds": bounds,
        "extent_diagonal": bbox_diagonal(bounds),
        "transform_determinant": determinant,
        "issue_count": sum(issues.values()),
        "issues": dict(issues),
        "details": details,
    }


ISSUE_WEIGHTS = {
    "missing_points": 1000,
    "missing_face_vertex_counts": 1000,
    "missing_face_vertex_indices": 1000,
    "points_wrong_type": 1000,
    "face_vertex_counts_wrong_type": 1000,
    "face_vertex_indices_wrong_type": 1000,
    "face_vertex_count_index_length_mismatch": 800,
    "negative_face_vertex_indices": 500,
    "out_of_range_face_vertex_indices": 500,
    "non_finite_points": 500,
    "empty_points": 300,
    "empty_or_negative_faces": 250,
    "one_or_two_vertex_faces": 100,
    "faces_with_repeated_vertices": 50,
    "zero_area_triangles": 20,
    "primvar_length_mismatch": 100,
    "primvar_element_size_mismatch": 100,
    "primvar_index_out_of_range": 100,
    "normals_length_mismatch": 50,
    "normals_wrong_type": 100,
    "authored_extent_mismatch": 10,
    "extent_wrong_type": 10,
    "negative_transform_determinant": 5,
    "near_zero_transform_determinant": 100,
}

SERIOUS_ISSUE_NAMES = frozenset(
    {
        "missing_points",
        "missing_face_vertex_counts",
        "missing_face_vertex_indices",
        "points_wrong_type",
        "face_vertex_counts_wrong_type",
        "face_vertex_indices_wrong_type",
        "face_vertex_count_index_length_mismatch",
        "negative_face_vertex_indices",
        "out_of_range_face_vertex_indices",
        "non_finite_points",
        "empty_points",
        "empty_or_negative_faces",
        "one_or_two_vertex_faces",
        "primvar_index_out_of_range",
        "near_zero_transform_determinant",
    }
)

LIKELY_BENIGN_COLLISION_ISSUES = frozenset(
    {"faces_with_repeated_vertices", "zero_area_triangles", "authored_extent_mismatch"}
)

COMPACT_RECORD_KEYS = (
    "path",
    "normalized_path",
    "category",
    "point_count",
    "face_count",
    "face_vertex_index_count",
    "extent_diagonal",
    "issue_count",
    "issues",
    "details",
)

RANKING_LIMIT = 40


def seriousness_score(record: dict[str, Any]) -> int:
    """Rank mesh records by likely severity."""
    return sum(ISSUE_WEIGHTS.get(issue, 1) * count for issue, count in record["issues"].items())


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of a mesh record shown in ranked lists."""
    return {key: record[key] for key in COMPACT_RECORD_KEYS if key in record}


def audit_meshes(
    mesh_prims: list[Usd.Prim],
    settings: MeshCheckSettings,
    *,
    time_code: Usd.TimeCode,
    phase_timer: PhaseTimer,
    face_cache: FaceAnalysisCache,
    error_log: CheckErrorLog,
    prototype_paths: PrototypePaths,
) -> list[dict[str, Any]]:
    """Return one record per mesh that could be audited."""
    xform_cache = UsdGeom.XformCache(time_code)
    records: list[dict[str, Any]] = []
    for prim in mesh_prims:
        try:
            records.append(
                mesh_record(
                    prim,
                    settings=settings,
                    xform_cache=xform_cache,
                    time_code=time_code,
                    phase_timer=phase_timer,
                    face_cache=face_cache,
                    error_log=error_log,
                    prototype_paths=prototype_paths,
                )
            )
        except Exception as error:  # noqa: BLE001 - recorded below; see CheckErrorLog
            # Last resort: each check phase inside mesh_record() already records
            # its own failures and keeps the rest of the record. Anything that
            # still escapes must not abort a multi-minute audit of every other
            # mesh. Such a mesh is counted in unaudited_mesh_count.
            error_log.record("mesh_record", prototype_paths.stable(str(prim.GetPath())), error)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mesh records into the report's summary, example, and ranking sections."""
    summary_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = {}
    for record in records:
        category = record["category"]
        category_counts[category] += 1
        for issue, count in record["issues"].items():
            summary_counts[issue] += count
            category_issue_counts[category][issue] += count
            add_example(
                examples,
                issue,
                {
                    "path": record["path"],
                    "normalized_path": record["normalized_path"],
                    "category": category,
                    "count": count,
                    "details": record["details"],
                },
            )

    def ranked(key) -> list[dict[str, Any]]:
        return [compact_record(record) for record in sorted(records, key=key, reverse=True)[:RANKING_LIMIT]]

    # category_issue_counts always carries a "collision_like" entry, empty when
    # there are none, so consumers can read it without a key check.
    collision_issue_counts = category_issue_counts["collision_like"]
    likely_benign = sum(
        count for issue, count in collision_issue_counts.items() if issue in LIKELY_BENIGN_COLLISION_ISSUES
    )
    return {
        "summary_counts": dict(summary_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "category_issue_counts": {
            category: dict(counter.most_common()) for category, counter in category_issue_counts.items()
        },
        "serious_geometry_failures": sum(summary_counts[name] for name in SERIOUS_ISSUE_NAMES),
        "likely_benign_collision_helper_warnings": likely_benign,
        "examples": examples,
        "worst_meshes": ranked(lambda item: (seriousness_score(item), item["issue_count"])),
        "largest_meshes_by_points": ranked(lambda item: item["point_count"]),
        "largest_meshes_by_faces": ranked(lambda item: item["face_count"]),
        "largest_meshes_by_extent": ranked(lambda item: item["extent_diagonal"]),
    }


@accepts_legacy_positional(
    "zero_area_epsilon",
    "huge_coord_threshold",
    "extent_tolerance",
    "geometry_engine",
    "audit_mode",
    "mesh_cache_mode",
    "frame",
)
def analyze(
    stage_path: Path,
    *,
    zero_area_epsilon: float = DEFAULT_ZERO_AREA_EPSILON,
    huge_coord_threshold: float = DEFAULT_HUGE_COORD_THRESHOLD,
    extent_tolerance: float = DEFAULT_EXTENT_TOLERANCE,
    geometry_engine: str = DEFAULT_GEOMETRY_ENGINE,
    audit_mode: str = DEFAULT_AUDIT_MODE,
    mesh_cache_mode: str = DEFAULT_MESH_CACHE_MODE,
    frame: float | None = None,
) -> dict[str, Any]:
    """Analyze all meshes in a USD stage and its prototypes.

    Options are keyword-only: the three thresholds are adjacent floats, so a
    positional call could swap two of them without any error. Positional
    options still work for one release, with a DeprecationWarning.
    """
    started = time.perf_counter()
    phase_timer = PhaseTimer()
    settings = MeshCheckSettings(
        zero_area_epsilon=zero_area_epsilon,
        huge_coord_threshold=huge_coord_threshold,
        extent_tolerance=extent_tolerance,
        face_analysis_engine=resolve_face_analysis_engine(geometry_engine),
        audit_mode=audit_mode,
    )
    if mesh_cache_mode not in MESH_CACHE_MODES:
        raise ValueError(f"Unsupported mesh cache mode: {mesh_cache_mode}")
    face_cache = FaceAnalysisCache(enabled=mesh_cache_mode == "face-hash")
    error_log = CheckErrorLog()
    with phase_timer.phase("stage.open"):
        stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    time_code = resolve_time_code(frame, stage)
    with phase_timer.phase("stage.traverse"):
        prims, prototype_count = prims_with_prototypes(stage)
        mesh_prims = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
        prototype_paths = PrototypePaths(stage)

    records = audit_meshes(
        mesh_prims,
        settings,
        time_code=time_code,
        phase_timer=phase_timer,
        face_cache=face_cache,
        error_log=error_log,
        prototype_paths=prototype_paths,
    )
    summary = summarize_records(records)

    return {
        "stage": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prototype_count": prototype_count,
        "mesh_count": len(mesh_prims),
        "unaudited_mesh_count": len(mesh_prims) - len(records),
        "thresholds": {
            "zero_area_epsilon": zero_area_epsilon,
            "huge_coord_threshold": huge_coord_threshold,
            "extent_tolerance": extent_tolerance,
        },
        "audit_mode": audit_mode,
        "geometry_engine": geometry_engine,
        "face_analysis_engine": settings.face_analysis_engine,
        "requested_frame": frame,
        "time_code": describe_time_code(time_code),
        "mesh_cache": face_cache.stats(),
        "summary_counts": summary["summary_counts"],
        "category_counts": summary["category_counts"],
        "category_issue_counts": summary["category_issue_counts"],
        "serious_geometry_failures": summary["serious_geometry_failures"],
        "likely_benign_collision_helper_warnings": summary["likely_benign_collision_helper_warnings"],
        "check_errors": error_log.as_report(),
        "examples": summary["examples"],
        "worst_meshes": summary["worst_meshes"],
        "largest_meshes_by_points": summary["largest_meshes_by_points"],
        "largest_meshes_by_faces": summary["largest_meshes_by_faces"],
        "largest_meshes_by_extent": summary["largest_meshes_by_extent"],
        "phase_timings": phase_timer.rounded(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact console summary."""
    print(f"Stage: {report['stage']}")
    print(f"Meshes scanned: {report['mesh_count']} (prototypes: {report['prototype_count']})")
    if report["unaudited_mesh_count"]:
        print(f"Meshes with no record: {report['unaudited_mesh_count']} (see check_errors in the JSON report)")
    print(f"Audit mode: {report['audit_mode']}")
    print(f"Evaluated at time code: {report['time_code']}")
    print(f"Face analysis engine: {report['face_analysis_engine']}")
    print(f"Mesh cache: {report['mesh_cache']}")
    print(f"Categories: {report['category_counts']}")
    print(f"Serious geometry failures: {report['serious_geometry_failures']}")
    print(f"Likely benign collision/helper warnings: {report['likely_benign_collision_helper_warnings']}")
    check_errors = report["check_errors"]["count"]
    if check_errors:
        print(f"Checks that failed to run: {check_errors} (see check_errors in the JSON report)")
    print("Summary counts:")
    for issue, count in report["summary_counts"].items():
        print(f"  {issue}: {count}")
    print("Top 10 worst offender paths:")
    for record in report["worst_meshes"][:10]:
        print(f"  {record['normalized_path']} [{record['category']}] issues={record['issues']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--zero-area-epsilon", type=float, default=DEFAULT_ZERO_AREA_EPSILON)
    parser.add_argument("--huge-coord-threshold", type=float, default=DEFAULT_HUGE_COORD_THRESHOLD)
    parser.add_argument("--extent-tolerance", type=float, default=DEFAULT_EXTENT_TOLERANCE)
    parser.add_argument(
        "--audit-mode",
        choices=AUDIT_MODES,
        default=DEFAULT_AUDIT_MODE,
        help=(
            "fast checks topology/bounds only, standard skips exact collision zero-area checks, "
            "exhaustive keeps all exact checks."
        ),
    )
    parser.add_argument(
        "--geometry-engine",
        choices=FACE_ANALYSIS_ENGINES,
        default=DEFAULT_GEOMETRY_ENGINE,
        help="Engine for expensive per-face checks. Auto uses Numba when installed, otherwise NumPy.",
    )
    parser.add_argument(
        "--mesh-cache",
        choices=MESH_CACHE_MODES,
        default=DEFAULT_MESH_CACHE_MODE,
        help="Optional cache for duplicate face arrays. face-hash can help repeated geometry but costs hashing time.",
    )
    parser.add_argument(
        "--frame",
        type=float,
        default=None,
        help=(
            "Time code at which to read mesh attributes. Defaults to the stage's authored "
            "startTimeCode, else the earliest authored time sample, falling back to the "
            "default value for static geometry."
        ),
    )
    args = parser.parse_args()

    report = analyze(
        args.stage,
        zero_area_epsilon=args.zero_area_epsilon,
        huge_coord_threshold=args.huge_coord_threshold,
        extent_tolerance=args.extent_tolerance,
        geometry_engine=args.geometry_engine,
        audit_mode=args.audit_mode,
        mesh_cache_mode=args.mesh_cache,
        frame=args.frame,
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
