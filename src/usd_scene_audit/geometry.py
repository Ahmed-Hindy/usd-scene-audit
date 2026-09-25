"""Audit mesh geometry in a composed USD stage, including prototypes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pxr import Gf, Usd, UsdGeom

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback exists for tiny ad hoc runs.
    np = None


MAX_EXAMPLES = 80
FACE_CHUNK_SIZE = 100_000
FACE_ANALYSIS_ENGINES = ("auto", "numpy", "numba")
AUDIT_MODES = ("fast", "standard", "exhaustive")
MESH_CACHE_MODES = ("off", "face-hash")
_NUMBA_FACE_KERNEL = None
_NUMBA_IMPORT_ERROR: Exception | None = None


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


def resolve_time_code(frame: float | None) -> Usd.TimeCode:
    """Return the time code used to read geometry attributes.

    Reading at ``Usd.TimeCode.Default()`` resolves only an attribute's default
    value. Deforming geometry normally authors ``points`` purely as time samples
    with no default, so a default-time read returns ``None`` and the mesh looks
    like it is missing its points entirely.

    ``EarliestTime()`` resolves to the first authored time sample when one
    exists and falls back to the default value otherwise, so it is safe for
    static and animated geometry alike.
    """
    if frame is None:
        return Usd.TimeCode.EarliestTime()
    return Usd.TimeCode(float(frame))


def describe_time_code(time_code: Usd.TimeCode) -> str | float:
    """Return a JSON-friendly description of an evaluated time code."""
    if time_code.IsEarliestTime():
        return "earliest"
    if time_code.IsDefault():
        return "default"
    return float(time_code.GetValue())


def prims_with_prototypes(stage: Usd.Stage) -> tuple[list[Usd.Prim], int]:
    """Return ordinary stage traversal plus prototype contents."""
    prims = list(stage.Traverse())
    prototypes = list(stage.GetPrototypes())
    for prototype in prototypes:
        prims.extend(list(Usd.PrimRange(prototype)))
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


def is_finite_vec3(value) -> bool:
    """Return true when a vec-like value contains finite xyz values."""
    return all(math.isfinite(float(value[i])) for i in range(3))


def vec3_tuple(value) -> tuple[float, float, float]:
    """Convert a Gf vec-like object to a Python tuple."""
    return (float(value[0]), float(value[1]), float(value[2]))


def bbox_from_points(points) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Compute min and max bounds from finite points."""
    if np is not None:
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
    finite_points = [p for p in points if is_finite_vec3(p)]
    if not finite_points:
        return None
    mins = [min(float(p[i]) for p in finite_points) for i in range(3)]
    maxs = [max(float(p[i]) for p in finite_points) for i in range(3)]
    return (tuple(mins), tuple(maxs))


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
    """Return authored extent as a bounds tuple, or None."""
    if time_code is None:
        time_code = Usd.TimeCode.EarliestTime()
    extent = mesh.GetExtentAttr().Get(time_code)
    if not extent or len(extent) != 2:
        return None
    return (vec3_tuple(extent[0]), vec3_tuple(extent[1]))


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


def expected_primvar_length(interpolation: str, point_count: int, face_count: int, face_vertex_count: int) -> int | None:
    """Return expected primvar element count for a USD interpolation."""
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


def vector_array(value, dtype=None):
    """Convert a USD Vt array to a NumPy array when NumPy is available."""
    if value is None or np is None:
        return None
    return np.asarray(value, dtype=dtype)


def array_digest(array) -> tuple[str, tuple[int, ...], str] | None:
    """Return a stable digest for a NumPy array without changing audit results."""
    if array is None or np is None:
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
    if np is None:
        if requested_engine == "numba":
            raise RuntimeError("Numba acceleration requires NumPy.")
        return "numpy"
    kernel = get_numba_face_kernel()
    if kernel is not None:
        return "numba"
    if requested_engine == "auto":
        return "numpy"
    raise RuntimeError("Numba acceleration is not installed. Install with: uv sync --extra numba")


def count_repeated_rows(index_rows) -> tuple[int, list[list[int]]]:
    """Count rows that contain repeated vertex indices."""
    if np is None or index_rows.size == 0:
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
    if np is None or points_np is None or index_rows.size == 0 or index_rows.shape[1] < 3:
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
        return 0, [], 0, []
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


def analyze_face_geometry(
    counts_np,
    indices_np,
    points_np,
    point_count: int,
    zero_area_epsilon: float,
    face_analysis_engine: str,
    check_repeated_vertices: bool = True,
    check_zero_area: bool = True,
    face_cache: FaceAnalysisCache | None = None,
) -> tuple[Counter[str], dict[str, Any]]:
    """Validate face arrays with exact checks."""
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
        if face_cache is not None
        else None
    )
    if face_cache is not None:
        cached = face_cache.get(cache_key)
        if cached is not None:
            return cached

    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}
    if np is None or counts_np is None or indices_np is None:
        return issues, details

    if counts_np.size == 0:
        return issues, details

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
        out_of_range_mask = indices_np >= point_count
        if np.any(negative_mask):
            bad = indices_np[negative_mask]
            issues["negative_face_vertex_indices"] += int(bad.size)
            details["negative_index_examples"] = bad[:10].astype(int).tolist()
        if np.any(out_of_range_mask):
            bad = indices_np[out_of_range_mask]
            issues["out_of_range_face_vertex_indices"] += int(bad.size)
            details["out_of_range_index_examples"] = bad[:10].astype(int).tolist()

    expected_index_count = int(counts_np.sum())
    if expected_index_count != indices_np.size:
        return issues, details

    if point_count == 0 or indices_np.size == 0 or np.any(counts_np < 0):
        return issues, details

    if np.any(indices_np < 0) or np.any(indices_np >= point_count):
        return issues, details

    if not check_repeated_vertices and not check_zero_area:
        if face_cache is not None:
            face_cache.set(cache_key, issues, details)
        return issues, details

    offsets = np.empty(counts_np.size, dtype=np.int64)
    offsets[0] = 0
    if counts_np.size > 1:
        offsets[1:] = np.cumsum(counts_np[:-1], dtype=np.int64)

    if face_analysis_engine == "numba":
        repeated_total, repeated_examples, zero_area_total, zero_area_examples = count_face_geometry_numba(
            counts_np,
            indices_np,
            points_np,
            offsets,
            zero_area_epsilon,
            check_repeated_vertices,
            check_zero_area,
        )
    else:
        repeated_total, repeated_examples, zero_area_total, zero_area_examples = count_face_geometry_numpy(
            counts_np,
            indices_np,
            points_np,
            offsets,
            zero_area_epsilon,
            check_repeated_vertices,
            check_zero_area,
        )

    if repeated_total:
        issues["faces_with_repeated_vertices"] += repeated_total
        details["repeated_vertex_face_examples"] = repeated_examples
    if zero_area_total:
        issues["zero_area_triangles"] += zero_area_total
        details["zero_area_triangle_examples"] = zero_area_examples

    if face_cache is not None:
        face_cache.set(cache_key, issues, details)
    return issues, details


def validate_primvars(
    prim: Usd.Prim,
    point_count: int,
    face_count: int,
    face_vertex_count: int,
    time_code: Usd.TimeCode | None = None,
) -> list[dict[str, Any]]:
    """Validate authored primvar lengths, with special attention to UV-like primvars."""
    if time_code is None:
        time_code = Usd.TimeCode.EarliestTime()
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
            if np is not None:
                indices_np = np.asarray(indices, dtype=np.int64)
                bad_positions = np.flatnonzero((indices_np < 0) | (indices_np >= value_element_count))
                bad_indices = indices_np[bad_positions[:10]].astype(int).tolist()
            else:
                bad_indices = [int(i) for i in indices if int(i) < 0 or int(i) >= value_element_count][:10]
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
    point_count: int,
    face_count: int,
    face_vertex_count: int,
    time_code: Usd.TimeCode | None = None,
) -> list[dict[str, Any]]:
    """Validate authored normals length and finite values."""
    if time_code is None:
        time_code = Usd.TimeCode.EarliestTime()
    issues: list[dict[str, Any]] = []
    normals = mesh.GetNormalsAttr().Get(time_code)
    if normals is None:
        return issues
    interpolation = mesh.GetNormalsInterpolation() or ""
    normal_count = len(normals)
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
    if np is not None:
        normals_np = np.asarray(normals)
        if normals_np.size == 0:
            non_finite = []
        elif normals_np.ndim == 2:
            finite_mask = np.isfinite(normals_np).all(axis=1)
            non_finite = np.flatnonzero(~finite_mask)[:10].astype(int).tolist()
        else:
            non_finite = [i for i, normal in enumerate(normals) if not is_finite_vec3(normal)][:10]
    else:
        non_finite = [i for i, normal in enumerate(normals) if not is_finite_vec3(normal)][:10]
    if non_finite:
        issues.append({"issue": "normals_non_finite", "index_examples": non_finite})
    return issues


def transform_determinant(prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> float | None:
    """Return local-to-world transform determinant, if computable."""
    try:
        transform = xform_cache.GetLocalToWorldTransform(prim)
        return float(transform.GetDeterminant())
    except Exception:
        return None


def mesh_record(
    prim: Usd.Prim,
    zero_area_epsilon: float,
    huge_coord_threshold: float,
    extent_tolerance: float,
    xform_cache: UsdGeom.XformCache,
    face_analysis_engine: str,
    audit_mode: str,
    phase_timer: PhaseTimer,
    face_cache: FaceAnalysisCache,
    time_code: Usd.TimeCode | None = None,
) -> dict[str, Any]:
    """Analyze a single mesh prim and return a report record."""
    if time_code is None:
        time_code = Usd.TimeCode.EarliestTime()
    path = str(prim.GetPath())
    name = prim.GetName()
    mesh = UsdGeom.Mesh(prim)
    category = classify_mesh(path, name)
    issues: Counter[str] = Counter()
    details: dict[str, Any] = {}

    with phase_timer.phase("mesh.read_attributes"):
        points = mesh.GetPointsAttr().Get(time_code)
        counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)

    point_count = len(points) if points is not None else 0
    face_count = len(counts) if counts is not None else 0
    index_count = len(indices) if indices is not None else 0
    points_np = vector_array(points)
    counts_np = vector_array(counts, dtype="int64")
    indices_np = vector_array(indices, dtype="int64")

    if points is None:
        issues["missing_points"] += 1
    elif point_count == 0:
        issues["empty_points"] += 1

    if counts is None:
        issues["missing_face_vertex_counts"] += 1
        counts = []
    if indices is None:
        issues["missing_face_vertex_indices"] += 1
        indices = []

    expected_index_count = int(counts_np.sum()) if counts_np is not None else sum(int(c) for c in counts)
    if expected_index_count != index_count:
        issues["face_vertex_count_index_length_mismatch"] += 1
        details["expected_index_count"] = expected_index_count

    with phase_timer.phase("mesh.point_checks"):
        if points_np is not None and points_np.size:
            finite_mask = np.isfinite(points_np).all(axis=1) if np is not None else None
            if finite_mask is not None:
                non_finite_indices = np.flatnonzero(~finite_mask)
                if non_finite_indices.size:
                    issues["non_finite_points"] += int(non_finite_indices.size)
                    details["non_finite_point_examples"] = non_finite_indices[:10].astype(int).tolist()
                finite_points = points_np[finite_mask]
                if finite_points.size:
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

    check_repeated_vertices, check_zero_area = face_checks_for_mode(audit_mode, category)
    with phase_timer.phase("mesh.face_checks"):
        face_issues, face_details = analyze_face_geometry(
            counts_np,
            indices_np,
            points_np,
            point_count,
            zero_area_epsilon,
            face_analysis_engine,
            check_repeated_vertices,
            check_zero_area,
            face_cache,
        )
    issues.update(face_issues)
    details.update(face_details)

    with phase_timer.phase("mesh.bounds_extent"):
        bounds = bbox_from_points(points if points is not None else [])
        extent_delta = bounds_mismatch(authored_extent_bounds(mesh, time_code), bounds, extent_tolerance)
        if extent_delta is not None:
            issues["authored_extent_mismatch"] += 1
            details["authored_extent_max_delta"] = extent_delta

    if audit_mode != "fast":
        with phase_timer.phase("mesh.normals"):
            for normal_issue in validate_normals(mesh, point_count, face_count, expected_index_count, time_code):
                issues[normal_issue["issue"]] += 1
                add_example(details, normal_issue["issue"], normal_issue, 10)

        with phase_timer.phase("mesh.primvars"):
            for primvar_issue in validate_primvars(prim, point_count, face_count, expected_index_count, time_code):
                issues[primvar_issue["issue"]] += 1
                add_example(details, primvar_issue["issue"], primvar_issue, 10)

    with phase_timer.phase("mesh.transforms"):
        determinant = transform_determinant(prim, xform_cache)
        if determinant is not None:
            if abs(determinant) <= 1e-12:
                issues["near_zero_transform_determinant"] += 1
            elif determinant < 0:
                issues["negative_transform_determinant"] += 1

    return {
        "path": path,
        "normalized_path": normalized_path(path),
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


def seriousness_score(record: dict[str, Any]) -> int:
    """Rank mesh records by likely severity."""
    weights = {
        "missing_points": 1000,
        "missing_face_vertex_counts": 1000,
        "missing_face_vertex_indices": 1000,
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
        "authored_extent_mismatch": 10,
        "negative_transform_determinant": 5,
        "near_zero_transform_determinant": 100,
    }
    return sum(weights.get(issue, 1) * count for issue, count in record["issues"].items())


def analyze(
    stage_path: Path,
    zero_area_epsilon: float,
    huge_coord_threshold: float,
    extent_tolerance: float,
    geometry_engine: str = "auto",
    audit_mode: str = "exhaustive",
    mesh_cache_mode: str = "off",
    frame: float | None = None,
) -> dict[str, Any]:
    """Analyze all meshes in a USD stage and its prototypes."""
    started = time.perf_counter()
    phase_timer = PhaseTimer()
    time_code = resolve_time_code(frame)
    face_analysis_engine = resolve_face_analysis_engine(geometry_engine)
    if audit_mode not in AUDIT_MODES:
        raise ValueError(f"Unsupported audit mode: {audit_mode}")
    if mesh_cache_mode not in MESH_CACHE_MODES:
        raise ValueError(f"Unsupported mesh cache mode: {mesh_cache_mode}")
    face_cache = FaceAnalysisCache(enabled=mesh_cache_mode == "face-hash")
    with phase_timer.phase("stage.open"):
        stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    with phase_timer.phase("stage.traverse"):
        prims, prototype_count = prims_with_prototypes(stage)
        mesh_prims = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]

    summary_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = {}
    records: list[dict[str, Any]] = []
    xform_cache = UsdGeom.XformCache(time_code)

    for prim in mesh_prims:
        record = mesh_record(
            prim,
            zero_area_epsilon,
            huge_coord_threshold,
            extent_tolerance,
            xform_cache,
            face_analysis_engine,
            audit_mode,
            phase_timer,
            face_cache,
            time_code,
        )
        records.append(record)
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

    worst_meshes = sorted(records, key=lambda item: (seriousness_score(item), item["issue_count"]), reverse=True)[:40]
    largest_by_points = sorted(records, key=lambda item: item["point_count"], reverse=True)[:40]
    largest_by_faces = sorted(records, key=lambda item: item["face_count"], reverse=True)[:40]
    largest_by_extent = sorted(records, key=lambda item: item["extent_diagonal"], reverse=True)[:40]

    serious_issue_names = {
        "missing_points",
        "missing_face_vertex_counts",
        "missing_face_vertex_indices",
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
    serious_geometry_failures = sum(summary_counts[name] for name in serious_issue_names)
    likely_benign_collision_helper_warnings = sum(
        count
        for issue, count in category_issue_counts["collision_like"].items()
        if issue in {"faces_with_repeated_vertices", "zero_area_triangles", "authored_extent_mismatch"}
    )

    compact_record_keys = [
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
    ]

    def compact(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in compact_record_keys if key in record}

    report = {
        "stage": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "prototype_count": prototype_count,
        "mesh_count": len(mesh_prims),
        "thresholds": {
            "zero_area_epsilon": zero_area_epsilon,
            "huge_coord_threshold": huge_coord_threshold,
            "extent_tolerance": extent_tolerance,
        },
        "audit_mode": audit_mode,
        "geometry_engine": geometry_engine,
        "face_analysis_engine": face_analysis_engine,
        "requested_frame": frame,
        "time_code": describe_time_code(time_code),
        "mesh_cache": face_cache.stats(),
        "summary_counts": dict(summary_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
        "category_issue_counts": {category: dict(counter.most_common()) for category, counter in category_issue_counts.items()},
        "serious_geometry_failures": serious_geometry_failures,
        "likely_benign_collision_helper_warnings": likely_benign_collision_helper_warnings,
        "examples": examples,
        "worst_meshes": [compact(record) for record in worst_meshes],
        "largest_meshes_by_points": [compact(record) for record in largest_by_points],
        "largest_meshes_by_faces": [compact(record) for record in largest_by_faces],
        "largest_meshes_by_extent": [compact(record) for record in largest_by_extent],
        "phase_timings": phase_timer.rounded(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact console summary."""
    print(f"Stage: {report['stage']}")
    print(f"Meshes scanned: {report['mesh_count']} (prototypes: {report['prototype_count']})")
    print(f"Audit mode: {report['audit_mode']}")
    print(f"Evaluated at time code: {report['time_code']}")
    print(f"Face analysis engine: {report['face_analysis_engine']}")
    print(f"Mesh cache: {report['mesh_cache']}")
    print(f"Categories: {report['category_counts']}")
    print(f"Serious geometry failures: {report['serious_geometry_failures']}")
    print(f"Likely benign collision/helper warnings: {report['likely_benign_collision_helper_warnings']}")
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
    parser.add_argument("--zero-area-epsilon", type=float, default=1e-12)
    parser.add_argument("--huge-coord-threshold", type=float, default=1e6)
    parser.add_argument("--extent-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--audit-mode",
        choices=AUDIT_MODES,
        default="exhaustive",
        help=(
            "fast checks topology/bounds only, standard skips exact collision zero-area checks, "
            "exhaustive keeps all exact checks."
        ),
    )
    parser.add_argument(
        "--geometry-engine",
        choices=FACE_ANALYSIS_ENGINES,
        default="auto",
        help="Engine for expensive per-face checks. Auto uses Numba when installed, otherwise NumPy.",
    )
    parser.add_argument(
        "--mesh-cache",
        choices=MESH_CACHE_MODES,
        default="off",
        help="Optional cache for duplicate face arrays. face-hash can help repeated geometry but costs hashing time.",
    )
    parser.add_argument(
        "--frame",
        type=float,
        default=None,
        help=(
            "Time code at which to read mesh attributes. Defaults to the earliest authored "
            "time sample, falling back to the default value for static geometry."
        ),
    )
    args = parser.parse_args()

    report = analyze(
        args.stage,
        args.zero_area_epsilon,
        args.huge_coord_threshold,
        args.extent_tolerance,
        args.geometry_engine,
        args.audit_mode,
        args.mesh_cache,
        args.frame,
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
