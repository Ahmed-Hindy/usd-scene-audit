# USD Geometry Audit Details

`usd-geometry-audit` is intended for large OpenUSD stages where correctness matters more than a fast shallow scan.

## Mesh Inclusion

The command scans:

- `stage.Traverse()`
- every prim under `stage.GetPrototypes()`

This is important for instanceable USD assets because ordinary traversal sees only the instance prims, not the prototype mesh contents.

## Time Samples

Mesh attributes are read at one time code per audit, chosen in this order:

1. An explicitly requested `--frame`.
2. The stage's authored `startTimeCode`, when it has one.
3. `Usd.TimeCode.EarliestTime()`.

This matters because USD attribute resolution distinguishes an attribute's *default value* from its *time samples*. Deforming geometry — simulation caches, cloth, crowd agents, imported Alembic — normally authors `points` purely as time samples with no default value at all. Reading such an attribute at `Usd.TimeCode.Default()` resolves to nothing, which previously made every animated mesh look like it was missing its points, and in turn made every face-vertex index look out of range.

### Why one concrete time code, not `EarliestTime()` everywhere

`EarliestTime()` is not a single moment. It resolves *each attribute* at that attribute's own first authored sample. Several checks compare two attributes against each other — authored `extent` against computed point bounds, authored `normals` length against the point count — so a mesh whose attributes are sampled over different frame ranges would be compared across two different moments.

A cache with pre-roll on `points` but `extent` authored only over the shot range reports an extent mismatch that exists at no real frame. Preferring the stage's authored `startTimeCode` keeps every attribute on the same frame, and makes the default audit describe the shot rather than whatever pre-roll sample happens to sort earliest. `tests/fixtures/animated_preroll_extent.usda` pins this behaviour.

`EarliestTime()` remains the fallback for stages with no authored time range: it resolves to the first authored sample when one exists and falls back to the default value otherwise, so static geometry is handled correctly too.

### Selecting a frame

```powershell
uv run usd-geometry-audit scene.usd --frame 1001
```

Reports record `requested_frame` (the value passed, or `null`) and `time_code` (the numeric frame evaluated, or `"earliest"`/`"default"`), so a report states which moment in time it describes.

### What a single-frame audit can and cannot find

Each audit evaluates one time code, so a defect is found only if it is present at that time code. A defect at a *specific* frame is reachable by auditing that frame: `--frame 5` finds a point that goes non-finite at frame 5.

What a single-frame audit cannot establish is that a value *changed* between frames — for example that a point count is 3 at one frame and 1 at another while `faceVertexIndices` stays static. That requires comparing samples, which needs a sampling pass. `tests/fixtures/animated_topology_change.usda` captures that case and is the regression target for adding time-sampled checks.

## Mesh Categories

Every mesh is included. Categories only make the report easier to read:

- `render_like`: default category
- `collision_like`: names containing `_COL`, ending in `_C` or `_CO`, or containing `collision`
- `unknown_helper_like`: helper/proxy/guide path tokens

## Output

The JSON report contains:

- `requested_frame` and `time_code`: the moment in time the audit describes
- `summary_counts`: total issue counts
- `category_counts`: mesh counts by category
- `category_issue_counts`: issue counts by category
- `worst_meshes`: issue-ranked mesh records
- `largest_meshes_by_points`
- `largest_meshes_by_faces`
- `largest_meshes_by_extent`
- bounded examples for each issue type
- `check_errors`: checks that raised instead of running. A failing check costs only its own findings; the rest of that mesh's record is kept. Treat a non-zero count as "this report is incomplete", not as a clean result.
- `unaudited_mesh_count`: meshes counted in `mesh_count` that have no record at all, because a failure escaped every per-check guard. Each one also appears in `check_errors`.

### Wrongly typed attributes

Attribute getters return the value type a layer authored, not the schema type. When `points`, `faceVertexCounts`, `faceVertexIndices`, `normals`, or `extent` is authored with a value the checks cannot use, the audit reports `points_wrong_type`, `face_vertex_counts_wrong_type`, `face_vertex_indices_wrong_type`, `normals_wrong_type`, or `extent_wrong_type`. Examples are `float[] normals`, a scalar `int faceVertexCounts`, and `float[]` or `bool[]` face-vertex indices, which would otherwise be truncated into plausible-looking topology. Each detail records `authored_type`, `expected_type`, and a `reason`.

An unusable topology attribute is then treated like a missing one, and the three topology codes count toward `serious_geometry_failures`. Numeric arrays of the right shape, such as `double3[]` or `half3[]` points or `int64[]` indices, are audited normally.

## Default Thresholds

- zero-area triangle area: `1e-12`
- huge coordinate warning: absolute component over `1e6`
- authored extent mismatch: greater than `1e-4`

Override them with:

```powershell
uv run usd-geometry-audit scene.usd --zero-area-epsilon 1e-10 --huge-coord-threshold 100000 --extent-tolerance 1e-3
```

## Audit Modes

`--audit-mode exhaustive` is the default and preserves the original exact behavior. It runs all supported checks on all mesh categories.

`--audit-mode standard` keeps all cheap topology, index, point, bounds, normal, primvar, and transform checks. It still scans render/helper meshes for repeated vertices and zero-area fan triangles, but only runs repeated-vertex face checks on collision-like meshes. This is meant for production triage where collision zero-area triangles are usually lower signal than render-mesh issues.

`--audit-mode fast` keeps the cheap topology/index/point/bounds/transform checks and skips exact deep face scans, normals, and primvars. Use this mode when you need a quick answer about serious mesh-array corruption before committing to the full exact audit.

Every report records `audit_mode` so downstream comparisons can tell exact and reduced-scope audits apart.

## Acceleration

The expensive part of the audit is the exact per-face geometry pass: repeated vertex checks and fan-triangulated zero-area triangle checks. The default `--geometry-engine auto` uses Numba when the optional extra is installed and falls back to NumPy otherwise.

Install the optional extra with:

```powershell
uv sync --extra numba
```

Force a specific engine with:

```powershell
uv run usd-geometry-audit scene.usd --geometry-engine numba
uv run usd-geometry-audit scene.usd --geometry-engine numpy
```

The JSON report includes both `geometry_engine`, the requested mode, and `face_analysis_engine`, the engine actually used. Numba only accelerates numeric mesh-array checks; USD composition, prototype traversal, attribute reads, JSON writing, normal validation, and primvar validation still run through the Python/OpenUSD path.

On one optimized full-scene USD audit, Numba matched the NumPy findings and reduced exact audit time from about 68 seconds to about 44 seconds. Keep benchmarking per scene, because the speedup only applies to the face-check portion of the run.

## Timing And Caching

Reports include `phase_timings`, which breaks the audit into broad buckets such as stage open, traversal, attribute reads, point checks, face checks, primvars, normals, bounds/extent checks, and transforms.

`--mesh-cache face-hash` caches face-check results for duplicate point/count/index arrays. This is exact, but not automatically faster: hashing large arrays costs time, and scenes with few duplicate mesh arrays may lose performance. The report includes `mesh_cache` stats with entries, hits, and misses so the cache can be evaluated per asset.

Example:

```powershell
uv run usd-geometry-audit scene.usd --audit-mode standard --mesh-cache face-hash --json-out usd_geometry_audit_standard_cached.json
```

## Large Scene Performance Notes

One large USD scene audit showed that the original bottleneck was not Numba-sized face math. It was authored normal validation walking huge normal arrays in Python. Vectorized normal and indexed-primvar checks reduced the full exact audit from roughly 11 minutes to about 68 seconds on the test machine.

Measured full-stage timings after optimization:

- `--audit-mode fast`: about 26 seconds; reduced-scope triage, no findings on this asset.
- `--audit-mode standard --geometry-engine numpy`: about 62 seconds; same `32` zero-area render-triangle findings as exhaustive for this asset.
- `--audit-mode standard --geometry-engine numba`: about 45 seconds; same findings as NumPy.
- `--audit-mode exhaustive --geometry-engine numpy`: about 68 seconds; exact full audit.
- `--audit-mode exhaustive --geometry-engine numba`: about 44 seconds; exact full audit with same findings as NumPy.
- `--audit-mode exhaustive --mesh-cache face-hash`: about 62 seconds; exact full audit, with `641` duplicate face-array cache hits.

These timings were measured before [#36](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/36) was fixed. At that point the default `--mesh-cache off` still hashed every mesh's face arrays, so the `off` rows include hashing work that no longer happens. That makes the `face-hash` row look better than it would now. Re-measure before relying on the cache comparison.
