# USD Scene Audit

Command-line OpenUSD scene audit tools for geometry, naming, hierarchy, materials, and authored asset references.

The tools are built for large composed USD stages where the root stage may be mostly layout and the real mesh data may live inside instance prototypes. Each command traverses the normal stage plus `stage.GetPrototypes()` so instanceable component geometry is included.

## Install

Use `uv`:

```powershell
uv sync
```

Install optional Numba acceleration for expensive face checks:

```powershell
uv sync --extra numba
```

Run commands with:

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --json-out usd_geometry_audit.json
```

For one-off usage from this checkout without syncing first:

```powershell
uv --native-tls run --with-editable . usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda"
```

## Example Assets

The [OpenChessSet](https://github.com/usd-wg/assets/tree/main/full_assets/OpenChessSet) asset in the USD Working Group sample asset repository is a useful real-world test scene for `usd-scene-audit`. It includes composed USD layers, referenced geometry, and MaterialX look files.

```powershell
uv run usd-scene-audit "F:\path\to\OpenChessSet\chess_set.usda" --json-out usd_scene_audit.json
```

By default, naming audits report universal USD name hygiene and hierarchy issues. Vendor or studio prefix-style naming checks are opt-in so generic sample assets such as OpenChessSet are not treated as naming-policy failures.

## Commands

### `usd-geometry-audit`

Audits every `UsdGeom.Mesh`, including render, collision, helper, and prototype meshes.

Checks include:

- point array presence, count, non-finite values, and huge coordinates
- face/index consistency, negative and out-of-range indices
- empty faces, 1/2-vertex faces, repeated vertices, and zero-area fan triangles
- authored normals and authored primvar length/index validity
- authored extent vs computed point bounds
- transform determinant warnings

Example:

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --json-out usd_geometry_audit.json
```

Mesh attributes are read at one time code per audit: an explicit `--frame`, else the stage's authored `startTimeCode`, else the earliest authored time sample. Deforming geometry such as simulation caches, cloth, and crowd agents is therefore audited against real point data rather than looking like it has no points at all.

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --frame 1001
```

Every report records both `requested_frame` and the `time_code` that was actually evaluated. See [docs/geometry-audit.md](docs/geometry-audit.md#time-samples) for why a single concrete time code is used rather than resolving each attribute at its own earliest sample.

Choose a performance/coverage mode:

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --audit-mode fast
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --audit-mode standard
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --audit-mode exhaustive
```

- `fast`: topology, index, point, bounds, and transform checks only.
- `standard`: deep checks on render/helper meshes; skips exact zero-area scans on collision-like meshes.
- `exhaustive`: all exact checks, matching the original audit behavior.

Use Numba explicitly for expensive per-face checks:

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --geometry-engine numba
```

The default `--geometry-engine auto` uses Numba when the optional extra is installed and falls back to NumPy otherwise. On one optimized large-scene audit, Numba kept the findings identical while reducing exact audit time from about 68 seconds to about 44 seconds.

For repeated mesh-array experiments, enable the face cache:

```powershell
uv run usd-geometry-audit "F:\path\to\OpenChessSet\chess_set.usda" --mesh-cache face-hash
```

The cache hashes mesh arrays before reusing face-check results, so benchmark it on the target asset before keeping it enabled.

JSON reports include `phase_timings` and `mesh_cache` stats so slow assets can be tuned with evidence instead of guesses.

### `usd-names-hierarchy-audit`

Audits naming and hierarchy oddities while normalizing OpenUSD-generated prototype roots in examples.

Example:

```powershell
uv run usd-names-hierarchy-audit "F:\path\to\OpenChessSet\chess_set.usda" --json-out usd_names_hierarchy_audit.json
```

Enable prefix-style naming policy checks with a custom regex:

```powershell
uv run usd-names-hierarchy-audit "F:\path\to\OpenChessSet\chess_set.usda" --prefix-style-pattern "USD_[A-Z]+_[A-Za-z0-9]+" --json-out usd_names_hierarchy_audit.json
```

### `usd-scene-audit`

Audits high-level scene composition, material bindings, material references, and broad naming counts. Prefix-style naming checks are disabled by default and the active naming policy is recorded in the JSON report.

Authored asset references are resolved through USD's asset resolution layer (`Ar`), so package-relative paths into `.usdz` archives and paths served by a custom resolver are judged correctly. Each authored reference is classified as:

- `resolved_asset_count`: the resolver found the asset.
- `missing_authored_asset_count`: the asset does not resolve, listed in `missing_authored_assets`.
- `unverifiable_asset_count`: the path names a family of files rather than one file, such as a `<UDIM>` texture set or a `<f4>` frame sequence, and no tile probe resolved. Listed in `unverifiable_assets` and never counted as missing, because tile and frame numbering are asset-specific.

Example:

```powershell
uv run usd-scene-audit "F:\path\to\OpenChessSet\chess_set.usda" --json-out usd_scene_audit.json
```

Enable prefix-style naming policy checks with a custom regex:

```powershell
uv run usd-scene-audit "F:\path\to\OpenChessSet\chess_set.usda" --prefix-style-pattern "USD_[A-Z]+_[A-Za-z0-9]+" --json-out usd_scene_audit.json
```

## Notes

- Missing material files may still be printed by OpenUSD while opening a stage. Redirect stderr if those references are intentionally absent.
- Generated `*_audit.json`, `*_audit_stdout.log`, and `*_audit_stderr.log` files are ignored by git.
- The geometry audit can take several minutes on very large scenes because it checks all mesh indices and all fan-triangulated face areas. The optional `numba` extra keeps the same report shape, but benchmark it on your asset before making it the default.
