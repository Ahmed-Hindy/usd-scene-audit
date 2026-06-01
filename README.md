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
uv run usd-geometry-audit "F:\path\to\scene.usd" --json-out usd_geometry_audit.json
```

For one-off usage from this checkout without syncing first:

```powershell
uv --native-tls run --with-editable . usd-geometry-audit "F:\path\to\scene.usd"
```

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
uv run usd-geometry-audit "F:\path\to\scene.usd" --json-out usd_geometry_audit.json
```

Choose a performance/coverage mode:

```powershell
uv run usd-geometry-audit "F:\path\to\scene.usd" --audit-mode fast
uv run usd-geometry-audit "F:\path\to\scene.usd" --audit-mode standard
uv run usd-geometry-audit "F:\path\to\scene.usd" --audit-mode exhaustive
```

- `fast`: topology, index, point, bounds, and transform checks only.
- `standard`: deep checks on render/helper meshes; skips exact zero-area scans on collision-like meshes.
- `exhaustive`: all exact checks, matching the original audit behavior.

Use Numba explicitly for expensive per-face checks:

```powershell
uv run usd-geometry-audit "F:\path\to\scene.usd" --geometry-engine numba
```

The default `--geometry-engine auto` uses Numba when the optional extra is installed and falls back to NumPy otherwise. On one optimized large-scene audit, Numba kept the findings identical while reducing exact audit time from about 68 seconds to about 44 seconds.

For repeated mesh-array experiments, enable the face cache:

```powershell
uv run usd-geometry-audit "F:\path\to\scene.usd" --mesh-cache face-hash
```

The cache hashes mesh arrays before reusing face-check results, so benchmark it on the target asset before keeping it enabled.

JSON reports include `phase_timings` and `mesh_cache` stats so slow assets can be tuned with evidence instead of guesses.

### `usd-names-hierarchy-audit`

Audits naming and hierarchy oddities while normalizing OpenUSD-generated prototype roots in examples.

Example:

```powershell
uv run usd-names-hierarchy-audit "F:\path\to\scene.usd" --json-out usd_names_hierarchy_audit.json
```

### `usd-scene-audit`

Audits high-level scene composition, material bindings, material references, and broad naming counts.

Example:

```powershell
uv run usd-scene-audit "F:\path\to\scene.usd" --json-out usd_scene_audit.json
```

## Notes

- Missing material files may still be printed by OpenUSD while opening a stage. Redirect stderr if those references are intentionally absent.
- Generated `*_audit.json`, `*_audit_stdout.log`, and `*_audit_stderr.log` files are ignored by git.
- The geometry audit can take several minutes on very large scenes because it checks all mesh indices and all fan-triangulated face areas. The optional `numba` extra keeps the same report shape, but benchmark it on your asset before making it the default.
