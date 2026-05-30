# USD Asset Audit

Small command-line tools for auditing large composed USD scenes with `usd-core`.

The tools were built for heavy KitBash3D-style scenes where the root stage is mostly layout and the real mesh data lives inside instance prototypes. Each command traverses the normal stage plus `stage.GetPrototypes()` so instanceable component geometry is included.

## Install

Use `uv`:

```powershell
uv sync
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
uv run usd-geometry-audit "F:\path\to\kb3d_stadiums.usd" --json-out usd_geometry_audit.json
```

### `usd-names-hierarchy-audit`

Audits naming and hierarchy oddities while normalizing OpenUSD-generated prototype roots in examples.

Example:

```powershell
uv run usd-names-hierarchy-audit "F:\path\to\kb3d_stadiums.usd" --json-out usd_names_hierarchy_audit.json
```

### `usd-scene-audit`

Audits high-level scene composition, material bindings, material references, and broad naming counts.

Example:

```powershell
uv run usd-scene-audit "F:\path\to\kb3d_stadiums.usd" --json-out usd_scene_audit.json
```

## Notes

- Missing material files may still be printed by OpenUSD while opening a stage. Redirect stderr if those references are intentionally absent.
- Generated `*_audit.json`, `*_audit_stdout.log`, and `*_audit_stderr.log` files are ignored by git.
- The geometry audit can take several minutes on very large scenes because it checks all mesh indices and all fan-triangulated face areas.
