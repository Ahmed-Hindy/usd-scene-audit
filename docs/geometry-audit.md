# Geometry Audit Details

`usd-geometry-audit` is intended for large USD stages where correctness matters more than a fast shallow scan.

## Mesh Inclusion

The command scans:

- `stage.Traverse()`
- every prim under `stage.GetPrototypes()`

This is important for instanceable USD assets because ordinary traversal sees only the instance prims, not the prototype mesh contents.

## Mesh Categories

Every mesh is included. Categories only make the report easier to read:

- `render_like`: default category
- `collision_like`: names containing `_COL`, ending in `_C` or `_CO`, or containing `collision`
- `unknown_helper_like`: helper/proxy/guide path tokens

## Output

The JSON report contains:

- `summary_counts`: total issue counts
- `category_counts`: mesh counts by category
- `category_issue_counts`: issue counts by category
- `worst_meshes`: issue-ranked mesh records
- `largest_meshes_by_points`
- `largest_meshes_by_faces`
- `largest_meshes_by_extent`
- bounded examples for each issue type

## Default Thresholds

- zero-area triangle area: `1e-12`
- huge coordinate warning: absolute component over `1e6`
- authored extent mismatch: greater than `1e-4`

Override them with:

```powershell
uv run usd-geometry-audit scene.usd --zero-area-epsilon 1e-10 --huge-coord-threshold 100000 --extent-tolerance 1e-3
```
