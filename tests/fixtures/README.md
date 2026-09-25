# Test Fixture Corpus

Small USD stages, each isolating one behaviour so a failing test names a specific defect class rather than "something changed".

Fixtures are committed as text `.usda` so they are reviewable in a diff and runnable by hand:

```bash
uv run usd-geometry-audit tests/fixtures/animated_points_valid.usda
```

Each file carries a docstring in its layer metadata stating what it represents and what the audit is expected to report.

## Stages that must report zero findings

These are the false-positive guards. A finding against any of them is a bug in the audit, not in the asset.

| Fixture | Covers |
|---|---|
| `static_mesh_clean.usda` | A wholly valid static mesh with matching extent, normals, and UVs |
| `animated_points_valid.usda` | Valid deforming mesh; `points` authored only as time samples |
| `animated_normals_primvars.usda` | Valid deforming mesh; points, normals, extent, and UVs all time-sampled only |

## Stages with genuine defects

| Fixture | Expected finding |
|---|---|
| `static_mesh_out_of_range_indices.usda` | `out_of_range_face_vertex_indices` |
| `static_mesh_missing_points.usda` | `missing_points` |

## Stages pinning down known gaps

| Fixture | Status |
|---|---|
| `animated_topology_change.usda` | Point count changes mid-sequence and points go non-finite at frame 5. Both are real defects invisible to a single-sample audit; the fixture asserts no *false* findings today and is the regression target for time-sampled checks. |

## Adding a fixture

Keep stages minimal — one defect, as few prims and points as express it. Author `metersPerUnit` and `upAxis` unless the fixture is specifically about missing stage metadata, so unrelated checks stay quiet. Resolve fixtures through the `stage_path` fixture in `conftest.py` rather than building paths in tests.
