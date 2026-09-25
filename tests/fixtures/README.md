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
| `animated_preroll_extent.usda` | Points carry a pre-roll sample while `extent` covers only the shot range; guards against comparing two attributes at two different times |

| `assets_udim_texture.usda` | Texture authored as a `<UDIM>` pattern, with tiles present in `tex/` |

## Stages with genuine defects

| Fixture | Expected finding |
|---|---|
| `static_mesh_out_of_range_indices.usda` | `out_of_range_face_vertex_indices` |
| `static_mesh_missing_points.usda` | `missing_points` |
| `assets_missing_texture.usda` | One missing authored asset (`./tex/absent.exr`) |

## Supporting files

`tex/` holds placeholder texture files for the asset-resolution fixtures. They are
not real EXRs — asset auditing only asks the resolver whether a path resolves, so
the contents are irrelevant and small text files keep the repository light.

Fixtures needing binary artifacts, such as a `.usdz` package, are built at test
runtime into `tmp_path` rather than committed.

## Stages pinning down known gaps

| Fixture | Status |
|---|---|
| `animated_topology_change.usda` | Point count changes mid-sequence and points go non-finite at frame 5. Each defect is reachable by auditing the frame it occurs on; only establishing that the count *changed* needs a sampling pass. The fixture asserts no *false* findings at the default time code and is the regression target for time-sampled checks. |

## Stages pinning a fix positively

Some fixes cannot be pinned by a zero-findings assertion, because the pre-fix code fails *silently* rather than loudly. At default time an unreadable `normals` or `extent` resolves to `None` and its check exits quietly, so a clean fixture passes whether or not the read was fixed. These stages carry a real defect that can only be seen once the read resolves real data.

| Fixture | Expected finding |
|---|---|
| `animated_mesh_authoring_defects.usda` | `normals_length_mismatch` and `authored_extent_mismatch` in time-sampled data |

## Adding a fixture

Keep stages minimal — one defect, as few prims and points as express it. Author `metersPerUnit` and `upAxis` unless the fixture is specifically about missing stage metadata, so unrelated checks stay quiet. Resolve fixtures through the `stage_path` fixture in `conftest.py` rather than building paths in tests.
