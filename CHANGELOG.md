# Changelog

## Unreleased

### Added

- Geometry: `points_wrong_type`, `face_vertex_counts_wrong_type`, `face_vertex_indices_wrong_type`, `normals_wrong_type`, and `extent_wrong_type` findings for attributes authored with a value type the checks cannot use. They replace crashes and silently truncated indices ([#48](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/48)).
- Geometry: `unaudited_mesh_count`. A check that raises now costs only its own findings and is reported in `check_errors`; the rest of the mesh's record is kept.
- Scene: exact `naming.case_collision_count`, `naming.duplicate_sibling_count`, `materials.direct_binding_targets_missing_count`, and `materials.direct_binding_targets_not_material_count` next to their capped example lists ([#47](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/47)).
- Python API: `geometry.MeshCheckSettings` and the `DEFAULT_*` constants shared by `analyze()` and the CLI.

### Changed

- Reports are identical across runs. Prims inside instancing prototypes are reported at the path of their first instance instead of `/__Prototype_N/...`, whose numbering changes on every open ([#49](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/49)).
- Scene: a valid `material:binding:collection:*` relationship no longer reports its collection as a missing binding target.
- `names_hierarchy`: case-collision and duplicate-sibling counts are no longer capped at 80 ([#37](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/37)).
- `MeshCheckSettings` rejects negative or non-finite thresholds, and `analyze()` validates them before opening the stage.
- Requesting the Numba engine without Numba installed raises instead of silently reporting no repeated-vertex or zero-area findings.
- Geometry: missing, empty, or unusable `points`, `faceVertexCounts`, or `faceVertexIndices` are reported once. Checks that compare against them are skipped instead of reporting every index as out of range and every primvar as the wrong length ([#54](https://github.com/Ahmed-Hindy/usd-scene-audit/issues/54)).

### Deprecated

- Passing options positionally to `geometry.analyze()`, `geometry.mesh_record()`, or `geometry.analyze_face_geometry()`. These calls still work and emit a `DeprecationWarning`. Pass options by keyword; for `mesh_record()`, group thresholds with `settings=MeshCheckSettings(...)`. Positional options will stop working in the next major release.
