# Meshless Voronoi Algorithm Exactness Audit

Date: 2026-05-20

## Scope

This audit compares the production geometry implementation against the SALSA
Appendix A Algorithm 1 and Algorithm 2 pseudocode supplied in the task prompt.
The animation is checked for scientific consistency, but it is not treated as
the authoritative implementation.

## Source Files Inspected

- `trident/meshless_voronoi_ray.py`
- `trident/ray_generator.py`
- `trident/meshless_ray_io.py`
- `trident/__init__.py`
- `tests/test_meshless_voronoi_ray.py`
- `tests/test_meshless_ray_generator_minimal.py`
- `tests/test_meshless_voronoi_algorithm_exactness.py`
- `dev/meshless_tests/meshless_voronoi_animation.py`

## Functions And Classes Inspected

- `MeshlessVoronoiRay`
- `MeshlessVoronoiRay.metadata`
- `MeshlessVoronoiRayTracer.__init__`
- `MeshlessVoronoiRayTracer.nearest_index`
- `MeshlessVoronoiRayTracer.trace_ray`
- `MeshlessVoronoiRayTracer._next_cell_crossing`
- `MeshlessVoronoiRayTracer._intersect_face_plane`
- `MeshlessVoronoiRayTracer._wrap_point`
- `MeshlessVoronoiRayTracer._minimum_image`
- `MeshlessVoronoiRayTracer._merge_segments`
- `meshless_voronoi_ray`
- `make_meshless_voronoi_ray`
- `write_meshless_ray_hdf5`
- `load_meshless_ray`
- `inspect_existing_trident_ray_schema`
- `ray_face_intersection` and `compute_meshless_path` in the animation script

## Classification

**Mathematically equivalent with modifications** for ordinary non-degenerate
Voronoi crossings tested here. It is **not an exact line-by-line equivalent** of
SALSA Algorithm 1 because it adds periodic support, tolerance-based branch
guards, zero-length nudges, a conservative fallback search, and duplicate
segment merging. These modifications can change behavior near faces, vertices,
parallel candidate planes, and numerical failures.

The implementation is not merely brute-force SALSA-inspired geometry: the
production path uses the SALSA tree query, bisection search, failed-candidate
stack, candidate face-plane intersection, and verification query structure.

## Algorithm 1 Checklist

| Step | Status | Location | Explanation |
| --- | --- | --- | --- |
| Inputs `T`, `x_i`, `r_0`, `r_hat`, `S` | PASS | `trident/meshless_voronoi_ray.py:132-171`, `178-218` | `cKDTree` is constructed over positions; `trace_ray` accepts start/end or start/direction/length and normalizes `r_hat`. |
| Output ordered `I`, `delta_x` | PASS with extra metadata | `trident/meshless_voronoi_ray.py:281-296` | Returns `indices` and `dl` in `MeshlessVoronoiRay`, plus diagnostics. |
| Initialize `dl = 0`, `r = r_0`, `r_f = r_0 + S r_hat` | PASS with naming differences | `trident/meshless_voronoi_ray.py:220-227` | Uses `travelled` for `dl`; wraps `r` and `r_f` when periodic. |
| Initialize `I_cur = tree_nearest(r_0)`, `I_f = tree_nearest(r_f)` | PASS | `trident/meshless_voronoi_ray.py:222-223` | Uses `nearest_index`, backed by `cKDTree.query`. |
| Initialize failed-candidate stack/list | PASS | `trident/meshless_voronoi_ray.py:228` | Uses `failed_stack: list[Tuple[int, float]]`. |
| Main loop while ray crosses cells | PASS with safety bound | `trident/meshless_voronoi_ray.py:233-271` | Uses a bounded `for` loop with `max_iter` rather than an unbounded while loop. |
| Per-cell setup `x_cur`, `I_end = -1`, `L = 0`, `R = S - dl`, `dl_local = infinity` | PASS | `trident/meshless_voronoi_ray.py:307-322` | Implemented in `_next_cell_crossing`. |
| Failed stack removes entries whose index equals current cell | PASS | `trident/meshless_voronoi_ray.py:313-315` | Filters out `idx == i_cur`. |
| Failed stack uses closest failed distance and seeds `R`, `I_end` | INTENTIONAL MODIFICATION | `trident/meshless_voronoi_ray.py:316-320` | Uses closest failed distance and candidate, but clamps `R` to the remaining ray length and at least `eps`. Paper pseudocode sets `R = 2 * failed_distance` directly. |
| Inner bisection computes `l_cen`, `r_end` | PASS | `trident/meshless_voronoi_ray.py:323-325` | Matches the paper. |
| Re-query `I_end` if not first iteration or no failed candidate | PASS | `trident/meshless_voronoi_ray.py:327-328` | Equivalent condition: `n > 0 or i_end < 0`. |
| If `I_end == I_cur`, move lower bound or expand upper bound | INTENTIONAL MODIFICATION | `trident/meshless_voronoi_ray.py:330-337` | Same structure, but expansion is clamped to `remaining` and has an extra `L + 2 eps` guard. |
| Compute face-plane intersection for candidate neighbor | PASS | `trident/meshless_voronoi_ray.py:339-340` | Calls `_intersect_face_plane`. |
| Verify candidate with `r_candidate` and `tree_nearest` | PASS | `trident/meshless_voronoi_ray.py:345-346` | Matches the paper structure. |
| If another cell lies in front, push failed candidate and continue | PARTIAL | `trident/meshless_voronoi_ray.py:348-354` | Implements the stack push and search update for `l_search > L + eps`. If `l_search <= L + eps`, the code falls through toward acceptance; this is a numerical degeneracy guard not present in the paper. |
| Save current segment after accepted crossing | PASS with pre-save guard | `trident/meshless_voronoi_ray.py:250-265` | Saves `i_cur` and `step`; before that, zero/non-finite steps are handled by a nudge instead of being saved. |
| Advance `r`, `dl`, `I_cur` | PASS | `trident/meshless_voronoi_ray.py:263-265` | Updates ray point, cumulative distance, and current cell. |
| Terminate when `I_cur == I_f` and save final segment | PARTIAL | `trident/meshless_voronoi_ray.py:238-242`, `273-277` | Final-cell check happens at the start of the next loop iteration rather than immediately after assigning `I_cur = I_end`. This is equivalent in ordinary cases but not line-by-line identical. |
| Do not construct the full Voronoi mesh | PASS | `trident/meshless_voronoi_ray.py:9-13`, `171` | Uses only a nearest-neighbor tree and site positions. |
| Fallback behavior | INTENTIONAL MODIFICATION | `trident/meshless_voronoi_ray.py:363-374` | The paper pseudocode does not include the conservative first-ownership-change bisection fallback. |
| Duplicate segment handling | INTENTIONAL MODIFICATION | `trident/meshless_voronoi_ray.py:279-280`, `423-434` | Adjacent duplicate indices are merged and non-positive `dl` rows are dropped. This changes the returned representation even when total path length is preserved. |

## Algorithm 2 Checklist

| Step | Status | Location | Explanation |
| --- | --- | --- | --- |
| `q = x_end - x_cur` | PASS with periodic extension | `trident/meshless_voronoi_ray.py:390` | Uses `_minimum_image(x_end - x_cur)` so periodic rays use the nearest periodic image. |
| `m = x_cur + q/2` | PASS with periodic extension | `trident/meshless_voronoi_ray.py:391` | Computes midpoint and wraps it into the periodic box when enabled. |
| `c = m - r` | PASS with periodic extension | `trident/meshless_voronoi_ray.py:392` | Uses minimum-image displacement. |
| `c_q = dot(c, q)` | PASS | `trident/meshless_voronoi_ray.py:393` | Matches the paper. |
| `h_q = dot(r_hat, q)` | PASS | `trident/meshless_voronoi_ray.py:394` | Matches the paper. |
| If `h_q == 0`, return unchanged | INTENTIONAL MODIFICATION | `trident/meshless_voronoi_ray.py:396-397` | Uses `abs(hq) <= eps` rather than exact equality. |
| If `c_q > 0`, set `s = c_q / h_q` | PASS | `trident/meshless_voronoi_ray.py:399-400` | Matches the paper. |
| Else if `h_q > 0`, set `s = 0` | PASS | `trident/meshless_voronoi_ray.py:401-403` | Matches the paper. |
| Else set `s = infinity` | PASS | `trident/meshless_voronoi_ray.py:404-405` | Matches the paper. |
| If `0 <= s <= dl_local`, update `dl_local` | PASS | `trident/meshless_voronoi_ray.py:407-409` | Matches the paper. |

## Deviations From The Paper Algorithm

1. **Periodic wrapping and minimum-image geometry**
   - Location: `trident/meshless_voronoi_ray.py:152-164`, `198`, `220-221`, `390-392`, `411-420`
   - Effect: Changes ray paths when `box_size`/periodic mode is active. No effect when periodic is disabled.
   - Justification: Required extension for periodic simulation domains; the SALSA pseudocode omits periodic details.

2. **Tolerance-based branch tests**
   - Location: `trident/meshless_voronoi_ray.py:168`, `235`, `250`, `331-336`, `396`
   - Effect: Can change numerical behavior near faces, vertices, or nearly parallel candidate planes.
   - Justification: Prevents infinite loops and unstable equality tests.

3. **Search-range clamping**
   - Location: `trident/meshless_voronoi_ray.py:319`, `334-336`, `351-352`
   - Effect: Can alter search dynamics near the ray end. It should not change ordinary accepted crossings when the next face is inside the remaining interval.
   - Justification: Keeps searches inside the requested ray segment.

4. **Non-finite candidate handling**
   - Location: `trident/meshless_voronoi_ray.py:341-343`
   - Effect: Candidate planes that do not yield finite crossings are skipped. This is robust behavior but not explicit in Algorithm 1.
   - Justification: Avoids accepting invalid parallel or behind-ray candidates.

5. **Rejected-candidate guard on `l_search`**
   - Location: `trident/meshless_voronoi_ray.py:348-354`
   - Effect: If another cell is detected but `l_search <= L + eps`, the implementation can fall through instead of pushing and continuing. This can change cell sequence only in numerical degeneracy cases.
   - Justification: Avoids no-progress loops near a boundary.

6. **Zero-length/nudge behavior**
   - Location: `trident/meshless_voronoi_ray.py:250-258`
   - Effect: Can change the cell sequence and omit an infinitesimal path near a face or vertex.
   - Justification: Prevents zero-length loops.

7. **Fallback bisection**
   - Location: `trident/meshless_voronoi_ray.py:363-374`
   - Effect: Can change the accepted crossing if the main failed-stack search does not resolve a valid candidate. It is failure handling, not the paper algorithm.
   - Justification: Conservative recovery that preserves total path length and first ownership change.

8. **Duplicate segment merging and non-positive segment filtering**
   - Location: `trident/meshless_voronoi_ray.py:279-280`, `423-434`
   - Effect: Changes the returned path representation and segment count. Total path length through a cell is preserved for adjacent duplicates.
   - Justification: Removes artifacts caused by numerical nudges.

9. **Final-cell termination placement**
   - Location: `trident/meshless_voronoi_ray.py:238-242`
   - Effect: Equivalent for ordinary crossings, but not the same control-flow location as the paper, which checks immediately after advancing to `I_end`.
   - Justification: Simpler loop structure and supports rays starting and ending in the same cell.

## Periodic Boundary Handling

Periodic handling is an extension beyond the supplied pseudocode. It affects
nearest-neighbor queries through `cKDTree(boxsize=...)` and face geometry through
minimum-image displacements. The extension is mathematically necessary for
periodic domains, but it means a periodic trace is not the exact non-periodic
Algorithm 1/2 pseudocode.

## Fallback Bisection

The fallback is not in the paper algorithm. It only runs if the main bisection
and failed-stack search exceed `max_bisect_iter`. When invoked, it can change
the accepted crossing because it chooses the first point that no longer belongs
to the current cell, not the candidate produced by the failed-stack procedure.
The new tests assert zero fallback use in random non-degenerate reference
comparisons.

## Duplicate Segment Merging

Duplicate merging changes the output representation by combining adjacent rows
with the same cell index. This can reduce segment counts relative to a literal
append-only Algorithm 1 trace. It should preserve the integrated path length per
cell for adjacent duplicates.

## Zero-Length And Nudge Behavior

The nudge behavior is not in the paper algorithm. It can change the cell
sequence for rays starting exactly on a face or passing through a vertex. It is
best classified as a robustness modification for degenerate geometry, not an
exact SALSA step.

## Final-Cell Termination

The final-cell termination is implemented at the start of each outer loop. This
is equivalent after the previous iteration has advanced `I_cur = I_end`, but it
is not line-by-line identical to the supplied pseudocode. It also naturally
handles rays whose start and end points are in the same Voronoi cell.

## Animation Consistency Check

| Question | Status | Location | Explanation |
| --- | --- | --- | --- |
| Correct inputs `T`, `x_i`, `r0`, `r_hat`, `S` | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:778-785` | Shows the expected input list. |
| Correct outputs `I` and `delta_x` | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:706-729`, `1060-1065`, `1090-1093` | Shows ordered cell IDs and path lengths as the stored geometry output. |
| Full Voronoi mesh shown only for explanation | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:291-295`, `747-750`, `1282-1315` | The script states the production method does not build/store the full mesh and that the educational walk differs from real SALSA acceleration. |
| Correct `tree_nearest` description | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:813-829` | Shows `I_cur = tree_nearest(r,T)`. |
| Candidate face is perpendicular bisector | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:900-928` | Shows candidate face using `q`, midpoint, and a perpendicular line. |
| Correct Algorithm 2 variables and formula | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:259-286`, `950-985` | Defines `q`, `m`, `c`, `c_q`, `h_q`, and `s = c_q/h_q`. |
| Correct Algorithm 2 branch logic | PASS with tolerance | `dev/meshless_tests/meshless_voronoi_animation.py:274-286`, `985-992` | Uses a tolerance for `h_q == 0`, like production. |
| Correct candidate verification | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:1006-1025` | Shows `r_cand` and `I_cand = tree_nearest`. |
| Failed candidates stored and search continues | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:1031-1038` | States another cell in front is stored and the search continues. |
| Avoids claiming educational brute-force path is exact SALSA | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:289-295` | Explicitly calls it an educational walk and says real SALSA uses tree and bisection. |
| Distinguishes yt/Trident LightRay from SALSA-style meshless geometry | PASS | `dev/meshless_tests/meshless_voronoi_animation.py:1282-1315` | States same spectrum machinery but different ray-sampling assumptions. |

## New Test Coverage

`tests/test_meshless_voronoi_algorithm_exactness.py` adds:

- Direct Algorithm 2 branch tests for simple, angled, parallel, zero-distance,
  infinite, existing-smaller, and negative-candidate cases:
  `tests/test_meshless_voronoi_algorithm_exactness.py:143-231`.
- Analytic two-site, 1D lattice, offset lattice, and 2D square tests:
  `tests/test_meshless_voronoi_algorithm_exactness.py:234-304`.
- An independent all-face reference walker:
  `tests/test_meshless_voronoi_algorithm_exactness.py:42-128`.
- Random 2D and 3D non-degenerate comparisons to that reference:
  `tests/test_meshless_voronoi_algorithm_exactness.py:307-333`.
- Invariant checks:
  `tests/test_meshless_voronoi_algorithm_exactness.py:336-366`.
- Failed-stack exercise:
  `tests/test_meshless_voronoi_algorithm_exactness.py:369-383`.
- Periodic minimum-image test:
  `tests/test_meshless_voronoi_algorithm_exactness.py:386-397`.

## Test Result Used For Classification

The exactness tests passed in isolation:

```text
17 passed in 2.93s
```

These tests support mathematical equivalence on the controlled non-degenerate
cases tested. They do not prove line-by-line equivalence, and they intentionally
do not claim equivalence for degenerate near-face/near-vertex cases where the
implementation has explicit robustness behavior.

## Recommended Fixes Or Follow-Ups

No production fix is required to claim ordinary-case mathematical equivalence
based on the new tests. If exact paper control flow is required, the following
would need separate design decisions:

1. Add an optional `strict_salsa=True` mode that disables nudges, fallback
   bisection, segment merging, and search-range clamping.
2. Expose a public or semi-private Algorithm 2 helper if direct downstream tests
   should avoid calling `_intersect_face_plane`.
3. Add targeted degenerate-case tests documenting expected tie behavior at
   vertices and faces.
4. Record fallback/nudge events in debug output whenever they occur in real
   science rays so users can identify non-paper recovery behavior.
