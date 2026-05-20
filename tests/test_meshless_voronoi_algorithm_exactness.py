import numpy as np
import pytest

from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer, meshless_voronoi_ray


def _tracer(points, **kwargs):
    defaults = {"eps": 1.0e-11, "max_iter": 10000, "max_bisect_iter": 512}
    defaults.update(kwargs)
    return MeshlessVoronoiRayTracer(np.asarray(points, dtype=float), **defaults)


def _nearest_index(points, point):
    dist2 = np.sum((points - point) ** 2, axis=1)
    return int(np.argmin(dist2))


def _algorithm2(tracer, r, rhat, xcur, xend, dl_local=np.inf):
    return tracer._intersect_face_plane(
        np.asarray(r, dtype=float),
        np.asarray(rhat, dtype=float),
        np.asarray(xcur, dtype=float),
        np.asarray(xend, dtype=float),
        float(dl_local),
    )


def _merge_adjacent(indices, lengths):
    out_i = []
    out_dl = []
    for idx, dl in zip(indices, lengths):
        if dl <= 0:
            continue
        if out_i and out_i[-1] == idx:
            out_dl[-1] += dl
        else:
            out_i.append(int(idx))
            out_dl.append(float(dl))
    return np.asarray(out_i, dtype=np.int64), np.asarray(out_dl, dtype=float)


def _reference_all_faces_walk(points, start, end, eps=1.0e-10):
    """Independent reference walker that checks every bisector plane.

    This deliberately does not use the production tree/bisection/failed-stack
    search.  It finds the next ownership change by testing all candidate
    Voronoi bisectors from the current cell.
    """
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    total = float(np.linalg.norm(delta))
    if total <= 0:
        raise ValueError("reference ray length must be positive")
    rhat = delta / total

    r = start.copy()
    travelled = 0.0
    current = _nearest_index(points, r)
    final = _nearest_index(points, end)
    indices = []
    lengths = []
    probe = max(1.0e-8 * total, 100.0 * eps)

    for _ in range(10000):
        remaining = total - travelled
        if remaining <= eps:
            break
        if current == final:
            indices.append(current)
            lengths.append(remaining)
            travelled = total
            break

        xcur = points[current]
        best_s = np.inf
        best_next = None
        tied = False

        for j, xend in enumerate(points):
            if j == current:
                continue
            q = xend - xcur
            hq = float(np.dot(rhat, q))
            if abs(hq) <= eps:
                continue
            midpoint = xcur + 0.5 * q
            cq = float(np.dot(midpoint - r, q))
            if cq > 0:
                s = cq / hq
            elif hq > 0:
                s = 0.0
            else:
                s = np.inf
            if not (eps < s <= remaining + eps):
                continue

            after_distance = min(remaining, s + probe)
            next_idx = _nearest_index(points, r + rhat * after_distance)
            if next_idx == current:
                continue
            if abs(s - best_s) <= max(1.0e-8 * total, 100.0 * eps):
                tied = True
            if s < best_s:
                best_s = float(s)
                best_next = next_idx

        if tied:
            raise pytest.SkipTest("degenerate equal-distance candidate crossing")
        if best_next is None:
            raise AssertionError("reference walker could not find next crossing")

        best_s = min(best_s, remaining)
        indices.append(current)
        lengths.append(best_s)
        travelled += best_s
        r = start + rhat * travelled
        current = int(best_next)

    else:
        raise AssertionError("reference walker exceeded iteration limit")

    if travelled < total - eps:
        indices.append(current)
        lengths.append(total - travelled)

    return _merge_adjacent(indices, lengths)


def _assert_matches_reference(points, start, end, atol=1.0e-8):
    tracer = _tracer(points)
    ray = tracer.trace_ray(start_position=np.asarray(start), end_position=np.asarray(end))
    ref_i, ref_dl = _reference_all_faces_walk(points, start, end)

    np.testing.assert_array_equal(ray.indices, ref_i)
    np.testing.assert_allclose(ray.dl, ref_dl, rtol=0.0, atol=atol)
    np.testing.assert_allclose(ray.dl.sum(), ray.length, rtol=0.0, atol=atol)
    assert ray.fallback_count == 0
    assert ray.nudge_count == 0


def test_algorithm2_simple_two_site_x_axis_face():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[-1.0, 0.0, 0.0],
        rhat=[1.0, 0.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
    )
    assert s == pytest.approx(2.0)


def test_algorithm2_angled_ray_uses_projected_distance():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rhat = np.array([2.0, -1.0, 0.0])
    rhat = rhat / np.linalg.norm(rhat)
    r = np.array([-1.0, 1.0, 0.0])
    xcur = np.array([0.0, 0.0, 0.0])
    xend = np.array([2.0, 0.0, 0.0])
    q = xend - xcur
    midpoint = xcur + 0.5 * q
    expected = np.dot(midpoint - r, q) / np.dot(rhat, q)

    s = _algorithm2(tracer, r, rhat, xcur, xend)
    assert s == pytest.approx(expected)


def test_algorithm2_parallel_ray_leaves_dl_local_unchanged():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[-1.0, 0.0, 0.0],
        rhat=[0.0, 1.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
        dl_local=7.0,
    )
    assert s == pytest.approx(7.0)


def test_algorithm2_behind_face_but_forward_normal_returns_zero():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[1.5, 0.0, 0.0],
        rhat=[1.0, 0.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
    )
    assert s == pytest.approx(0.0)


def test_algorithm2_behind_face_and_backward_normal_leaves_dl_local():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[1.5, 0.0, 0.0],
        rhat=[-1.0, 0.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
        dl_local=4.0,
    )
    assert s == pytest.approx(4.0)


def test_algorithm2_existing_smaller_candidate_is_preserved():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[-1.0, 0.0, 0.0],
        rhat=[1.0, 0.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
        dl_local=1.0,
    )
    assert s == pytest.approx(1.0)


def test_algorithm2_negative_candidate_does_not_update_dl_local():
    tracer = _tracer([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    s = _algorithm2(
        tracer,
        r=[-1.0, 0.0, 0.0],
        rhat=[-1.0, 0.0, 0.0],
        xcur=[0.0, 0.0, 0.0],
        xend=[2.0, 0.0, 0.0],
        dl_local=5.0,
    )
    assert s == pytest.approx(5.0)


def test_two_site_full_ray_matches_midpoint_boundary():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-1.0, 0.0, 0.0]),
        end_position=np.array([3.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(ray.indices, [0, 1])
    np.testing.assert_allclose(ray.dl, [2.0, 2.0], rtol=0.0, atol=1.0e-10)


def test_regular_1d_lattice_matches_midpoint_boundaries():
    positions = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-0.5, 0.0, 0.0]),
        end_position=np.array([4.5, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(ray.indices, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(ray.dl, np.ones(5), rtol=0.0, atol=1.0e-10)


def test_offset_1d_lattice_matches_analytic_boundaries():
    positions = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([0.25, 0.0, 0.0]),
        end_position=np.array([3.25, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(ray.indices, [0, 1, 2, 3])
    np.testing.assert_allclose(ray.dl, [0.25, 1.0, 1.0, 0.75], rtol=0.0, atol=1.0e-10)


def test_2d_square_horizontal_and_vertical_rays_match_analytic_bisectors():
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
    )
    horizontal = meshless_voronoi_ray(
        positions,
        start_position=np.array([-1.0, 0.0, 0.0]),
        end_position=np.array([3.0, 0.0, 0.0]),
    )
    vertical = meshless_voronoi_ray(
        positions,
        start_position=np.array([0.0, -1.0, 0.0]),
        end_position=np.array([0.0, 3.0, 0.0]),
    )

    np.testing.assert_array_equal(horizontal.indices, [0, 1])
    np.testing.assert_allclose(horizontal.dl, [2.0, 2.0], rtol=0.0, atol=1.0e-10)
    np.testing.assert_array_equal(vertical.indices, [0, 2])
    np.testing.assert_allclose(vertical.dl, [2.0, 2.0], rtol=0.0, atol=1.0e-10)


def test_2d_square_diagonal_vertex_case_is_deterministic_and_conservative():
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
    )
    kwargs = {
        "start_position": np.array([-1.0, -1.0, 0.0]),
        "end_position": np.array([3.0, 3.0, 0.0]),
    }
    ray1 = meshless_voronoi_ray(positions, **kwargs)
    ray2 = meshless_voronoi_ray(positions, **kwargs)

    np.testing.assert_array_equal(ray1.indices, ray2.indices)
    np.testing.assert_allclose(ray1.dl, ray2.dl, rtol=0.0, atol=1.0e-10)
    assert ray1.indices[0] == 0
    assert ray1.indices[-1] == 3
    assert np.all(ray1.dl > 0)
    np.testing.assert_allclose(ray1.dl.sum(), ray1.length, rtol=0.0, atol=1.0e-10)


def test_random_2d_non_degenerate_rays_match_all_faces_reference():
    rng = np.random.default_rng(91231)
    checked = 0
    for _ in range(20):
        points = rng.uniform(-1.0, 1.0, size=(18, 3))
        points[:, 2] = 0.0
        start = np.array([*rng.uniform(-1.15, 1.15, size=2), 0.0])
        end = np.array([*rng.uniform(-1.15, 1.15, size=2), 0.0])
        if np.linalg.norm(end - start) < 0.4:
            continue
        _assert_matches_reference(points, start, end, atol=5.0e-8)
        checked += 1
    assert checked >= 8


def test_random_3d_non_degenerate_rays_match_all_faces_reference():
    rng = np.random.default_rng(31029)
    checked = 0
    for _ in range(18):
        points = rng.uniform(-1.0, 1.0, size=(24, 3))
        start = rng.uniform(-1.15, 1.15, size=3)
        end = rng.uniform(-1.15, 1.15, size=3)
        if np.linalg.norm(end - start) < 0.5:
            continue
        _assert_matches_reference(points, start, end, atol=5.0e-8)
        checked += 1
    assert checked >= 8


def test_random_ray_invariants_for_non_degenerate_cases():
    rng = np.random.default_rng(4477)
    points = rng.uniform(-1.0, 1.0, size=(30, 3))
    tracer = _tracer(points)

    for _ in range(12):
        start = rng.uniform(-1.15, 1.15, size=3)
        end = rng.uniform(-1.15, 1.15, size=3)
        if np.linalg.norm(end - start) < 0.5:
            continue
        ray = tracer.trace_ray(start_position=start, end_position=end)
        assert np.all(ray.indices >= 0)
        assert np.all(ray.indices < len(points))
        assert np.all(ray.dl > 0.0)
        assert len(ray.indices) == len(ray.dl)
        assert len(set(zip(ray.indices[:-1], ray.indices[1:]))) == len(ray.indices) - 1
        assert not np.any(ray.indices[:-1] == ray.indices[1:])
        np.testing.assert_allclose(ray.dl.sum(), ray.length, rtol=0.0, atol=5.0e-8)
        assert ray.indices[0] == tracer.nearest_index(start)
        assert ray.indices[-1] == tracer.nearest_index(end)

        direction = (end - start) / np.linalg.norm(end - start)
        lower = 0.0
        for idx, dl in zip(ray.indices, ray.dl):
            midpoint = start + direction * (lower + 0.5 * dl)
            assert tracer.nearest_index(midpoint) == idx
            lower += dl
        cumulative = np.cumsum(ray.dl)
        for next_idx, boundary in zip(ray.indices[1:], cumulative[:-1]):
            probe = min(boundary + max(1.0e-8 * ray.length, 100.0 * tracer.eps), ray.length)
            assert tracer.nearest_index(start + direction * probe) == next_idx


def test_failed_candidate_stack_behavior_is_exercised():
    rng = np.random.default_rng(12345)
    for _ in range(50):
        points = rng.uniform(-1.0, 1.0, size=(35, 3))
        points[:, 2] = 0.0
        start = np.array([*rng.uniform(-1.2, 1.2, size=2), 0.0])
        end = np.array([*rng.uniform(-1.2, 1.2, size=2), 0.0])
        if np.linalg.norm(end - start) < 0.5:
            continue
        ray = _tracer(points).trace_ray(start_position=start, end_position=end)
        if ray.failed_stack_recoveries > 0:
            assert ray.dl.sum() == pytest.approx(ray.length)
            assert len(ray.indices) >= 2
            return
    pytest.fail("expected at least one deterministic random case to use the failed stack")


def test_periodic_two_site_boundary_uses_minimum_image():
    positions = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    tracer = _tracer(positions, box_size=10.0)
    ray = tracer.trace_ray(
        start_position=np.array([9.5, 0.0, 0.0]),
        end_position=np.array([1.5, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(ray.indices, [1, 0])
    np.testing.assert_allclose(ray.dl, [0.5, 1.5], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(ray.dl.sum(), ray.length, rtol=0.0, atol=1.0e-10)
    assert ray.periodic
