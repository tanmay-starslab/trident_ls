import numpy as np

from trident.meshless_sightlines import (
    generate_radial_sightlines,
    generate_random_parallel_sightlines,
    generate_uniform_grid_sightlines,
    trace_meshless_sightline_batch,
)
from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer


def _random_points_and_rays(seed=81):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(32, 3))
    starts = rng.uniform(-1.1, 1.1, size=(8, 3))
    ends = rng.uniform(-1.1, 1.1, size=(8, 3))
    too_short = np.linalg.norm(ends - starts, axis=1) < 0.4
    ends[too_short] += np.array([0.5, 0.1, -0.2])
    return points, starts, ends


def test_batch_trace_list_equals_single_trace():
    points, starts, ends = _random_points_and_rays()
    tracer = MeshlessVoronoiRayTracer(points)
    singles = [tracer.trace_ray(start, end_position=end) for start, end in zip(starts, ends)]
    batch = tracer.trace_rays(starts, end_positions=ends, return_format="list")

    assert len(batch) == len(singles)
    for one, many in zip(singles, batch):
        np.testing.assert_array_equal(many.indices, one.indices)
        np.testing.assert_allclose(many.dl, one.dl)
        assert many.failed_stack_recoveries == one.failed_stack_recoveries
        assert many.fallback_count == one.fallback_count
        assert many.nudge_count == one.nudge_count


def test_batch_trace_ragged_roundtrips_to_single_rays():
    points, starts, ends = _random_points_and_rays(seed=82)
    tracer = MeshlessVoronoiRayTracer(points)
    singles = [tracer.trace_ray(start, end_position=end) for start, end in zip(starts, ends)]
    ragged = tracer.trace_rays(starts, end_positions=ends, return_format="ragged")

    assert ragged.n_rays == len(singles)
    assert ragged.ray_offsets[0] == 0
    assert ragged.ray_offsets[-1] == len(ragged.indices)
    for ray_id, one in enumerate(singles):
        many = ragged.ray(ray_id)
        np.testing.assert_array_equal(many.indices, one.indices)
        np.testing.assert_allclose(many.dl, one.dl)


def test_threaded_batch_equals_serial_batch():
    points, starts, ends = _random_points_and_rays(seed=83)
    tracer = MeshlessVoronoiRayTracer(points)
    serial = tracer.trace_rays(
        starts, end_positions=ends, parallel="none", return_format="ragged"
    )
    threaded = tracer.trace_rays(
        starts, end_positions=ends, parallel="threads", n_jobs=2, return_format="ragged"
    )

    np.testing.assert_array_equal(threaded.ray_offsets, serial.ray_offsets)
    np.testing.assert_array_equal(threaded.indices, serial.indices)
    np.testing.assert_allclose(threaded.dl, serial.dl)


def test_trace_meshless_sightline_batch_helper():
    points, starts, ends = _random_points_and_rays(seed=84)
    tracer = MeshlessVoronoiRayTracer(points)
    batch = trace_meshless_sightline_batch(
        tracer, starts, ends=ends, return_format="ragged", chunksize=3
    )
    assert batch.n_rays == len(starts)
    np.testing.assert_allclose(
        [batch.ray(i).dl.sum() for i in range(batch.n_rays)],
        np.linalg.norm(ends - starts, axis=1),
    )


def test_uniform_grid_sightline_generation():
    sightlines = generate_uniform_grid_sightlines(
        center=[0.0, 0.0, 0.0],
        width=2.0,
        height=4.0,
        nx=3,
        ny=2,
        plane="xy",
        length=10.0,
    )

    assert sightlines.starts.shape == (6, 3)
    assert sightlines.ends.shape == (6, 3)
    np.testing.assert_allclose(sightlines.directions, np.tile([0.0, 0.0, 1.0], (6, 1)))
    np.testing.assert_allclose(sightlines.lengths, np.full(6, 10.0))
    np.testing.assert_array_equal(sightlines.metadata["grid_i"], [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(sightlines.metadata["grid_j"], [0, 0, 0, 1, 1, 1])


def test_radial_sightline_generation_is_area_uniform_and_seeded():
    seed = 398784
    nrays = 5
    sightlines = generate_radial_sightlines(
        center=[1.0, 2.0, 3.0],
        radius=10.0,
        r_min=2.0,
        nrays=nrays,
        plane="xy",
        length=20.0,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    u = rng.random(nrays)
    expected_r = np.sqrt(u * (10.0**2 - 2.0**2) + 2.0**2)

    np.testing.assert_allclose(sightlines.metadata["impact_parameter"], expected_r)
    np.testing.assert_allclose(sightlines.lengths, np.full(nrays, 20.0))
    assert np.all(sightlines.metadata["impact_parameter"] >= 2.0)
    assert np.all(sightlines.metadata["impact_parameter"] <= 10.0)


def test_random_parallel_sightline_generation_is_deterministic():
    kwargs = dict(
        center=[0.0, 0.0, 0.0],
        width=3.0,
        height=4.0,
        nrays=4,
        normal_vector=[0.0, 0.0, 1.0],
        length=5.0,
        seed=123,
    )
    one = generate_random_parallel_sightlines(**kwargs)
    two = generate_random_parallel_sightlines(**kwargs)

    np.testing.assert_allclose(one.starts, two.starts)
    np.testing.assert_allclose(one.ends, two.ends)
    np.testing.assert_allclose(one.lengths, np.full(4, 5.0))
    assert np.all(np.abs(one.metadata["projected_x"]) <= 1.5)
    assert np.all(np.abs(one.metadata["projected_y"]) <= 2.0)
