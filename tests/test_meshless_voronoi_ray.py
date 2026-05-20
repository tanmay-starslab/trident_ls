import numpy as np
import pytest

from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer, meshless_voronoi_ray


def test_two_site_ray_crosses_midplane():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-1.0, 0.0, 0.0]),
        end_position=np.array([3.0, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(ray.indices, np.array([0, 1]))
    np.testing.assert_allclose(ray.dl, np.array([2.0, 2.0]), rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(ray.dl.sum(), ray.length)


def test_regular_1d_lattice_path_lengths():
    positions = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    tracer = MeshlessVoronoiRayTracer(positions)
    ray = tracer.trace_ray(
        start_position=np.array([-0.5, 0.0, 0.0]),
        end_position=np.array([4.5, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(ray.indices, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_allclose(ray.dl, np.ones(5), rtol=0.0, atol=1.0e-10)


def test_direction_and_length_interface():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-1.0, 0.0, 0.0]),
        direction=np.array([10.0, 0.0, 0.0]),
        length=4.0,
    )

    np.testing.assert_array_equal(ray.indices, np.array([0, 1]))
    np.testing.assert_allclose(ray.dl, np.array([2.0, 2.0]), rtol=0.0, atol=1.0e-10)


def test_duplicate_adjacent_segments_are_merged():
    indices, dl = MeshlessVoronoiRayTracer._merge_segments(
        [1, 1, 2, 2, 2, 3], [0.25, 0.75, 1.0, 0.5, 0.5, 2.0]
    )
    assert indices == [1, 2, 3]
    np.testing.assert_allclose(dl, [1.0, 2.0, 2.0])


def test_diagonal_ray_through_simple_cubic_generators():
    positions = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-0.25, -0.25, -0.25]),
        end_position=np.array([1.25, 1.25, 1.25]),
    )

    assert ray.indices[0] == 0
    assert ray.indices[-1] == 7
    assert np.all(ray.dl > 0)
    np.testing.assert_allclose(ray.dl.sum(), ray.length)


def test_periodic_boundary_crossing_uses_minimum_image():
    positions = np.array([[0.25, 0.0, 0.0], [9.25, 0.0, 0.0]])
    tracer = MeshlessVoronoiRayTracer(positions, box_size=10.0)
    ray = tracer.trace_ray(
        start_position=np.array([9.6, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=0.8,
    )

    np.testing.assert_array_equal(ray.indices, np.array([1, 0]))
    np.testing.assert_allclose(ray.dl, np.array([0.15, 0.65]), atol=1.0e-10)
    assert ray.periodic
    np.testing.assert_allclose(ray.box_size, np.array([10.0, 10.0, 10.0]))


def test_ray_starting_near_face_remains_finite():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([1.0 - 1.0e-12, 0.0, 0.0]),
        end_position=np.array([3.0, 0.0, 0.0]),
    )

    assert np.all(np.isfinite(ray.dl))
    assert np.all(ray.dl > 0)
    np.testing.assert_allclose(ray.dl.sum(), ray.length)


def test_ray_passing_near_vertex_remains_deterministic():
    positions = np.array(
        [[x, y, 0.0] for x in (0.0, 1.0) for y in (0.0, 1.0)]
    )
    kwargs = dict(
        start_position=np.array([-0.25, 0.500000001, 0.0]),
        end_position=np.array([1.25, 0.500000001, 0.0]),
    )

    ray1 = meshless_voronoi_ray(positions, **kwargs)
    ray2 = meshless_voronoi_ray(positions, **kwargs)

    np.testing.assert_array_equal(ray1.indices, ray2.indices)
    np.testing.assert_allclose(ray1.dl, ray2.dl)
    np.testing.assert_allclose(ray1.dl.sum(), ray1.length)


def test_invalid_inputs_raise():
    positions = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        meshless_voronoi_ray(
            positions,
            start_position=np.array([0.0, 0.0, 0.0]),
            end_position=np.array([0.0, 0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="non-zero"):
        meshless_voronoi_ray(
            positions,
            start_position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([0.0, 0.0, 0.0]),
            length=1.0,
        )


def test_trace_metadata_is_reproducible():
    positions = np.array([[float(i), 0.0, 0.0] for i in range(4)])
    ray = meshless_voronoi_ray(
        positions,
        start_position=np.array([-0.5, 0.0, 0.0]),
        end_position=np.array([3.5, 0.0, 0.0]),
    )

    assert ray.metadata["algorithm_version"] == "salsa_meshless_voronoi_v1"
    assert ray.metadata["nearest_index_start"] == 0
    assert ray.metadata["nearest_index_end"] == 3
    assert ray.metadata["number_of_cell_crossings"] == 3
