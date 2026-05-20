import numpy as np

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
