import numpy as np

import trident as tri
from trident.meshless_ray_io import (
    load_meshless_ray_catalog_hdf5,
    read_meshless_catalog_ray,
    write_meshless_ray_catalog_hdf5,
)
from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer


def test_catalog_hdf5_roundtrip_geometry(tmp_path):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    starts = np.array([[-0.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    ends = np.array([[3.5, 0.0, 0.0], [2.25, 0.0, 0.0]])
    tracer = MeshlessVoronoiRayTracer(points)
    batch = tracer.trace_rays(starts, end_positions=ends, return_format="ragged")
    filename = tmp_path / "meshless_catalog.h5"

    write_meshless_ray_catalog_hdf5(
        filename,
        batch,
        fields={("gas", "density"): np.arange(len(batch.indices), dtype=float)},
        metadata={"source": "unit-test"},
    )
    catalog = load_meshless_ray_catalog_hdf5(filename)

    assert catalog["attrs"]["data_type"] == "meshless_voronoi_ray_catalog"
    np.testing.assert_array_equal(catalog["rays"]["offsets"], batch.ray_offsets)
    np.testing.assert_array_equal(catalog["segments"]["cell_indices"], batch.indices)
    np.testing.assert_allclose(catalog["segments"]["dl"], batch.dl)
    assert "gas__density" in catalog["fields"]

    ray1 = read_meshless_catalog_ray(filename, 1)
    expected = batch.ray(1)
    np.testing.assert_array_equal(ray1["indices"], expected.indices)
    np.testing.assert_allclose(ray1["dl"], expected.dl)
    np.testing.assert_allclose(ray1["start_position"], expected.start_position)


def test_high_level_meshless_catalog_api_writes_fields(tmp_path):
    ds = tri.make_onezone_dataset()
    starts = ds.arr([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]], "unitary")
    ends = ds.arr([[1.0, 1.0, 1.0], [1.0, 0.5, 0.5]], "unitary")
    filename = tmp_path / "onezone_catalog.h5"

    result = tri.make_meshless_voronoi_ray_catalog(
        ds,
        starts,
        ends=ends,
        fields=[("gas", "density")],
        output_filename=str(filename),
        position_field=(("index", "x"), ("index", "y"), ("index", "z")),
        periodic=False,
        overwrite=True,
    )

    assert result == str(filename)
    catalog = load_meshless_ray_catalog_hdf5(filename)
    assert catalog["attrs"]["n_rays"] == 2
    assert catalog["segments"]["dl"].size == 2
    assert "gas__density" in catalog["fields"]
    assert "gas__velocity_los" in catalog["fields"]
