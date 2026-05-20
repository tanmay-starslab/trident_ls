import numpy as np
import yt
from yt.utilities.physical_constants import speed_of_light_cgs

import trident as tri


def test_make_meshless_voronoi_ray_onezone_spectrum(tmp_path):
    ds = tri.make_onezone_dataset()
    ray_filename = tmp_path / "meshless_ray.h5"

    ray = tri.make_meshless_voronoi_ray(
        ds,
        start_position=ds.arr([0.0, 0.0, 0.0], "unitary"),
        end_position=ds.arr([1.0, 1.0, 1.0], "unitary"),
        lines=["H I"],
        data_filename=str(ray_filename),
        position_field=(("index", "x"), ("index", "y"), ("index", "z")),
        periodic=False,
    )

    ad = ray.all_data()
    for field in [
        ("gas", "dl"),
        ("gas", "l"),
        ("gas", "redshift"),
        ("gas", "redshift_dopp"),
        ("gas", "redshift_eff"),
        ("gas", "velocity_los"),
        ("gas", "meshless_cell_index"),
        ("gas", "cumulative_dl"),
    ]:
        assert field in ray.derived_field_list
        assert len(ad[field]) == 1

    expected_length = ds.quan(np.sqrt(3.0), "unitary").in_cgs()
    np.testing.assert_allclose(ad[("gas", "dl")].sum().d, expected_length.d)
    assert ray.parameters["meshless_algorithm_version"] == "salsa_meshless_voronoi_v1"

    sg = tri.SpectrumGenerator(lambda_min=1210.0, lambda_max=1220.0, dlambda=0.1)
    sg.make_spectrum(ray, lines=["H I 1216"])


def test_make_meshless_voronoi_ray_can_omit_extra_metadata(tmp_path):
    ds = tri.make_onezone_dataset()
    ray_filename = tmp_path / "meshless_ray_minimal.h5"

    ray = tri.make_meshless_voronoi_ray(
        ds,
        start_position=ds.arr([0.0, 0.0, 0.0], "unitary"),
        end_position=ds.arr([1.0, 0.0, 0.0], "unitary"),
        fields=[("gas", "density")],
        data_filename=str(ray_filename),
        position_field=(("index", "x"), ("index", "y"), ("index", "z")),
        extra_ray_fields=False,
        store_meshless_metadata=False,
        periodic=False,
    )

    assert ("gas", "dl") in ray.derived_field_list
    assert ("gas", "meshless_cell_index") not in ray.derived_field_list
    assert "meshless_algorithm_version" not in ray.parameters


def test_make_meshless_voronoi_ray_matches_lightray_redshift_convention(tmp_path):
    velocity = 3.0e7
    data = {
        "density": np.ones((1, 1, 1)) * 1.0e-26,
        "temperature": np.ones((1, 1, 1)) * 1.0e4,
        "metallicity": np.ones((1, 1, 1)) * 0.3,
        "velocity_x": np.ones((1, 1, 1)) * velocity,
        "velocity_y": np.zeros((1, 1, 1)),
        "velocity_z": np.zeros((1, 1, 1)),
    }
    ds = yt.load_uniform_grid(
        data, (1, 1, 1), length_unit="cm", bbox=np.array([[0.0, 1.0]] * 3)
    )
    start = ds.arr([0.0, 0.5, 0.5], "code_length")
    end = ds.arr([1.0, 0.5, 0.5], "code_length")

    mesh_ray = tri.make_meshless_voronoi_ray(
        ds,
        start_position=start,
        end_position=end,
        redshift=0.1,
        data_filename=str(tmp_path / "meshless_redshift.h5"),
        position_field=(("index", "x"), ("index", "y"), ("index", "z")),
        periodic=False,
    )
    light_ray = tri.make_simple_ray(
        ds,
        start_position=start,
        end_position=end,
        redshift=0.1,
        data_filename=str(tmp_path / "lightray_redshift.h5"),
    )

    mesh_ad = mesh_ray.all_data()
    light_ad = light_ray.all_data()
    expected_vlos = -velocity
    beta = velocity / speed_of_light_cgs.v
    expected_zdopp = (1.0 + expected_vlos / speed_of_light_cgs.v) / np.sqrt(1.0 - beta**2) - 1.0
    np.testing.assert_allclose(
        mesh_ad[("gas", "velocity_los")].to("cm/s").d,
        light_ad[("gas", "velocity_los")].to("cm/s").d,
    )
    np.testing.assert_allclose(
        mesh_ad[("gas", "redshift_dopp")].d,
        expected_zdopp,
    )
    np.testing.assert_allclose(
        mesh_ad[("gas", "redshift_eff")].d,
        ((1.0 + mesh_ad[("gas", "redshift")].d) * (1.0 + expected_zdopp)) - 1.0,
    )


def test_meshless_and_lightray_column_densities_match_for_onezone(tmp_path):
    ds = tri.make_onezone_dataset()
    start = ds.arr([0.0, 0.0, 0.0], "unitary")
    end = ds.arr([1.0, 1.0, 1.0], "unitary")
    lines = ["H I", "O VI", "C IV"]
    tri.add_ion_fields(ds, ions=lines)

    mesh_ray = tri.make_meshless_voronoi_ray(
        ds,
        start_position=start,
        end_position=end,
        lines=lines,
        data_filename=str(tmp_path / "meshless_columns.h5"),
        position_field=(("index", "x"), ("index", "y"), ("index", "z")),
        periodic=False,
    )
    light_ray = tri.make_simple_ray(
        ds,
        start_position=start,
        end_position=end,
        lines=lines,
        data_filename=str(tmp_path / "lightray_columns.h5"),
    )

    columns = []
    for atom, ion, field in [
        ("H", 1, ("gas", "H_p0_number_density")),
        ("O", 6, ("gas", "O_p5_number_density")),
        ("C", 4, ("gas", "C_p3_number_density")),
    ]:
        assert field in mesh_ray.derived_field_list
        assert field in light_ray.derived_field_list
        mesh_ad = mesh_ray.all_data()
        light_ad = light_ray.all_data()
        mesh_column = (mesh_ad[field] * mesh_ad[("gas", "dl")]).sum().to("cm**-2")
        light_column = (light_ad[field] * light_ad[("gas", "dl")]).sum().to("cm**-2")
        assert np.isfinite(mesh_column.d)
        columns.append(mesh_column.d)
        if mesh_column.d > 0 and light_column.d > 0:
            np.testing.assert_allclose(
                np.log10(mesh_column.d), np.log10(light_column.d), rtol=1.0e-12
            )
        else:
            np.testing.assert_allclose(mesh_column.d, light_column.d, rtol=1.0e-12)
    assert np.any(np.asarray(columns) > 0)
