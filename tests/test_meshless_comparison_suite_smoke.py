"""Smoke tests for the LightRay-vs-meshless comparison utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_suite():
    root = Path(__file__).resolve().parents[1]
    path = root / "dev" / "meshless_tests" / "compare_lightray_vs_meshless_suite.py"
    spec = importlib.util.spec_from_file_location("compare_lightray_vs_meshless_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_bounds():
    return {
        "min": np.array([0.0, 0.0, 0.0]),
        "max": np.array([10.0, 12.0, 14.0]),
        "median": np.array([5.0, 6.0, 7.0]),
        "extent": np.array([10.0, 12.0, 14.0]),
        "n_gas_positions": 100,
        "percentiles": {
            "p01": np.array([1.0, 1.0, 1.0]),
            "p05": np.array([2.0, 2.0, 2.0]),
            "p50": np.array([5.0, 6.0, 7.0]),
            "p95": np.array([8.0, 10.0, 12.0]),
            "p99": np.array([9.0, 11.0, 13.0]),
        },
    }


def test_interpolate_to_common_grid_handles_unsorted_points():
    suite = _load_suite()
    grid, values = suite.interpolate_to_common_grid([2.0, 0.0, 1.0], [4.0, 0.0, 2.0], n=5)

    np.testing.assert_allclose(grid, np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(values, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_equivalent_width_basic_trapezoid():
    suite = _load_suite()
    wavelength = np.array([0.0, 1.0, 2.0])
    flux = np.array([1.0, 0.5, 1.0])

    assert suite.equivalent_width(wavelength, flux) == 0.5


def test_generate_sightlines_shapes_and_bounds():
    suite = _load_suite()
    starts, ends, metadata = suite.generate_sightlines_for_scenario(
        "uniform_xy_grid", 7, _fake_bounds(), seed=398784
    )

    assert starts.shape == (7, 3)
    assert ends.shape == (7, 3)
    assert metadata["scenario"] == "uniform_xy_grid"
    assert np.all(starts >= _fake_bounds()["percentiles"]["p01"])
    assert np.all(ends <= _fake_bounds()["percentiles"]["p99"])


def test_timing_summary_projection():
    suite = _load_suite()
    rows = [
        {"elapsed_s": 2.0, "status": "ok"},
        {"elapsed_s": 4.0, "status": "ok"},
        {"elapsed_s": 99.0, "status": "failed"},
    ]

    summary = suite.build_timing_summary("case", 10, 5.0, rows, 123)

    assert summary["nrays_lightray_run"] == 3
    assert summary["n_lightray_failures"] == 1
    assert summary["lightray_mean_per_ray_s"] == 3.0
    assert summary["lightray_projected_total_s"] == 30.0
    assert summary["speedup_vs_lightray_projected"] == 6.0


def test_summary_plots_create_files(tmp_path):
    suite = _load_suite()
    ray_rows = [
        {
            "status": "ok",
            "sum_dl_lightray": 1.0,
            "sum_dl_meshless": 1.1,
            "n_elements_lightray": 10,
            "n_elements_meshless": 8,
            "log10_column_density_HI_lightray": 13.0,
            "log10_column_density_HI_meshless": 13.2,
            "path_weighted_temperature_lightray": 1.0e6,
            "path_weighted_temperature_meshless": 1.2e6,
        }
    ]
    spectra_rows = [
        {
            "status": "ok",
            "EW_HI_lightray": 0.1,
            "EW_HI_meshless": 0.12,
            "EW_OVI_lightray": 0.02,
            "EW_OVI_meshless": 0.03,
            "rms_flux_diff_HI": 0.01,
            "max_flux_diff_HI": 0.05,
        }
    ]
    timing_rows = [
        {
            "scenario": "case",
            "meshless_total_time_s": 1.0,
            "lightray_projected_total_s": 5.0,
            "speedup_vs_lightray_projected": 5.0,
        }
    ]
    starts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ends = starts + np.array([0.0, 0.0, 1.0])

    suite.plot_summary_scatter(ray_rows, tmp_path / "ray_summary.png", "ray summary")
    suite.plot_spectra_summary(spectra_rows, tmp_path / "spectra_summary.png", "spectra summary")
    suite.plot_sightline_layout(starts, ends, {"scenario": "case"}, tmp_path / "layout.png")
    suite.plot_timing_summary(timing_rows, tmp_path)

    for name in [
        "ray_summary.png",
        "spectra_summary.png",
        "layout.png",
        "timing_summary.png",
        "speedup_summary.png",
    ]:
        assert (tmp_path / name).exists()
