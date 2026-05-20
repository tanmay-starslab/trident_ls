#!/usr/bin/env python
"""Validation suite comparing Trident LightRay and meshless Voronoi rays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_CONTEXT = Path("/private/tmp/trident_meshless_suite_context")
_RUN_CONTEXT.mkdir(parents=True, exist_ok=True)
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(_RUN_CONTEXT)
for _path in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_KNOWN_ION_TABLES = [
    Path("/Users/wavefunction/github_repos/COS-GASS/tests/56372_sightlines/y/hm2012_ss_hr.h5"),
    Path("/Users/wavefunction/github_repos/COS-GASS/notebooks/y/hm2012_ss_hr.h5"),
    Path("/Users/wavefunction/github_repos/m61-tng/notebooks/y/hm2012_ss_hr.h5"),
    Path("/Users/wavefunction/github_repos/tng-mw-kinematics-spec/notebooks/y/hm2012_ss_hr.h5"),
]
for _table in _KNOWN_ION_TABLES:
    if _table.exists():
        (_RUN_CONTEXT / "config.tri").write_text(
            "[Trident]\n"
            f"ion_table_dir = {_table.parent}\n"
            f"ion_table_file = {_table.name}\n"
        )
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

import trident
from plot_meshless_vs_lightray_diagnostics import (
    ION_FIELDS,
    LINE_GROUPS,
    active_group_keys,
    add_requested_ion_fields,
    column_density,
    ew_approx,
    flux_diff,
    make_spectrum,
    plot_ray_diagnostics,
    plot_spectrum_diagnostics,
    ray_arrays,
    ray_stats,
)
from trident.meshless_ray_io import read_meshless_catalog_ray
from trident.meshless_ray_io import load_meshless_ray_catalog_hdf5


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)
DEFAULT_LINES = ["H I 1216", "O VI 1032", "O VI 1038", "Mg II 2796", "Mg II 2803"]
DEFAULT_IONS = ["H I", "O VI", "Mg II"]
ALL_SCENARIOS = [
    "uniform_xy_grid",
    "uniform_xz_grid",
    "uniform_yz_grid",
    "radial_area_uniform_xy",
    "radial_radius_uniform_xy",
    "random_parallel",
    "diagonal_parallel",
    "short_local_rays",
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def ensure_output_tree(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    for sub in [
        "summary",
        "rays/lightray",
        "rays/meshless",
        "spectra/lightray",
        "spectra/meshless",
        "plots/ray_diagnostics",
        "plots/spectra_diagnostics",
        "plots/summary",
        "logs",
    ]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    return output_dir


def write_rows(rows, csv_path, json_path):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(csv_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with Path(json_path).open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def write_json(payload, path):
    with Path(path).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def resolve_gas_positions(ds):
    ad = ds.all_data()
    candidates = [
        ("gas", "coordinates"),
        ("PartType0", "particle_position"),
        ("PartType0", "Coordinates"),
    ]
    errors = []
    for field in candidates:
        try:
            values = ad[field].to("code_length")
        except Exception as exc:
            errors.append(f"{field}: {exc}")
            continue
        if len(values.shape) == 2 and values.shape[1] == 3:
            return field, values
    raise RuntimeError("Could not resolve gas positions: " + "; ".join(errors))


def gas_position_bounds(positions):
    arr = positions.to("code_length").d
    percentiles = {
        f"p{p:02d}": np.percentile(arr, p, axis=0)
        for p in [1, 5, 50, 95, 99]
    }
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "median": percentiles["p50"],
        "percentiles": percentiles,
        "extent": np.max(arr, axis=0) - np.min(arr, axis=0),
        "n_gas_positions": int(len(arr)),
    }


def normalize_scenarios(values):
    if not values or values == ["all"]:
        return list(ALL_SCENARIOS)
    out = []
    for value in values:
        if value == "all":
            out.extend(ALL_SCENARIOS)
        else:
            out.append(value)
    unknown = sorted(set(out) - set(ALL_SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    return list(dict.fromkeys(out))


def _grid_shape(n):
    nx = int(np.floor(np.sqrt(n)))
    ny = int(np.ceil(n / max(nx, 1)))
    return max(nx, 1), max(ny, 1)


def _clip_to_bounds(starts, ends, lo, hi):
    return np.clip(starts, lo, hi), np.clip(ends, lo, hi)


def generate_sightlines_for_scenario(name, nrays, bounds, seed):
    p01 = bounds["percentiles"]["p01"]
    p05 = bounds["percentiles"]["p05"]
    p50 = bounds["percentiles"]["p50"]
    p95 = bounds["percentiles"]["p95"]
    p99 = bounds["percentiles"]["p99"]
    span = p95 - p05
    rng = np.random.default_rng(seed)

    if name.startswith("uniform_"):
        plane = name.split("_")[1]
        nx, ny = _grid_shape(nrays)
        axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[plane]
        width = 0.9 * span[axes[0]]
        height = 0.9 * span[axes[1]]
        sightlines = trident.generate_uniform_grid_sightlines(
            center=p50,
            width=width,
            height=height,
            nx=nx,
            ny=ny,
            plane=plane,
            start_offset=p01[axes[2]] - p50[axes[2]],
            end_offset=p99[axes[2]] - p50[axes[2]],
        )
        starts, ends = sightlines.starts[:nrays], sightlines.ends[:nrays]
        metadata = dict(sightlines.metadata)
    elif name in {"radial_area_uniform_xy", "radial_radius_uniform_xy"}:
        distribution = "area-uniform" if "area" in name else "radius-uniform"
        radius = 0.45 * min(span[0], span[1])
        sightlines = trident.generate_radial_sightlines(
            center=p50,
            radius=radius,
            nrays=nrays,
            plane="xy",
            length=float(p99[2] - p01[2]),
            radial_distribution=distribution,
            seed=seed,
        )
        starts, ends = sightlines.starts, sightlines.ends
        metadata = dict(sightlines.metadata)
    elif name == "random_parallel":
        sightlines = trident.generate_random_parallel_sightlines(
            center=p50,
            width=0.9 * span[0],
            height=0.9 * span[1],
            nrays=nrays,
            normal_vector=[0.0, 0.0, 1.0],
            length=float(p99[2] - p01[2]),
            seed=seed,
        )
        starts, ends = sightlines.starts, sightlines.ends
        metadata = dict(sightlines.metadata)
    elif name == "diagonal_parallel":
        sightlines = trident.generate_random_parallel_sightlines(
            center=p50,
            width=0.45 * min(span),
            height=0.45 * min(span),
            nrays=nrays,
            normal_vector=[1.0, 1.0, 1.0],
            length=0.55 * min(span),
            seed=seed,
        )
        starts, ends = sightlines.starts, sightlines.ends
        metadata = dict(sightlines.metadata)
    elif name == "short_local_rays":
        starts = []
        ends = []
        length = 0.10 * min(span)
        for _ in range(nrays):
            center = p05 + rng.random(3) * (p95 - p05)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            starts.append(center - 0.5 * length * direction)
            ends.append(center + 0.5 * length * direction)
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        metadata = {"type": "short_local", "length": length, "seed": seed}
    else:
        raise ValueError(name)

    starts, ends = _clip_to_bounds(np.asarray(starts), np.asarray(ends), p01, p99)
    metadata.update({"scenario": name, "nrays_requested": int(nrays)})
    return starts, ends, metadata


def requested_length_cm(ds, start, end):
    return float(ds.arr(np.linalg.norm(np.asarray(end) - np.asarray(start)), "code_length").to("cm").d)


def interpolate_to_common_grid(path, values, n=512):
    path = np.asarray(path, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(path) & np.isfinite(values)
    if np.count_nonzero(mask) == 0:
        return np.linspace(0.0, 1.0, n), np.full(n, np.nan)
    path = path[mask]
    values = values[mask]
    order = np.argsort(path)
    path = path[order]
    values = values[order]
    total = float(np.max(path)) if np.max(path) > 0 else 1.0
    x = np.clip(path / total, 0.0, 1.0)
    unique, unique_index = np.unique(x, return_index=True)
    grid = np.linspace(0.0, 1.0, n)
    if unique.size == 1:
        return grid, np.full(n, values[unique_index[0]])
    return grid, np.interp(grid, unique, values[unique_index])


def field_residual_metrics(light_ray, mesh_ray, fields=None):
    fields = fields or {
        "density": "density",
        "temperature": "temperature",
        "velocity_los": "velocity_los",
        "redshift_eff": "redshift_eff",
    }
    light = ray_arrays(light_ray)
    mesh = ray_arrays(mesh_ray)
    out = {}
    for label, key in fields.items():
        lv = light.get(key)
        mv = mesh.get(key)
        if lv is None or mv is None:
            continue
        _, li = interpolate_to_common_grid(light["pos"], lv[light["order"]])
        _, mi = interpolate_to_common_grid(mesh["pos"], mv[mesh["order"]])
        diff = mi - li
        out[f"{label}_rms_diff"] = float(np.sqrt(np.nanmean(diff**2)))
        out[f"{label}_max_abs_diff"] = float(np.nanmax(np.abs(diff)))
        positive = (li > 0) & (mi > 0)
        if np.any(positive):
            logdiff = np.log10(mi[positive]) - np.log10(li[positive])
            out[f"{label}_log10_rms_diff"] = float(np.sqrt(np.nanmean(logdiff**2)))
    return out


def equivalent_width(wavelength, flux):
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wavelength.size == 0 or flux.size == 0:
        return np.nan
    return float(np.trapezoid(1.0 - flux, wavelength))


def path_weighted_mean(ray, field):
    ad = ray.all_data()
    try:
        values = ad[field].d
        dl = ad[("gas", "dl")].to("cm").d
    except Exception:
        return np.nan
    denom = np.sum(dl)
    return float(np.sum(values * dl) / denom) if denom > 0 else np.nan


def ray_comparison_row(ds, scenario, ray_id, start, end, light_ray, mesh_ray, light_file, mesh_file):
    row = {
        "scenario": scenario,
        "ray_id": int(ray_id),
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "start_z": float(start[2]),
        "end_x": float(end[0]),
        "end_y": float(end[1]),
        "end_z": float(end[2]),
        "requested_length_cm": requested_length_cm(ds, start, end),
        "lightray_file": str(light_file),
        "meshless_file": str(mesh_file),
        "status": "ok",
        "warning_or_error": "",
    }
    try:
        row.update(ray_stats(light_ray, "lightray"))
        row.update(ray_stats(mesh_ray, "meshless"))
        for label, ray in [("lightray", light_ray), ("meshless", mesh_ray)]:
            row[f"path_weighted_density_{label}"] = path_weighted_mean(ray, ("gas", "density"))
            row[f"path_weighted_temperature_{label}"] = path_weighted_mean(ray, ("gas", "temperature"))
            row[f"v_los_std_{label}"] = _field_std(ray, ("gas", "velocity_los"))
            for ion_key in ION_FIELDS:
                col = column_density(ray, ion_key)
                row[f"log10_column_density_{ion_key}_{label}"] = col
                row[f"total_column_like_{ion_key}_{label}"] = col
        row.update(field_residual_metrics(light_ray, mesh_ray))
        row["dl_coverage_lightray"] = row["sum_dl_lightray"] / row["requested_length_cm"]
        row["dl_coverage_meshless"] = row["sum_dl_meshless"] / row["requested_length_cm"]
    except Exception:
        row["status"] = "failed"
        row["warning_or_error"] = traceback.format_exc()
    return row


def _field_std(ray, field):
    try:
        values = ray.all_data()[field].d
    except Exception:
        return np.nan
    return float(np.nanstd(values))


def run_meshless_catalog(ds, starts, ends, scenario_dir, args):
    catalog_file = scenario_dir / "meshless_catalog.h5"
    t0 = time.perf_counter()
    trident.make_meshless_voronoi_ray_catalog(
        ds,
        ds.arr(starts, "code_length"),
        ends=ds.arr(ends, "code_length"),
        lines=args.ions,
        fields=[("gas", "metallicity")],
        output_filename=str(catalog_file),
        position_field=args.position_field,
        periodic=args.periodic,
        parallel=args.parallel,
        chunksize=args.chunksize,
        overwrite=True,
    )
    elapsed = time.perf_counter() - t0
    return catalog_file, elapsed


def count_catalog_segments(catalog_file):
    """Return the number of flat ray segments in a meshless catalog."""
    try:
        catalog = load_meshless_ray_catalog_hdf5(catalog_file)
        return int(len(catalog["segments"]["cell_indices"]))
    except Exception:
        return -1


def run_single_ray_files(ds, starts, ends, count, out_dir, args, method):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rays = {}
    for ray_id in range(count):
        filename = out_dir / f"ray_{ray_id:04d}_{method}.h5"
        t0 = time.perf_counter()
        status = "ok"
        error = ""
        ray = None
        try:
            if method == "lightray":
                ray = trident.make_simple_ray(
                    ds,
                    start_position=ds.arr(starts[ray_id], "code_length"),
                    end_position=ds.arr(ends[ray_id], "code_length"),
                    lines=args.ions,
                    fields=[("gas", "metallicity")],
                    data_filename=str(filename),
                )
            elif method == "meshless":
                ray = trident.make_meshless_voronoi_ray(
                    ds,
                    start_position=ds.arr(starts[ray_id], "code_length"),
                    end_position=ds.arr(ends[ray_id], "code_length"),
                    lines=args.ions,
                    fields=[("gas", "metallicity")],
                    data_filename=str(filename),
                    periodic=args.periodic,
                )
            else:
                raise ValueError(method)
            rays[ray_id] = ray
        except Exception:
            status = "failed"
            error = traceback.format_exc()
        rows.append({
            "ray_id": ray_id,
            "method": method,
            "elapsed_s": time.perf_counter() - t0,
            "status": status,
            "error": error,
            "file": str(filename),
        })
    return rows, rays


def run_spectra_for_pairs(scenario, ray_ids, light_rays, mesh_rays, out_dirs, plot_dir, args, row_lookup):
    spectra_rows = []
    group_keys = active_group_keys(args.lines)
    for ray_id in ray_ids:
        if ray_id not in light_rays or ray_id not in mesh_rays:
            continue
        light_specs = {}
        mesh_specs = {}
        diff_cache = {}
        row = {"scenario": scenario, "ray_id": int(ray_id), "status": "ok", "warning_or_error": ""}
        for group_key in group_keys:
            light_spec, light_err = make_spectrum(
                light_rays[ray_id],
                group_key,
                args.lines,
                out_dirs["lightray"] / f"ray_{ray_id:04d}_{group_key}_spectrum.h5",
                args.instrument,
                1000.0,
            )
            mesh_spec, mesh_err = make_spectrum(
                mesh_rays[ray_id],
                group_key,
                args.lines,
                out_dirs["meshless"] / f"ray_{ray_id:04d}_{group_key}_spectrum.h5",
                args.instrument,
                1000.0,
            )
            light_specs[group_key] = light_spec
            mesh_specs[group_key] = mesh_spec
            row[f"EW_{group_key}_lightray"] = ew_approx(light_spec)
            row[f"EW_{group_key}_meshless"] = ew_approx(mesh_spec)
            rms, max_abs, lam, diff = flux_diff(light_spec, mesh_spec)
            row[f"rms_flux_diff_{group_key}"] = rms
            row[f"max_flux_diff_{group_key}"] = max_abs
            if light_spec is not None:
                row[f"min_flux_{group_key}_lightray"] = float(np.nanmin(light_spec["flux"]))
            if mesh_spec is not None:
                row[f"min_flux_{group_key}_meshless"] = float(np.nanmin(mesh_spec["flux"]))
            if lam is not None:
                diff_cache[group_key] = (lam, diff)
            if light_err or mesh_err:
                row["warning_or_error"] += f"{group_key}: {light_err or ''} {mesh_err or ''}; "
        if args.make_plots:
            plot_spectrum_diagnostics(
                ray_id,
                light_specs,
                mesh_specs,
                plot_dir / f"ray_{ray_id:04d}_spectrum_comparison.png",
                diff_cache,
                group_keys,
                row_lookup.get(ray_id, {}),
            )
        spectra_rows.append(row)
    return spectra_rows


def plot_field_residuals(ray_id, light_ray, mesh_ray, output_path):
    fields = [("density", "density"), ("temperature", "temperature"), ("velocity_los", "v_los"), ("redshift_eff", "redshift_eff")]
    light = ray_arrays(light_ray)
    mesh = ray_arrays(mesh_ray)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), fields):
        lv = light.get(key)
        mv = mesh.get(key)
        if lv is None or mv is None:
            ax.set_title(f"{title}: missing")
            continue
        grid, li = interpolate_to_common_grid(light["pos"], lv[light["order"]])
        _, mi = interpolate_to_common_grid(mesh["pos"], mv[mesh["order"]])
        ax.plot(grid, mi - li, lw=1)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("normalized path")
        ax.set_ylabel("meshless - LightRay")
    fig.suptitle(f"Ray {ray_id:04d} Field Residuals")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sightline_layout(starts, ends, metadata, output_path):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if "projected_x" in metadata and "projected_y" in metadata:
        x = metadata["projected_x"]
        y = metadata["projected_y"]
        ax.set_xlabel("projected x")
        ax.set_ylabel("projected y")
    else:
        x = starts[:, 0]
        y = starts[:, 1]
        ax.set_xlabel("start x [code_length]")
        ax.set_ylabel("start y [code_length]")
    ax.scatter(x, y, s=8, alpha=0.7)
    ax.scatter([0], [0], marker="+", color="k", s=60)
    ax.set_title(f"{metadata.get('scenario', 'scenario')} sightline layout")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_scatter(rows, output_path, title):
    ok = [row for row in rows if row.get("status") == "ok"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    pairs = [
        ("sum_dl_lightray", "sum_dl_meshless", "total dl [cm]"),
        ("n_elements_lightray", "n_elements_meshless", "n segments"),
        ("log10_column_density_HI_lightray", "log10_column_density_HI_meshless", "log10 HI column"),
        ("path_weighted_temperature_lightray", "path_weighted_temperature_meshless", "path-weighted T"),
    ]
    for ax, (xkey, ykey, label) in zip(axes.ravel(), pairs):
        x = np.array([float(row.get(xkey, np.nan)) for row in ok])
        y = np.array([float(row.get(ykey, np.nan)) for row in ok])
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=14, alpha=0.75)
        if np.any(mask):
            lo = min(np.min(x[mask]), np.min(y[mask]))
            hi = max(np.max(x[mask]), np.max(y[mask]))
            ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
        ax.set_xlabel("LightRay")
        ax.set_ylabel("Meshless")
        ax.set_title(label)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spectra_summary(rows, output_path, title):
    ok = [row for row in rows if row.get("status") == "ok"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    pairs = [
        ("EW_HI_lightray", "EW_HI_meshless", "HI EW [A]"),
        ("EW_OVI_lightray", "EW_OVI_meshless", "OVI EW [A]"),
        ("rms_flux_diff_HI", None, "HI RMS flux residual"),
        ("max_flux_diff_HI", None, "HI max abs flux residual"),
    ]
    for ax, (xkey, ykey, label) in zip(axes.ravel(), pairs):
        if ykey is None:
            vals = np.array([float(row.get(xkey, np.nan)) for row in ok])
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=min(20, max(5, len(vals)))) if vals.size else None
            ax.set_xlabel(label)
            ax.set_ylabel("count")
        else:
            x = np.array([float(row.get(xkey, np.nan)) for row in ok])
            y = np.array([float(row.get(ykey, np.nan)) for row in ok])
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=14, alpha=0.75)
            if np.any(mask):
                lo = min(np.min(x[mask]), np.min(y[mask]))
                hi = max(np.max(x[mask]), np.max(y[mask]))
                ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
            ax.set_xlabel("LightRay")
            ax.set_ylabel("Meshless")
        ax.set_title(label)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_timing_summary(scenario, nrays_total, meshless_time, light_timing_rows, meshless_catalog_segments):
    light_ok = [row for row in light_timing_rows if row["status"] == "ok"]
    light_times = np.array([row["elapsed_s"] for row in light_ok], dtype=float)
    mean_light = float(np.mean(light_times)) if light_times.size else np.nan
    median_light = float(np.median(light_times)) if light_times.size else np.nan
    projected = mean_light * nrays_total if np.isfinite(mean_light) else np.nan
    return {
        "scenario": scenario,
        "nrays_total": int(nrays_total),
        "nrays_meshless_run": int(nrays_total),
        "nrays_lightray_run": int(len(light_timing_rows)),
        "meshless_total_time_s": float(meshless_time),
        "meshless_rays_per_sec": float(nrays_total / meshless_time) if meshless_time > 0 else np.nan,
        "meshless_total_segments": int(meshless_catalog_segments),
        "lightray_total_time_s": float(np.sum(light_times)) if light_times.size else np.nan,
        "lightray_mean_per_ray_s": mean_light,
        "lightray_median_per_ray_s": median_light,
        "lightray_projected_total_s": projected,
        "speedup_vs_lightray_projected": float(projected / meshless_time) if meshless_time > 0 and np.isfinite(projected) else np.nan,
        "n_lightray_failures": int(len(light_timing_rows) - len(light_ok)),
        "n_meshless_failures": 0,
    }


def plot_timing_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["scenario"] for row in rows]
    mesh = np.array([row["meshless_total_time_s"] for row in rows], dtype=float)
    projected = np.array([row["lightray_projected_total_s"] for row in rows], dtype=float)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(rows)), 5), constrained_layout=True)
    ax.bar(x - 0.18, mesh, width=0.36, label="meshless batch")
    ax.bar(x + 0.18, projected, width=0.36, label="LightRay projected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("runtime [s]")
    ax.legend()
    fig.savefig(output_dir / "timing_summary.png", dpi=180)
    plt.close(fig)

    speedup = np.array([row["speedup_vs_lightray_projected"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(rows)), 5), constrained_layout=True)
    ax.bar(x, speedup)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("projected LightRay / meshless speedup")
    fig.savefig(output_dir / "speedup_summary.png", dpi=180)
    plt.close(fig)


def run_scenario(ds, scenario, starts, ends, metadata, output_dir, args):
    summary_dir = output_dir / "summary"
    light_ray_dir = output_dir / "rays" / "lightray" / scenario
    mesh_ray_dir = output_dir / "rays" / "meshless" / scenario
    light_spec_dir = output_dir / "spectra" / "lightray" / scenario
    mesh_spec_dir = output_dir / "spectra" / "meshless" / scenario
    ray_plot_dir = output_dir / "plots" / "ray_diagnostics" / scenario
    spec_plot_dir = output_dir / "plots" / "spectra_diagnostics" / scenario
    summary_plot_dir = output_dir / "plots" / "summary"
    for path in [light_ray_dir, mesh_ray_dir, light_spec_dir, mesh_spec_dir, ray_plot_dir, spec_plot_dir]:
        path.mkdir(parents=True, exist_ok=True)

    catalog_file, meshless_time = run_meshless_catalog(ds, starts, ends, mesh_ray_dir, args)
    catalog_ray_count = len(starts)
    meshless_segments = count_catalog_segments(catalog_file)

    subset_count = min(len(starts), max(args.nrays_lightray, args.nrays_spectra))
    light_count = min(args.nrays_lightray, subset_count)
    spectra_count = min(args.nrays_spectra, light_count)
    light_timing, light_rays = run_single_ray_files(ds, starts, ends, light_count, light_ray_dir, args, "lightray")
    mesh_timing, mesh_rays = run_single_ray_files(ds, starts, ends, subset_count, mesh_ray_dir, args, "meshless")

    ray_rows = []
    for ray_id in range(light_count):
        if ray_id not in light_rays or ray_id not in mesh_rays:
            continue
        row = ray_comparison_row(
            ds,
            scenario,
            ray_id,
            starts[ray_id],
            ends[ray_id],
            light_rays[ray_id],
            mesh_rays[ray_id],
            light_ray_dir / f"ray_{ray_id:04d}_lightray.h5",
            mesh_ray_dir / f"ray_{ray_id:04d}_meshless.h5",
        )
        ray_rows.append(row)
        if args.make_plots and ray_id < min(10, light_count):
            plot_ray_diagnostics(ray_id, light_rays[ray_id], mesh_rays[ray_id], ray_plot_dir / f"ray_{ray_id:04d}_ray_fields.png")
            plot_field_residuals(ray_id, light_rays[ray_id], mesh_rays[ray_id], ray_plot_dir / f"ray_{ray_id:04d}_field_residuals.png")

    spectra_rows = []
    if args.make_spectra:
        spectra_rows = run_spectra_for_pairs(
            scenario,
            list(range(spectra_count)),
            light_rays,
            mesh_rays,
            {"lightray": light_spec_dir, "meshless": mesh_spec_dir},
            spec_plot_dir,
            args,
            {row["ray_id"]: row for row in ray_rows},
        )

    if args.make_plots:
        plot_sightline_layout(starts, ends, metadata, summary_plot_dir / f"{scenario}_sightline_layout.png")
        plot_summary_scatter(ray_rows, summary_plot_dir / f"{scenario}_ray_summary.png", f"{scenario} ray summary")
        if spectra_rows:
            plot_spectra_summary(
                spectra_rows,
                summary_plot_dir / f"{scenario}_spectra_summary.png",
                f"{scenario} spectra summary",
            )

    timing_row = build_timing_summary(scenario, len(starts), meshless_time, light_timing, meshless_segments)
    write_rows(ray_rows, summary_dir / f"{scenario}_ray_comparison_summary.csv", summary_dir / f"{scenario}_ray_comparison_summary.json")
    write_rows(spectra_rows, summary_dir / f"{scenario}_spectra_comparison_summary.csv", summary_dir / f"{scenario}_spectra_comparison_summary.json")
    write_rows(light_timing + mesh_timing, summary_dir / f"{scenario}_ray_generation_timing.csv", summary_dir / f"{scenario}_ray_generation_timing.json")
    return ray_rows, spectra_rows, timing_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--nrays-total", type=int, default=500)
    parser.add_argument("--nrays-lightray", type=int, default=100)
    parser.add_argument("--nrays-spectra", type=int, default=50)
    parser.add_argument("--scenario", nargs="+", default=["all"])
    parser.add_argument("--periodic", type=parse_bool, default=False)
    parser.add_argument("--instrument", default="COS-G130M")
    parser.add_argument("--lines", nargs="+", default=DEFAULT_LINES)
    parser.add_argument("--ions", nargs="+", default=DEFAULT_IONS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--make-spectra", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--parallel", choices=["none", "threads"], default="none")
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--full-lightray-500", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/private/tmp/trident_meshless_lightray_comparison_{stamp}"
    if args.full_lightray_500:
        args.nrays_lightray = args.nrays_total

    output_dir = ensure_output_tree(args.output_dir, overwrite=args.overwrite)
    yt.set_log_level(40)
    t_start = time.perf_counter()
    ds = yt.load(args.dataset)
    ion_warning = add_requested_ion_fields(ds, args.ions)
    position_field, positions = resolve_gas_positions(ds)
    args.position_field = position_field
    bounds = gas_position_bounds(positions)
    scenarios = normalize_scenarios(args.scenario)

    metadata = {
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "scenarios": scenarios,
        "nrays_total": args.nrays_total,
        "nrays_lightray": args.nrays_lightray,
        "nrays_spectra": args.nrays_spectra,
        "position_field": str(position_field),
        "gas_bounds": bounds,
        "ion_warning": ion_warning,
    }
    write_json(metadata, output_dir / "summary" / "run_metadata.json")

    all_ray_rows = []
    all_spectra_rows = []
    timing_rows = []
    for scenario_index, scenario in enumerate(scenarios):
        starts, ends, sightline_metadata = generate_sightlines_for_scenario(
            scenario, args.nrays_total, bounds, args.seed + 1000 * scenario_index
        )
        ray_rows, spectra_rows, timing_row = run_scenario(
            ds, scenario, starts, ends, sightline_metadata, output_dir, args
        )
        all_ray_rows.extend(ray_rows)
        all_spectra_rows.extend(spectra_rows)
        timing_rows.append(timing_row)
        print(
            f"{scenario}: meshless {timing_row['nrays_meshless_run']} rays in "
            f"{timing_row['meshless_total_time_s']:.3f}s; LightRay "
            f"{timing_row['nrays_lightray_run']} rays projected speedup "
            f"{timing_row['speedup_vs_lightray_projected']:.2f}x"
        )

    write_rows(all_ray_rows, output_dir / "summary" / "ray_comparison_summary.csv", output_dir / "summary" / "ray_comparison_summary.json")
    write_rows(all_spectra_rows, output_dir / "summary" / "spectra_comparison_summary.csv", output_dir / "summary" / "spectra_comparison_summary.json")
    write_rows(timing_rows, output_dir / "summary" / "timing_summary.csv", output_dir / "summary" / "timing_summary.json")
    if args.make_plots:
        plot_timing_summary(timing_rows, output_dir / "plots" / "summary")

    elapsed = time.perf_counter() - t_start
    print("Comparison suite complete")
    print(f"  output_dir: {output_dir}")
    print(f"  scenarios: {len(scenarios)}")
    print(f"  ray rows: {len(all_ray_rows)}")
    print(f"  spectra rows: {len(all_spectra_rows)}")
    print(f"  elapsed_s: {elapsed:.3f}")


if __name__ == "__main__":
    main()
