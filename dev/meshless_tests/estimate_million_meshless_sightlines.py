#!/usr/bin/env python
"""Estimate meshless CGM sightline and spectrum cost for a million rays.

This script is intentionally meshless-only.  It creates an area-uniform radial
set of z-parallel sightlines through a spherical TNG cutout, runs the optimized
meshless catalog path for a configurable smoke-test count, generates spectra
for a configurable subset, and extrapolates the measured rates to 10^6 rays.
"""

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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_CONTEXT = Path("/private/tmp/trident_meshless_million_context")
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

import trident
from plot_meshless_vs_lightray_diagnostics import (
    ION_FIELDS,
    LINE_GROUPS,
    add_requested_ion_fields,
    active_group_keys,
    column_density,
    ew_approx,
    make_spectrum,
)
from trident.meshless_ray_io import load_meshless_ray_catalog_hdf5


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/out_sub_488530/"
    "cutout_ALLFIELDS_sphere_2p1Rvir_sub488530.hdf5"
)
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT / "dev" / "meshless_tests" / "output" /
    "million_ray_projection_sub488530"
)
DEFAULT_LINES = [
    line
    for group in LINE_GROUPS.values()
    for line in group["lines"]
]
DEFAULT_IONS = ["H I", "C II", "C III", "C IV", "N V", "O I", "O VI", "Si II", "Si IV", "Mg II"]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_cutout_info(dataset):
    info = {
        "center_source": "unknown",
        "selection_radius_source": "unknown",
    }
    with h5py.File(dataset, "r") as handle:
        header = handle["Header"].attrs
        h = float(header.get("HubbleParam", 1.0))
        z = float(header.get("Redshift", 0.0))
        box_size = float(header.get("BoxSize", np.nan))
        gas_count = int(header.get("NumPart_ThisFile", [0])[0])
        info.update(
            hubble_param=h,
            redshift=z,
            box_size_ckpc_h=box_size,
            gas_particle_count=gas_count,
        )
        if "CutoutInfo" in handle:
            attrs = handle["CutoutInfo"].attrs
            selection = attrs.get("Selection")
            if selection is not None:
                if isinstance(selection, bytes):
                    selection = selection.decode("utf-8")
                try:
                    parsed = json.loads(str(selection))
                    if "center_ckpc_h" in parsed:
                        info["center_ckpc_h"] = np.asarray(parsed["center_ckpc_h"], dtype=float)
                        info["center_source"] = "CutoutInfo.Selection.center_ckpc_h"
                    if "radius_ckpc_h" in parsed:
                        info["selection_radius_ckpc_h"] = float(parsed["radius_ckpc_h"])
                        info["selection_radius_source"] = "CutoutInfo.Selection.radius_ckpc_h"
                    info["selection"] = parsed
                except Exception as exc:
                    info["selection_parse_error"] = str(exc)
    if "center_ckpc_h" not in info:
        raise RuntimeError("Could not find CutoutInfo.Selection.center_ckpc_h in dataset.")
    if "selection_radius_ckpc_h" in info:
        info["inferred_rvir_ckpc_h"] = info["selection_radius_ckpc_h"] / 2.1
        info["inferred_rvir_pkpc"] = info["inferred_rvir_ckpc_h"] / (h * (1.0 + z))
    return info


def code_per_physical_kpc(cutout_info):
    return float(cutout_info["hubble_param"]) * (1.0 + float(cutout_info["redshift"]))


def radial_z_sightlines(center_code, nrays, impact_radius_code, half_length_code, seed, code_per_kpc=1.0):
    rng = np.random.default_rng(seed)
    u = rng.random(int(nrays))
    phi_u = rng.random(int(nrays))
    impact = np.sqrt(u) * float(impact_radius_code)
    phi = 2.0 * np.pi * phi_u
    x = center_code[0] + impact * np.cos(phi)
    y = center_code[1] + impact * np.sin(phi)
    starts = np.column_stack([
        x,
        y,
        np.full(int(nrays), center_code[2] - float(half_length_code)),
    ])
    ends = np.column_stack([
        x,
        y,
        np.full(int(nrays), center_code[2] + float(half_length_code)),
    ])
    metadata = {
        "impact_parameter_code": impact,
        "impact_parameter_kpc": impact / float(code_per_kpc),
        "phi": phi,
        "projected_x_code": impact * np.cos(phi),
        "projected_y_code": impact * np.sin(phi),
    }
    return np.ascontiguousarray(starts), np.ascontiguousarray(ends), metadata


def dataset_size(path):
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def seconds_to_human(seconds):
    if not np.isfinite(seconds):
        return "nan"
    seconds = float(seconds)
    if seconds < 120:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 120:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 72:
        return f"{hours:.1f} hr"
    return f"{hours / 24.0:.1f} days"


def plot_layout(output_path, starts, center_code, impact_radius_code):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    dx = starts[:, 0] - center_code[0]
    dy = starts[:, 1] - center_code[1]
    ax.scatter(dx, dy, s=2, alpha=0.35)
    circle = plt.Circle((0.0, 0.0), impact_radius_code, fill=False, color="black", lw=1.0)
    ax.add_patch(circle)
    ax.scatter([0.0], [0.0], marker="+", s=80, color="red", label="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x - center [ckpc/h]")
    ax.set_ylabel("y - center [ckpc/h]")
    ax.set_title("Area-uniform radial z-parallel sightlines")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_timing_projection(output_path, projection):
    labels = ["rays only", "spectra", "rays + spectra"]
    smoke = [
        projection["smoke_meshless_catalog_time_s"],
        projection["smoke_spectrum_time_s"],
        projection["smoke_total_with_spectra_s"],
    ]
    million = [
        projection["million_meshless_catalog_time_s"],
        projection["million_spectrum_time_s"],
        projection["million_total_with_spectra_s"],
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(x, smoke)
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("time [s]")
    axes[0].set_title("Measured smoke test")
    axes[1].bar(x, np.asarray(million) / 3600.0)
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("projected time [hr]")
    axes[1].set_title("Projected 10^6 rays")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_segments(output_path, n_segments):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(n_segments, bins=40, histtype="stepfilled", alpha=0.75)
    ax.set_xlabel("segments per ray")
    ax.set_ylabel("count")
    ax.set_title("Meshless segment count distribution")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_catalog(catalog_file, nrays):
    catalog = load_meshless_ray_catalog_hdf5(catalog_file)
    offsets = np.asarray(catalog["rays"]["offsets"], dtype=np.int64)
    dl = np.asarray(catalog["segments"]["dl"], dtype=np.float64)
    lengths = np.asarray(catalog["rays"]["lengths"], dtype=np.float64)
    n_segments = np.diff(offsets)
    sum_dl = np.add.reduceat(dl, offsets[:-1]) if len(dl) else np.zeros(nrays)
    valid = n_segments > 0
    summary = {
        "n_rays": int(nrays),
        "total_segments": int(len(dl)),
        "mean_segments_per_ray": float(np.mean(n_segments)),
        "median_segments_per_ray": float(np.median(n_segments)),
        "min_segments_per_ray": int(np.min(n_segments)),
        "max_segments_per_ray": int(np.max(n_segments)),
        "mean_sum_dl_code": float(np.mean(sum_dl[valid])) if np.any(valid) else np.nan,
        "requested_length_code": float(np.median(lengths)),
        "max_relative_length_error": float(np.max(np.abs(sum_dl[valid] - lengths[valid]) / lengths[valid]))
        if np.any(valid)
        else np.nan,
        "empty_rays": int(np.sum(~valid)),
    }
    return summary, n_segments


def spectrum_subset(
    ds,
    starts,
    ends,
    output_dir,
    args,
):
    ray_dir = output_dir / "rays" / "spectrum_subset"
    spectrum_dir = output_dir / "spectra"
    ray_dir.mkdir(parents=True, exist_ok=True)
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    subset_count = min(int(args.spectra_rays), len(starts))
    if subset_count <= 0:
        return [], {
            "spectra_subset_rays": 0,
            "spectrum_time_s": 0.0,
            "successful_spectra": 0,
            "failed_or_skipped_spectra": 0,
            "spectrum_groups_attempted_per_ray": 0,
        }

    t0 = time.perf_counter()
    trident.make_meshless_voronoi_ray_catalog(
        ds,
        starts[:subset_count],
        ends=ends[:subset_count],
        lines=args.ions,
        fields=[("gas", "metallicity")],
        output_filename=str(output_dir / "catalogs" / "spectrum_subset_catalog.h5"),
        output_dir=str(ray_dir),
        one_file_per_ray=True,
        periodic=args.periodic,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        chunksize=args.chunksize,
        overwrite=True,
        fail_empty=False,
    )
    ray_write_time = time.perf_counter() - t0

    group_keys = active_group_keys(args.lines)
    rows = []
    spectrum_start = time.perf_counter()
    for ray_id in range(subset_count):
        ray_file = ray_dir / f"meshless_ray_{ray_id:05d}.h5"
        try:
            ray = yt.load(str(ray_file))
        except Exception as exc:
            rows.append(
                {
                    "ray_id": ray_id,
                    "group": "",
                    "status": "ray_load_failed",
                    "warning_or_error": str(exc),
                }
            )
            continue

        columns = {}
        for ion_key in ION_FIELDS:
            if ion_key not in group_keys:
                continue
            columns[f"log10_column_{ion_key}"] = column_density(ray, ion_key)

        for group_key in group_keys:
            out = spectrum_dir / group_key / f"ray_{ray_id:05d}_{group_key}.h5"
            out.parent.mkdir(parents=True, exist_ok=True)
            t_group = time.perf_counter()
            spec, warning = make_spectrum(
                ray,
                group_key,
                args.lines,
                out,
                args.instrument,
                args.velocity_window_kms,
            )
            elapsed = time.perf_counter() - t_group
            rows.append(
                {
                    "ray_id": ray_id,
                    "group": group_key,
                    "status": "success" if spec is not None and not warning else "skipped_or_failed",
                    "warning_or_error": warning or "",
                    "runtime_s": elapsed,
                    "ew_approx_A": ew_approx(spec),
                    "min_flux": float(np.nanmin(spec["flux"])) if spec is not None else np.nan,
                    "spectrum_file": spec["file"] if spec is not None else "",
                    **columns,
                }
            )
    spectrum_time = time.perf_counter() - spectrum_start
    successful = sum(1 for row in rows if row.get("status") == "success")
    stats = {
        "spectra_subset_rays": int(subset_count),
        "spectrum_ray_write_time_s": float(ray_write_time),
        "spectrum_generation_time_s": float(spectrum_time),
        "spectrum_total_time_s": float(ray_write_time + spectrum_time),
        "successful_spectra": int(successful),
        "failed_or_skipped_spectra": int(len(rows) - successful),
        "spectrum_groups_attempted_per_ray": int(len(group_keys)),
    }
    return rows, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--nrays", type=int, default=10000)
    parser.add_argument("--target-rays", type=float, default=1_000_000)
    parser.add_argument("--spectra-rays", type=int, default=25)
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--half-length-kpc", type=float, default=500.0)
    parser.add_argument("--impact-radius-kpc", type=float, default=500.0)
    parser.add_argument("--velocity-window-kms", type=float, default=1000.0)
    parser.add_argument("--instrument", default="custom")
    parser.add_argument("--lines", nargs="+", default=DEFAULT_LINES)
    parser.add_argument("--ions", nargs="+", default=DEFAULT_IONS)
    parser.add_argument("--periodic", type=parse_bool, default=False)
    parser.add_argument("--parallel", choices=["none", "threads"], default="none")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-spectra", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["catalogs", "rays", "spectra", "plots", "summary", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "logs" / "run.log"
    run_status = "success"
    warnings = []
    try:
        t_run = time.perf_counter()
        cutout_info = read_cutout_info(args.dataset)
        factor = code_per_physical_kpc(cutout_info)
        center_code = np.asarray(cutout_info["center_ckpc_h"], dtype=float)
        half_length_code = float(args.half_length_kpc) * factor
        impact_radius_code = float(args.impact_radius_kpc) * factor
        starts, ends, sightline_metadata = radial_z_sightlines(
            center_code,
            args.nrays,
            impact_radius_code,
            half_length_code,
            args.seed,
            factor,
        )
        extent_code = np.sqrt(impact_radius_code**2 + half_length_code**2)
        if "selection_radius_ckpc_h" in cutout_info and extent_code > cutout_info["selection_radius_ckpc_h"]:
            warnings.append(
                "Requested ray endpoints reach %.3f ckpc/h from center, outside cutout radius %.3f ckpc/h."
                % (extent_code, cutout_info["selection_radius_ckpc_h"])
            )

        write_json(
            output_dir / "summary" / "sightline_metadata.json",
            {
                "cutout_info": cutout_info,
                "center_ckpc_h": center_code,
                "half_length_kpc": args.half_length_kpc,
                "half_length_ckpc_h": half_length_code,
                "total_path_length_kpc": 2.0 * args.half_length_kpc,
                "total_path_length_ckpc_h": 2.0 * half_length_code,
                "impact_radius_kpc": args.impact_radius_kpc,
                "impact_radius_ckpc_h": impact_radius_code,
                "endpoint_radius_ckpc_h": extent_code,
                "nrays": args.nrays,
                "seed": args.seed,
            },
        )
        plot_layout(output_dir / "plots" / "sightline_layout.png", starts, center_code, impact_radius_code)

        yt.set_log_level(40)
        t0 = time.perf_counter()
        ds = yt.load(args.dataset)
        yt_load_s = time.perf_counter() - t0
        ion_warning = add_requested_ion_fields(ds, args.ions)
        if ion_warning:
            warnings.append(ion_warning)

        catalog_file = output_dir / "catalogs" / f"meshless_radial_z_{args.nrays}.h5"
        t0 = time.perf_counter()
        trident.make_meshless_voronoi_ray_catalog(
            ds,
            starts,
            ends=ends,
            lines=args.ions,
            fields=[("gas", "metallicity")],
            output_filename=str(catalog_file),
            periodic=args.periodic,
            parallel=args.parallel,
            n_jobs=args.n_jobs,
            chunksize=args.chunksize,
            overwrite=True,
            fail_empty=False,
        )
        catalog_time_s = time.perf_counter() - t0
        catalog_summary, n_segments = summarize_catalog(str(catalog_file), args.nrays)
        catalog_size = dataset_size(catalog_file)
        plot_segments(output_dir / "plots" / "segments_per_ray.png", n_segments)

        spectrum_rows = []
        spectrum_stats = {
            "spectra_subset_rays": 0,
            "spectrum_ray_write_time_s": 0.0,
            "spectrum_generation_time_s": 0.0,
            "spectrum_total_time_s": 0.0,
            "successful_spectra": 0,
            "failed_or_skipped_spectra": 0,
            "spectrum_groups_attempted_per_ray": 0,
        }
        if not args.skip_spectra:
            spectrum_rows, spectrum_stats = spectrum_subset(ds, starts, ends, output_dir, args)

        target = float(args.target_rays)
        ray_scale = target / float(args.nrays)
        spectra_scale = (
            target / float(spectrum_stats["spectra_subset_rays"])
            if spectrum_stats["spectra_subset_rays"] > 0
            else np.nan
        )
        per_ray_spectrum_total = (
            spectrum_stats["spectrum_total_time_s"] / spectrum_stats["spectra_subset_rays"]
            if spectrum_stats["spectra_subset_rays"] > 0
            else np.nan
        )
        per_ray_spectrum_generation = (
            spectrum_stats["spectrum_generation_time_s"] / spectrum_stats["spectra_subset_rays"]
            if spectrum_stats["spectra_subset_rays"] > 0
            else np.nan
        )
        per_ray_spectrum_export = (
            spectrum_stats["spectrum_ray_write_time_s"] / spectrum_stats["spectra_subset_rays"]
            if spectrum_stats["spectra_subset_rays"] > 0
            else np.nan
        )
        projection = {
            "target_rays": int(target),
            "smoke_nrays": int(args.nrays),
            "smoke_spectra_rays": int(spectrum_stats["spectra_subset_rays"]),
            "yt_load_time_s": float(yt_load_s),
            "smoke_meshless_catalog_time_s": float(catalog_time_s),
            "smoke_meshless_rays_per_s": float(args.nrays / catalog_time_s),
            "smoke_meshless_seconds_per_ray": float(catalog_time_s / args.nrays),
            "smoke_spectrum_time_s": float(spectrum_stats["spectrum_total_time_s"]),
            "smoke_spectrum_generation_only_time_s": float(spectrum_stats["spectrum_generation_time_s"]),
            "smoke_spectrum_ray_export_time_s": float(spectrum_stats["spectrum_ray_write_time_s"]),
            "smoke_spectrum_seconds_per_ray_all_attempted_groups": float(per_ray_spectrum_total),
            "smoke_spectrum_generation_seconds_per_ray_all_attempted_groups": float(per_ray_spectrum_generation),
            "smoke_spectrum_ray_export_seconds_per_ray": float(per_ray_spectrum_export),
            "smoke_total_with_spectra_s": float(catalog_time_s + spectrum_stats["spectrum_total_time_s"]),
            "million_meshless_catalog_time_s": float(catalog_time_s * ray_scale),
            "million_spectrum_time_s": float(per_ray_spectrum_total * target)
            if np.isfinite(per_ray_spectrum_total)
            else np.nan,
            "million_spectrum_generation_only_time_s": float(per_ray_spectrum_generation * target)
            if np.isfinite(per_ray_spectrum_generation)
            else np.nan,
            "million_spectrum_ray_export_upper_bound_time_s": float(per_ray_spectrum_export * target)
            if np.isfinite(per_ray_spectrum_export)
            else np.nan,
            "million_total_with_spectra_s": float(catalog_time_s * ray_scale + per_ray_spectrum_total * target)
            if np.isfinite(per_ray_spectrum_total)
            else np.nan,
            "million_meshless_catalog_human": seconds_to_human(catalog_time_s * ray_scale),
            "million_spectrum_human": seconds_to_human(per_ray_spectrum_total * target)
            if np.isfinite(per_ray_spectrum_total)
            else "nan",
            "million_spectrum_generation_only_human": seconds_to_human(per_ray_spectrum_generation * target)
            if np.isfinite(per_ray_spectrum_generation)
            else "nan",
            "million_spectrum_ray_export_upper_bound_human": seconds_to_human(per_ray_spectrum_export * target)
            if np.isfinite(per_ray_spectrum_export)
            else "nan",
            "million_total_with_spectra_human": seconds_to_human(catalog_time_s * ray_scale + per_ray_spectrum_total * target)
            if np.isfinite(per_ray_spectrum_total)
            else "nan",
            "catalog_size_bytes": int(catalog_size),
            "catalog_bytes_per_ray": float(catalog_size / args.nrays),
            "projected_million_catalog_size_gb": float(catalog_size * ray_scale / 1024**3),
            "spectrum_subset_output_size_bytes": int(directory_size(output_dir / "spectra")),
            "ray_file_subset_output_size_bytes": int(directory_size(output_dir / "rays" / "spectrum_subset")),
        }
        projection["projected_million_individual_spectrum_output_gb"] = (
            projection["spectrum_subset_output_size_bytes"] / max(spectrum_stats["spectra_subset_rays"], 1)
            * target / 1024**3
        )
        projection["projected_million_individual_ray_file_output_gb"] = (
            projection["ray_file_subset_output_size_bytes"] / max(spectrum_stats["spectra_subset_rays"], 1)
            * target / 1024**3
        )
        projection.update(catalog_summary)
        projection.update(spectrum_stats)
        projection["warnings"] = warnings
        projection["run_elapsed_s"] = float(time.perf_counter() - t_run)
        projection["run_status"] = run_status

        plot_timing_projection(output_dir / "plots" / "timing_projection.png", projection)
        write_json(output_dir / "summary" / "projection_summary.json", projection)
        write_csv(output_dir / "summary" / "projection_summary.csv", [projection])
        write_csv(output_dir / "summary" / "spectrum_summary.csv", spectrum_rows)
        write_json(output_dir / "summary" / "spectrum_summary.json", spectrum_rows)

        print("Meshless million-ray projection complete")
        print(f"Output: {output_dir}")
        print(f"Center ckpc/h: {center_code.tolist()} ({cutout_info['center_source']})")
        if "inferred_rvir_pkpc" in cutout_info:
            print(f"Inferred Rvir: {cutout_info['inferred_rvir_pkpc']:.2f} pkpc")
        print(f"Smoke catalog: {args.nrays} rays in {catalog_time_s:.3f} s")
        print(f"Projected 1e6 meshless catalog time: {projection['million_meshless_catalog_human']}")
        if not args.skip_spectra:
            print(
                "Spectrum subset: "
                f"{spectrum_stats['spectra_subset_rays']} rays, "
                f"{spectrum_stats['successful_spectra']} successful spectra, "
                f"{spectrum_stats['failed_or_skipped_spectra']} skipped/failed"
            )
            print(f"Projected 1e6 spectra time: {projection['million_spectrum_human']}")
            print(f"Projected total with spectra: {projection['million_total_with_spectra_human']}")
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    except Exception:
        run_status = "failed"
        error = traceback.format_exc()
        log_file.write_text(error)
        write_json(output_dir / "summary" / "failure.json", {"status": run_status, "traceback": error})
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
