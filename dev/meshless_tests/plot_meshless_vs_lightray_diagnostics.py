#!/usr/bin/env python
"""Plot LightRay vs meshless Voronoi ray and spectrum diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import traceback
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CONTEXT = Path("/private/tmp/trident_meshless_run_context")
_RUN_CONTEXT.mkdir(parents=True, exist_ok=True)
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(_RUN_CONTEXT)

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


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)
SPEED_OF_LIGHT_KMS = 299792.458
DEFAULT_LINES = [
    "H I 1216",
    "C II 1335",
    "C III 977",
    "C IV 1548",
    "C IV 1551",
    "N V 1239",
    "N V 1243",
    "O I 1302",
    "O VI 1032",
    "O VI 1038",
    "Si II 1260",
    "Si IV 1403",
]
LINE_GROUPS = {
    "HI": {"lines": ["H I 1216"], "wavelengths": [1215.67]},
    "CII": {"lines": ["C II 1335"], "wavelengths": [1334.53]},
    "CIII": {"lines": ["C III 977"], "wavelengths": [977.02]},
    "CIV": {"lines": ["C IV 1548", "C IV 1551"], "wavelengths": [1548.187, 1550.772]},
    "NV": {"lines": ["N V 1239", "N V 1243"], "wavelengths": [1238.821, 1242.804]},
    "OI": {"lines": ["O I 1302"], "wavelengths": [1302.17]},
    "OVI": {"lines": ["O VI 1032", "O VI 1038"], "wavelengths": [1031.912, 1037.613]},
    "SiII": {"lines": ["Si II 1260"], "wavelengths": [1260.42]},
    "SiIV": {"lines": ["Si IV 1403"], "wavelengths": [1402.77]},
    "MgII": {"lines": ["Mg II 2796", "Mg II 2803"], "wavelengths": [2796.35, 2803.53]},
}
ION_FIELDS = {
    "HI": ("gas", "H_p0_number_density"),
    "CII": ("gas", "C_p1_number_density"),
    "CIII": ("gas", "C_p2_number_density"),
    "CIV": ("gas", "C_p3_number_density"),
    "NV": ("gas", "N_p4_number_density"),
    "OI": ("gas", "O_p0_number_density"),
    "OVI": ("gas", "O_p5_number_density"),
    "SiII": ("gas", "Si_p1_number_density"),
    "SiIV": ("gas", "Si_p3_number_density"),
    "MgII": ("gas", "Mg_p1_number_density"),
}
ION_SPECS = {
    "HI": ("H", 1),
    "CII": ("C", 2),
    "CIII": ("C", 3),
    "CIV": ("C", 4),
    "NV": ("N", 5),
    "OI": ("O", 1),
    "OVI": ("O", 6),
    "SiII": ("Si", 2),
    "SiIV": ("Si", 4),
    "MgII": ("Mg", 2),
}
SUMMARY_COLUMNS = [
    "ray_id", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
    "ray_length_requested", "lightray_file", "meshless_file",
    "lightray_spectrum_file", "meshless_spectrum_file",
    "n_elements_lightray", "n_elements_meshless", "sum_dl_lightray",
    "sum_dl_meshless", "median_dl_lightray", "median_dl_meshless",
    "min_dl_lightray", "min_dl_meshless", "max_dl_lightray",
    "max_dl_meshless", "available_ion_fields",
    "total_column_like_HI_lightray", "total_column_like_HI_meshless",
    "total_column_like_OVI_lightray", "total_column_like_OVI_meshless",
    "total_column_like_MgII_lightray", "total_column_like_MgII_meshless",
    "log10_column_density_HI_lightray", "log10_column_density_HI_meshless",
    "log10_column_density_OVI_lightray", "log10_column_density_OVI_meshless",
    "EW_HI_lightray", "EW_HI_meshless", "EW_OVI_lightray",
    "EW_OVI_meshless", "EW_MgII_lightray", "EW_MgII_meshless",
    "rms_flux_diff_HI", "max_flux_diff_HI", "rms_flux_diff_OVI",
    "max_flux_diff_OVI", "rms_flux_diff_MgII", "max_flux_diff_MgII",
    "requested_length_cm", "dl_coverage_lightray", "dl_coverage_meshless",
    "l_min_lightray", "l_max_lightray", "l_min_meshless", "l_max_meshless",
    "status", "warning_or_error",
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def requested_ions(lines):
    ions = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            ion = f"{parts[0]} {parts[1]}"
            if ion not in ions:
                ions.append(ion)
    return ions or ["H I"]


def add_requested_ion_fields(ds, ions):
    if not ions:
        return ""
    try:
        trident.add_ion_fields(ds, ions=ions)
        return ""
    except Exception as exc:
        return f"trident.add_ion_fields failed for {ions}: {exc}"


def active_group_keys(lines):
    return [
        key for key, group in LINE_GROUPS.items()
        if any(line in lines for line in group["lines"])
    ]


def make_rays(ds, nrays, seed, ray_length_fraction=0.5):
    left = ds.domain_left_edge.to("code_length")
    right = ds.domain_right_edge.to("code_length")
    ad = ds.all_data()
    try:
        gas_positions = ad[("gas", "coordinates")].to("code_length")
    except Exception:
        gas_positions = ad[("PartType0", "Coordinates")].to("code_length")
    gas = gas_positions.d
    core_left = np.percentile(gas, 1, axis=0)
    core_right = np.percentile(gas, 99, axis=0)
    core_width = core_right - core_left
    total_length = float(ray_length_fraction * np.min(core_width))
    half_length = 0.5 * total_length
    rng = np.random.default_rng(seed)
    candidate_indices = rng.choice(len(gas_positions), min(len(gas_positions), max(nrays * 25, 128)), replace=False)
    centers = gas[candidate_indices]

    def centered_ray(center, direction, kind):
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        span = half_length * direction
        lo = np.minimum(core_left + np.abs(span), core_right - np.abs(span))
        hi = np.maximum(core_left + np.abs(span), core_right - np.abs(span))
        center = np.clip(np.asarray(center, dtype=float), lo, hi)
        start = np.clip(center - span, left.d, right.d)
        end = np.clip(center + span, left.d, right.d)
        return ds.arr(start, "code_length"), ds.arr(end, "code_length"), kind

    rays = [
        centered_ray(centers[0], [1.0, 0.0, 0.0], "x"),
        centered_ray(centers[1], [0.0, 1.0, 0.0], "y"),
        centered_ray(centers[2], [0.0, 0.0, 1.0], "z"),
        centered_ray(centers[3], [1.0, 1.0, 1.0], "diagonal"),
        centered_ray(centers[4], [1.0, 1.0, 0.0], "xy_diagonal"),
    ]
    while len(rays) < nrays:
        center = centers[len(rays) % len(centers)]
        direction = rng.normal(size=3)
        rays.append(centered_ray(center, direction, "random"))
    return rays[:nrays]


def ray_arrays(ray):
    ad = ray.all_data()
    dl = ad[("gas", "dl")].to("cm").d
    if ("gas", "l") in ray.derived_field_list:
        path_position = ad[("gas", "l")].to("cm").d
    else:
        path_position = np.cumsum(dl) - 0.5 * dl
    order = np.argsort(path_position)
    dl = dl[order]
    path_position = path_position[order]
    out = {"dl": dl, "pos": path_position, "ad": ad, "order": order}
    for name, field in [
        ("density", ("gas", "density")),
        ("temperature", ("gas", "temperature")),
        ("metallicity", ("gas", "metallicity")),
        ("velocity_los", ("gas", "velocity_los")),
        ("redshift_dopp", ("gas", "redshift_dopp")),
        ("redshift_eff", ("gas", "redshift_eff")),
    ]:
        out[name] = maybe_field(ad, field)
    return out


def maybe_field(ad, field):
    try:
        return ad[field].d
    except Exception:
        return None


def ray_stats(ray, label):
    ad = ray.all_data()
    dl = ad[("gas", "dl")].to("cm").d
    path_position = ad[("gas", "l")].to("cm").d if ("gas", "l") in ray.derived_field_list else np.cumsum(dl)
    return {
        f"n_elements_{label}": int(dl.size),
        f"sum_dl_{label}": float(np.sum(dl)),
        f"median_dl_{label}": float(np.median(dl)),
        f"min_dl_{label}": float(np.min(dl)),
        f"max_dl_{label}": float(np.max(dl)),
        f"l_min_{label}": float(np.min(path_position)),
        f"l_max_{label}": float(np.max(path_position)),
    }


def column_density(ray, ion_key):
    field = ION_FIELDS[ion_key]
    ad = ray.all_data()
    try:
        column = float((ad[field] * ad[("gas", "dl")]).sum().to("cm**-2").d)
    except Exception:
        return np.nan
    if column > 0.0:
        return float(np.log10(column))
    if column == 0.0:
        return -np.inf
    return np.nan


def available_ions(ray):
    available = []
    fields = set(ray.derived_field_list)
    for ion_key, field in ION_FIELDS.items():
        if field in fields:
            available.append(ion_key)
    return available


def spectrum_window(group_key, velocity_window_kms):
    wavelengths = np.asarray(LINE_GROUPS[group_key]["wavelengths"], dtype=float)
    factor = float(velocity_window_kms) / SPEED_OF_LIGHT_KMS
    return (
        float(np.min(wavelengths * (1.0 - factor))),
        float(np.max(wavelengths * (1.0 + factor))),
    )


def make_spectrum(ray, group_key, requested_lines, output_path, instrument, velocity_window_kms):
    group = LINE_GROUPS[group_key]
    group_lines = [line for line in group["lines"] if line in requested_lines]
    if not group_lines:
        return None, f"{group_key} not requested"
    try:
        lambda_min, lambda_max = spectrum_window(group_key, velocity_window_kms)
        sg = trident.SpectrumGenerator(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            dlambda=0.02,
        )
        if len(sg.line_database.parse_subset(group_lines)) == 0:
            return None, f"{group_key} lines not found in Trident line database: {group_lines}"
        sg.make_spectrum(ray, lines=group_lines)
    except Exception as exc:
        warnings.warn(f"{group_key}: spectrum failed in +/-{velocity_window_kms:g} km/s window: {exc}")
        return None, f"{group_key} spectrum failed: {exc}"
    sg.save_spectrum(str(output_path))
    lam = np.asarray(sg.lambda_field.d if hasattr(sg.lambda_field, "d") else sg.lambda_field)
    flux = np.asarray(sg.flux_field)
    tau = None if sg.tau_field is None else np.asarray(sg.tau_field)
    return {"lambda": lam, "flux": flux, "tau": tau, "file": str(output_path)}, None


def ew_approx(spec):
    if spec is None:
        return np.nan
    return float(np.trapezoid(1.0 - spec["flux"], spec["lambda"]))


def flux_diff(light_spec, mesh_spec):
    if light_spec is None or mesh_spec is None:
        return np.nan, np.nan, None, None
    light_lam = light_spec["lambda"]
    mesh_lam = mesh_spec["lambda"]
    if light_lam.size == mesh_lam.size and np.allclose(light_lam, mesh_lam):
        lam = light_lam
        diff = mesh_spec["flux"] - light_spec["flux"]
    else:
        lo = max(light_lam.min(), mesh_lam.min())
        hi = min(light_lam.max(), mesh_lam.max())
        if hi <= lo:
            return np.nan, np.nan, None, None
        lam = light_lam[(light_lam >= lo) & (light_lam <= hi)]
        diff = np.interp(lam, mesh_lam, mesh_spec["flux"]) - np.interp(lam, light_lam, light_spec["flux"])
    return float(np.sqrt(np.mean(diff**2))), float(np.max(np.abs(diff))), lam, diff


def plot_ray_diagnostics(ray_id, light_ray, mesh_ray, output_path):
    light = ray_arrays(light_ray)
    mesh = ray_arrays(mesh_ray)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        ("dl", "dl [cm]", False),
        ("density", "density", True),
        ("temperature", "temperature [K]", True),
        ("metallicity", "metallicity", False),
        ("velocity_los", "v_los [cm/s]", False),
        ("redshift_eff", "redshift_eff", False),
    ]
    for ax, (key, ylabel, logy) in zip(axes[:6], panels):
        for label, data in [("LightRay", light), ("Meshless", mesh)]:
            values = data[key]
            if values is None:
                continue
            if key != "dl":
                values = values[data["order"]]
            ax.plot(data["pos"], values, marker=".", lw=1, label=label)
        ax.set_xlabel("path position l [cm]")
        ax.set_ylabel(ylabel)
        if logy:
            set_log_if_positive(ax)
        ax.legend(fontsize=8)
    ax = axes[6]
    available = False
    for ion_key, field in ION_FIELDS.items():
        for label, data, style in [("LightRay", light, "-"), ("Meshless", mesh, "--")]:
            values = maybe_field(data["ad"], field)
            if values is None:
                continue
            values = values[data["order"]]
            available = True
            ax.plot(data["pos"], values, style, lw=1, label=f"{label} {ion_key}")
    ax.set_xlabel("path position l [cm]")
    ax.set_ylabel("ion number density")
    if available:
        set_log_if_positive(ax)
        ax.legend(fontsize=7)
    ax = axes[7]
    for ion_key, field in ION_FIELDS.items():
        for label, data, style in [("LightRay", light, "-"), ("Meshless", mesh, "--")]:
            values = maybe_field(data["ad"], field)
            if values is None:
                continue
            values = values[data["order"]]
            column = np.cumsum(values * data["dl"])
            log_column = np.full_like(column, np.nan, dtype=float)
            positive = column > 0.0
            log_column[positive] = np.log10(column[positive])
            ax.plot(data["pos"], log_column, style, lw=1, label=f"{label} {ion_key}")
    ax.set_xlabel("path position l [cm]")
    ax.set_ylabel("log10 cumulative N_ion [cm^-2]")
    if ax.lines:
        set_log_if_positive(ax)
        ax.legend(fontsize=7)
    fig.suptitle(f"Ray {ray_id:03d} Ray Diagnostics", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def set_log_if_positive(ax):
    ys = np.concatenate([line.get_ydata() for line in ax.lines]) if ax.lines else np.array([])
    if ys.size and np.all(ys[np.isfinite(ys)] > 0):
        ax.set_yscale("log")


def format_log_column(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    if np.isneginf(value):
        return "-inf"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def plot_spectrum_diagnostics(ray_id, light_specs, mesh_specs, output_path, diff_cache, group_keys, row):
    nrows = max(1, len(group_keys))
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 2.8 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])
    for row_index, group_key in enumerate(group_keys):
        flux_ax = axes[row_index, 0]
        residual_ax = axes[row_index, 1]
        for label, specs in [("LightRay", light_specs), ("Meshless", mesh_specs)]:
            spec = specs.get(group_key)
            if spec is None:
                continue
            flux_ax.plot(spec["lambda"], spec["flux"], lw=1, label=label)
        light_col = format_log_column(row.get(f"log10_column_density_{group_key}_lightray"))
        mesh_col = format_log_column(row.get(f"log10_column_density_{group_key}_meshless"))
        flux_ax.set_title(f"{group_key} flux  logN LightRay={light_col}, Meshless={mesh_col}")
        flux_ax.set_xlabel("wavelength [A]")
        flux_ax.set_ylabel("normalized flux")
        if flux_ax.lines:
            flux_ax.legend(fontsize=8)
        lam, diff = diff_cache.get(group_key, (None, None))
        if lam is not None:
            residual_ax.plot(lam, diff, lw=1, color="C3")
        residual_ax.axhline(0.0, color="0.3", lw=0.8)
        residual_ax.set_title(f"{group_key} residual: Meshless - LightRay")
        residual_ax.set_xlabel("wavelength [A]")
        residual_ax.set_ylabel("delta flux")
    fig.suptitle(f"Ray {ray_id:03d} Spectrum Diagnostics", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overview(rows, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    axes = axes.ravel()
    scatter_pair(axes[0], rows, "sum_dl_lightray", "sum_dl_meshless", "total dl [cm]", one_to_one=True)
    scatter_pair(axes[1], rows, "n_elements_lightray", "n_elements_meshless", "n elements", one_to_one=True)
    scatter_pair(axes[2], rows, "log10_column_density_HI_lightray", "log10_column_density_HI_meshless", "log10 HI column [cm^-2]", one_to_one=True)
    scatter_pair(axes[3], rows, "EW_HI_lightray", "EW_HI_meshless", "HI EW approx [A]", one_to_one=True)
    axes[4].bar([r["ray_id"] for r in rows], [safe_float(r.get("rms_flux_diff_HI")) for r in rows])
    axes[4].set_title("HI RMS flux difference")
    axes[4].set_xlabel("ray id")
    axes[5].bar([r["ray_id"] for r in rows], [safe_float(r.get("max_flux_diff_HI")) for r in rows])
    axes[5].set_title("HI max abs flux difference")
    axes[5].set_xlabel("ray id")
    fig.suptitle("All Rays Summary", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def scatter_pair(ax, rows, xkey, ykey, title, one_to_one=False):
    x = np.array([safe_float(r.get(xkey)) for r in rows])
    y = np.array([safe_float(r.get(ykey)) for r in rows])
    good = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[good], y[good])
    if one_to_one and np.any(good):
        lo = min(np.min(x[good]), np.min(y[good]))
        hi = max(np.max(x[good]), np.max(y[good]))
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
    ax.set_xlabel("LightRay")
    ax.set_ylabel("Meshless")
    ax.set_title(title)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def write_summary(rows, output_dir):
    csv_path = output_dir / "diagnostics_summary.csv"
    json_path = output_dir / "diagnostics_summary.json"
    fieldnames = SUMMARY_COLUMNS[:]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extras)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="/private/tmp/trident_meshless_diagnostics")
    parser.add_argument("--nrays", type=int, default=8)
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--lines", nargs="+", default=DEFAULT_LINES)
    parser.add_argument("--instrument", default="COS-G130M")
    parser.add_argument("--periodic", type=parse_bool, default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rays-for-spectra", type=int, default=8)
    parser.add_argument("--ray-length-fraction", type=float, default=0.5)
    parser.add_argument("--velocity-window-kms", type=float, default=1000.0)
    parser.add_argument("--smoothing-factor", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = yt.load(args.dataset, smoothing_factor=args.smoothing_factor)
    rays = make_rays(ds, args.nrays, args.seed, args.ray_length_fraction)
    field_lines = requested_ions(args.lines)
    group_keys = active_group_keys(args.lines)
    ion_field_warning = add_requested_ion_fields(ds, field_lines)
    rows = []
    successes = 0
    failures = 0
    skipped = []

    for ray_id, (start, end, ray_kind) in enumerate(rays):
        row = {
            "ray_id": ray_id,
            "ray_kind": ray_kind,
            "start_x": float(start.to("code_length").d[0]),
            "start_y": float(start.to("code_length").d[1]),
            "start_z": float(start.to("code_length").d[2]),
            "end_x": float(end.to("code_length").d[0]),
            "end_y": float(end.to("code_length").d[1]),
            "end_z": float(end.to("code_length").d[2]),
            "ray_length_requested": float(np.sqrt(np.sum((end.to("code_length").d - start.to("code_length").d) ** 2))),
            "status": "ok",
            "warning_or_error": ion_field_warning,
            "smoothing_factor": args.smoothing_factor,
        }
        row["requested_length_cm"] = float(
            ds.quan(row["ray_length_requested"], "code_length").to("cm").d
        )
        try:
            light_file = output_dir / f"ray_{ray_id:03d}_lightray.h5"
            mesh_file = output_dir / f"ray_{ray_id:03d}_meshless.h5"
            row["lightray_file"] = str(light_file)
            row["meshless_file"] = str(mesh_file)
            light_ray, mesh_ray, warning_text = make_ray_pair(
                ds, start, end, field_lines, light_file, mesh_file, args.periodic
            )
            if warning_text:
                row["warning_or_error"] = append_warning(
                    row["warning_or_error"], warning_text
                )
            ion_warnings = ensure_ion_fields(light_ray, mesh_ray, args.lines)
            if ion_warnings:
                row["warning_or_error"] = append_warning(
                    row["warning_or_error"], "; ".join(ion_warnings)
                )
            row.update(ray_stats(light_ray, "lightray"))
            row.update(ray_stats(mesh_ray, "meshless"))
            row["dl_coverage_lightray"] = (
                row["sum_dl_lightray"] / row["requested_length_cm"]
            )
            row["dl_coverage_meshless"] = (
                row["sum_dl_meshless"] / row["requested_length_cm"]
            )
            ions = sorted(set(available_ions(light_ray)) | set(available_ions(mesh_ray)))
            row["available_ion_fields"] = ",".join(ions)
            for ion_key in group_keys:
                light_column = column_density(light_ray, ion_key)
                mesh_column = column_density(mesh_ray, ion_key)
                row[f"column_density_{ion_key}_lightray"] = light_column
                row[f"column_density_{ion_key}_meshless"] = mesh_column
                row[f"log10_column_density_{ion_key}_lightray"] = light_column
                row[f"log10_column_density_{ion_key}_meshless"] = mesh_column
                row[f"total_column_like_{ion_key}_lightray"] = light_column
                row[f"total_column_like_{ion_key}_meshless"] = mesh_column

            plot_ray_diagnostics(
                ray_id, light_ray, mesh_ray, output_dir / f"ray_{ray_id:03d}_ray_diagnostics.png"
            )

            light_specs = {}
            mesh_specs = {}
            if ray_id < args.max_rays_for_spectra:
                for ion_key in group_keys:
                    light_spec, light_err = make_spectrum(
                        light_ray, ion_key, args.lines,
                        output_dir / f"ray_{ray_id:03d}_lightray_{ion_key}_spectrum.h5",
                        args.instrument,
                        args.velocity_window_kms,
                    )
                    mesh_spec, mesh_err = make_spectrum(
                        mesh_ray, ion_key, args.lines,
                        output_dir / f"ray_{ray_id:03d}_meshless_{ion_key}_spectrum.h5",
                        args.instrument,
                        args.velocity_window_kms,
                    )
                    if light_err or mesh_err:
                        message = f"ray {ray_id} {ion_key}: {light_err or ''} {mesh_err or ''}".strip()
                        skipped.append(message)
                        row["warning_or_error"] = append_warning(
                            row["warning_or_error"], message
                        )
                    light_specs[ion_key] = light_spec
                    mesh_specs[ion_key] = mesh_spec
                    row[f"EW_{ion_key}_lightray"] = ew_approx(light_spec)
                    row[f"EW_{ion_key}_meshless"] = ew_approx(mesh_spec)
                    rms, max_abs, lam, diff = flux_diff(light_spec, mesh_spec)
                    row[f"rms_flux_diff_{ion_key}"] = rms
                    row[f"max_flux_diff_{ion_key}"] = max_abs
                row["lightray_spectrum_file"] = ";".join(
                    spec["file"] for spec in light_specs.values() if spec is not None
                )
                row["meshless_spectrum_file"] = ";".join(
                    spec["file"] for spec in mesh_specs.values() if spec is not None
                )
                diff_cache = {
                    ion_key: flux_diff(light_specs.get(ion_key), mesh_specs.get(ion_key))[2:]
                    for ion_key in group_keys
                }
                plot_spectrum_diagnostics(
                    ray_id, light_specs, mesh_specs,
                    output_dir / f"ray_{ray_id:03d}_spectrum_diagnostics.png",
                    diff_cache,
                    group_keys,
                    row,
                )
            else:
                skipped.append(f"ray {ray_id}: spectra skipped by --max-rays-for-spectra")
            successes += 1
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["warning_or_error"] = f"{exc}\n{traceback.format_exc()}"
        rows.append(row)

    if any(row.get("status") == "ok" for row in rows):
        plot_overview([row for row in rows if row.get("status") == "ok"], output_dir / "all_rays_summary.png")
    csv_path, json_path = write_summary(rows, output_dir)
    print("Meshless diagnostics complete")
    print(f"  output_dir: {output_dir}")
    print(f"  rays requested: {args.nrays}")
    print(f"  successful rays: {successes}")
    print(f"  failed rays: {failures}")
    print(f"  summary_csv: {csv_path}")
    print(f"  summary_json: {json_path}")
    if skipped:
        print("  warnings/skips:")
        for item in skipped[:12]:
            print(f"    - {item}")
        if len(skipped) > 12:
            print(f"    - ... {len(skipped) - 12} more")


def make_ray_pair(ds, start, end, field_lines, light_file, mesh_file, periodic):
    warning_text = ""
    try:
        light_ray = trident.make_simple_ray(
            ds, start_position=start, end_position=end,
            lines=field_lines, data_filename=str(light_file)
        )
        mesh_ray = trident.make_meshless_voronoi_ray(
            ds, start_position=start, end_position=end,
            lines=field_lines, data_filename=str(mesh_file), periodic=periodic
        )
        return light_ray, mesh_ray, warning_text
    except Exception as exc:
        if field_lines != ["H I"]:
            warning_text = f"Full ion ray generation failed; retried with H I only: {exc}"
            light_ray = trident.make_simple_ray(
                ds, start_position=start, end_position=end,
                lines=["H I"], data_filename=str(light_file)
            )
            mesh_ray = trident.make_meshless_voronoi_ray(
                ds, start_position=start, end_position=end,
                lines=["H I"], data_filename=str(mesh_file), periodic=periodic
            )
            return light_ray, mesh_ray, warning_text
        raise


def ensure_ion_fields(light_ray, mesh_ray, requested_lines):
    warnings_out = []
    requested = set()
    for group_key, group in LINE_GROUPS.items():
        if any(line in requested_lines for line in group["lines"]):
            requested.add(group_key)
    for ray_label, ray in [("LightRay", light_ray), ("Meshless", mesh_ray)]:
        for group_key in sorted(requested):
            field = ION_FIELDS[group_key]
            if field in ray.derived_field_list:
                continue
            atom, ion = ION_SPECS[group_key]
            try:
                trident.add_ion_number_density_field(atom, ion, ray)
            except Exception as exc:
                warnings_out.append(f"{ray_label} {group_key} ion field unavailable: {exc}")
    return warnings_out


def append_warning(existing, addition):
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}; {addition}"


if __name__ == "__main__":
    main()
