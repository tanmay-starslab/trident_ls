#!/usr/bin/env python
"""Run meshless/LightRay diagnostics for multiple yt smoothing factors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)


def sf_label(value):
    return f"sf_{float(value):.1f}".replace(".", "p")


def read_rows(path, smoothing_factor):
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for row in csv.DictReader(handle):
            row["smoothing_factor"] = float(smoothing_factor)
            rows.append(row)
    return rows


def write_rows(rows, path):
    fieldnames = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def run_command(cmd, env):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env)


def diagnostic_command(args, subdir, smoothing_factor):
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "dev/meshless_tests/plot_meshless_vs_lightray_diagnostics.py"),
        "--dataset", args.dataset,
        "--output-dir", str(subdir),
        "--nrays", str(args.nrays),
        "--seed", str(args.seed),
        "--instrument", args.instrument,
        "--periodic", str(args.periodic).lower(),
        "--max-rays-for-spectra", str(args.max_rays_for_spectra),
        "--velocity-window-kms", str(args.velocity_window_kms),
        "--smoothing-factor", str(smoothing_factor),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.lines:
        cmd.extend(["--lines", *args.lines])
    return cmd


def debug_command(args, subdir, smoothing_factor):
    return [
        sys.executable,
        str(_REPO_ROOT / "dev/meshless_tests/debug_meshless_vs_lightray_geometry.py"),
        "--dataset", args.dataset,
        "--output-dir", str(subdir),
        "--smoothing-factor", str(smoothing_factor),
    ]


def plot_metric(ax, rows, y_light, y_mesh, title, ylabel):
    ray_ids = sorted({int(row["ray_id"]) for row in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, ray_id in enumerate(ray_ids):
        ray_rows = sorted(
            [row for row in rows if int(row["ray_id"]) == ray_id],
            key=lambda row: safe_float(row["smoothing_factor"]),
        )
        x = np.array([safe_float(row["smoothing_factor"]) for row in ray_rows])
        light = np.array([safe_float(row.get(y_light)) for row in ray_rows])
        mesh = np.array([safe_float(row.get(y_mesh)) for row in ray_rows])
        color = colors[index % len(colors)]
        ax.plot(x, light, "o-", color=color, label=f"ray {ray_id} LightRay")
        ax.plot(x, mesh, "s--", color=color, label=f"ray {ray_id} Meshless")
    ax.set_title(title)
    ax.set_xlabel("yt smoothing_factor")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_smoothing_summary(rows, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    axes = axes.ravel()
    plot_metric(axes[0], rows, "n_elements_lightray", "n_elements_meshless", "ray elements", "count")
    plot_metric(axes[1], rows, "dl_coverage_lightray", "dl_coverage_meshless", "sum(dl) / requested", "coverage")
    plot_metric(axes[2], rows, "log10_column_density_HI_lightray", "log10_column_density_HI_meshless", "H I column", "log10 N [cm^-2]")
    plot_metric(axes[3], rows, "log10_column_density_OVI_lightray", "log10_column_density_OVI_meshless", "O VI column", "log10 N [cm^-2]")
    plot_metric(axes[4], rows, "EW_HI_lightray", "EW_HI_meshless", "H I EW approx", "A")
    plot_metric(axes[5], rows, "rms_flux_diff_HI", "max_flux_diff_HI", "H I flux difference", "flux difference")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Smoothing Factor Sweep: LightRay vs Meshless")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ion_keys(rows):
    keys = set()
    prefix = "log10_column_density_"
    suffix = "_lightray"
    for row in rows:
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                keys.add(key[len(prefix):-len(suffix)])
    return sorted(keys)


def plot_column_delta_heatmap(rows, output_path):
    ions = ion_keys(rows)
    sfs = sorted({safe_float(row["smoothing_factor"]) for row in rows})
    values = np.full((len(ions), len(sfs)), np.nan)
    for i, ion in enumerate(ions):
        for j, sf in enumerate(sfs):
            sf_rows = [row for row in rows if safe_float(row["smoothing_factor"]) == sf]
            deltas = []
            for row in sf_rows:
                light = safe_float(row.get(f"log10_column_density_{ion}_lightray"))
                mesh = safe_float(row.get(f"log10_column_density_{ion}_meshless"))
                if np.isfinite(light) and np.isfinite(mesh):
                    deltas.append(mesh - light)
            if deltas:
                values[i, j] = float(np.mean(deltas))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(ions))), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=0.3)
    ax.set_xticks(np.arange(len(sfs)), [f"{sf:g}" for sf in sfs])
    ax.set_yticks(np.arange(len(ions)), ions)
    ax.set_xlabel("yt smoothing_factor")
    ax.set_title("Mean log10 column delta: Meshless - LightRay")
    for i in range(len(ions)):
        for j in range(len(sfs)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="dex")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="dev/meshless_tests/output/smoothing_factor_sweep")
    parser.add_argument("--smoothing-factors", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--nrays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--lines", nargs="+", default=None)
    parser.add_argument("--instrument", default="COS-G130M")
    parser.add_argument("--periodic", default="false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rays-for-spectra", type=int, default=3)
    parser.add_argument("--velocity-window-kms", type=float, default=1000.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

    diagnostics = []
    geometry = []
    cell = []
    for sf in args.smoothing_factors:
        subdir = output_dir / sf_label(sf)
        run_command(diagnostic_command(args, subdir, sf), env)
        run_command(debug_command(args, subdir, sf), env)
        diagnostics.extend(read_rows(subdir / "diagnostics_summary.csv", sf))
        geometry.extend(read_rows(subdir / "geometry_debug_summary.csv", sf))
        cell.extend(read_rows(subdir / "cell_level_debug_summary.csv", sf))

    write_rows(diagnostics, output_dir / "smoothing_factor_diagnostics_summary")
    write_rows(geometry, output_dir / "smoothing_factor_geometry_summary")
    write_rows(cell, output_dir / "smoothing_factor_cell_level_summary")
    plot_smoothing_summary(diagnostics, output_dir / "smoothing_factor_summary.png")
    plot_column_delta_heatmap(diagnostics, output_dir / "smoothing_factor_column_delta_heatmap.png")

    print("Smoothing-factor sweep complete")
    print(f"  output_dir: {output_dir}")
    print(f"  smoothing_factors: {', '.join(str(sf) for sf in args.smoothing_factors)}")
    print(f"  diagnostic rows: {len(diagnostics)}")
    print(f"  geometry rows: {len(geometry)}")
    print(f"  cell rows: {len(cell)}")


if __name__ == "__main__":
    main()
