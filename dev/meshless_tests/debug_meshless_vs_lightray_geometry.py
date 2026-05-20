#!/usr/bin/env python
"""Deeper geometry debugging for LightRay versus meshless Voronoi rays."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from scipy.spatial import cKDTree


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)


def load_summary(output_dir):
    path = output_dir / "diagnostics_summary.csv"
    return list(csv.DictReader(path.open()))


def ray_data(ray_path):
    ray = yt.load(str(ray_path))
    ad = ray.all_data()
    data = {
        "dl": ad[("gas", "dl")].to("cm").d,
        "l": ad[("gas", "l")].to("cm").d,
        "x": ad[("gas", "x")].to("cm").d,
        "y": ad[("gas", "y")].to("cm").d,
        "z": ad[("gas", "z")].to("cm").d,
    }
    optional_fields = {
        "meshless_cell_index": ("gas", "meshless_cell_index"),
        "density": ("gas", "density"),
        "temperature": ("gas", "temperature"),
        "metallicity": ("gas", "metallicity"),
    }
    for name, field in optional_fields.items():
        if field in ray.derived_field_list:
            try:
                values = ad[field]
                data[name] = values.d if hasattr(values, "d") else np.asarray(values)
            except Exception:
                pass
    data["order"] = np.argsort(data["l"])
    return data


def percentiles(values):
    if len(values) == 0:
        return {f"p{p}": np.nan for p in [0, 1, 5, 50, 95, 99, 100]}
    pts = np.percentile(values, [0, 1, 5, 50, 95, 99, 100])
    return dict(zip(["p0", "p1", "p5", "p50", "p95", "p99", "p100"], pts))


def projection_cm(ds, row, data):
    start_code = np.array([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])])
    end_code = np.array([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])])
    start_cm = ds.arr(start_code, "code_length").to("cm").d
    end_cm = ds.arr(end_code, "code_length").to("cm").d
    ray_vec = end_cm - start_cm
    length = np.linalg.norm(ray_vec)
    direction = ray_vec / length
    pos = np.column_stack([data["x"], data["y"], data["z"]])
    return np.dot(pos - start_cm, direction), length


def summarize_one(ds, row, label, data):
    requested_cm = float(row["requested_length_cm"])
    projected, projected_length = projection_cm(ds, row, data)
    dl_pct = percentiles(data["dl"])
    ordered_l = data["l"][data["order"]]
    out = {
        "ray_id": int(row["ray_id"]),
        "ray_kind": row.get("ray_kind", ""),
        "type": label,
        "requested_length_cm": requested_cm,
        "projected_endpoint_length_cm": projected_length,
        "n_elements": int(len(data["dl"])),
        "sum_dl_cm": float(np.sum(data["dl"])),
        "sum_dl_over_requested": float(np.sum(data["dl"]) / requested_cm),
        "l_min_cm": float(np.min(data["l"])),
        "l_max_cm": float(np.max(data["l"])),
        "l_span_over_requested": float((np.max(data["l"]) - np.min(data["l"])) / requested_cm),
        "projected_min_cm": float(np.min(projected)),
        "projected_max_cm": float(np.max(projected)),
        "projected_span_over_requested": float((np.max(projected) - np.min(projected)) / requested_cm),
        "l_first_sorted_cm": float(ordered_l[0]),
        "l_last_sorted_cm": float(ordered_l[-1]),
    }
    for key, value in dl_pct.items():
        out[f"dl_{key}_cm"] = float(value)
    return out


def plot_one(row, light, mesh, ds, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    requested_cm = float(row["requested_length_cm"])
    for label, data, color in [("LightRay", light, "C0"), ("Meshless", mesh, "C1")]:
        order = data["order"]
        projected, _ = projection_cm(ds, row, data)
        axes[0].plot(data["l"][order], data["dl"][order], ".", ms=3, label=label, color=color)
        axes[1].hist(data["dl"], bins=40, histtype="step", label=label, color=color)
        axes[2].plot(data["l"][order], projected[order], ".", ms=3, label=label, color=color)
        axes[3].plot(data["l"][order], np.cumsum(data["dl"][order]), label=label, color=color)
        axes[4].plot(projected[order], data["dl"][order], ".", ms=3, label=label, color=color)
    axes[0].set_title("dl vs path-position l")
    axes[0].set_xlabel("l [cm]")
    axes[0].set_ylabel("dl [cm]")
    axes[0].set_yscale("log")
    axes[1].set_title("dl distribution")
    axes[1].set_xlabel("dl [cm]")
    axes[1].set_yscale("log")
    axes[2].plot([0, requested_cm], [0, requested_cm], "k--", lw=0.8)
    axes[2].set_title("projected xyz position vs l")
    axes[2].set_xlabel("l [cm]")
    axes[2].set_ylabel("projected position [cm]")
    axes[3].axhline(requested_cm, color="k", ls="--", lw=0.8, label="requested length")
    axes[3].set_title("cumsum(dl) vs l")
    axes[3].set_xlabel("l [cm]")
    axes[3].set_ylabel("cumulative dl [cm]")
    axes[4].set_title("dl vs projected position")
    axes[4].set_xlabel("projected position [cm]")
    axes[4].set_ylabel("dl [cm]")
    axes[4].set_yscale("log")
    axes[5].axis("off")
    text = [
        f"ray {row['ray_id']} ({row.get('ray_kind', '')})",
        f"requested_length_cm = {requested_cm:.4e}",
        f"LightRay sum(dl)/requested = {np.sum(light['dl'])/requested_cm:.4f}",
        f"Meshless sum(dl)/requested = {np.sum(mesh['dl'])/requested_cm:.4f}",
        f"LightRay n = {len(light['dl'])}",
        f"Meshless n = {len(mesh['dl'])}",
        "Interpretation:",
        "LightRay dl is sampled material/kernel path,",
        "not guaranteed to tile the full line.",
        "Meshless Voronoi dl tiles the full ray.",
    ]
    axes[5].text(0.0, 1.0, "\n".join(text), va="top", family="monospace")
    for ax in axes[:5]:
        ax.legend(fontsize=8)
    fig.suptitle(f"Ray {int(row['ray_id']):03d} Geometry Debug")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(rows, output_dir):
    csv_path = output_dir / "geometry_debug_summary.csv"
    json_path = output_dir / "geometry_debug_summary.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def write_table(rows, output_dir, stem):
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def load_gas_context(ds):
    ad = ds.all_data()
    try:
        positions = ad[("gas", "coordinates")].to("code_length").d
        position_field = "gas,coordinates"
    except Exception:
        positions = ad[("PartType0", "Coordinates")].to("code_length").d
        position_field = "PartType0,Coordinates"
    smoothing = None
    smoothing_field = ""
    for field in [("gas", "smoothing_length"), ("PartType0", "smoothing_length")]:
        try:
            smoothing = ad[field].to("code_length").d
            smoothing_field = ",".join(field)
            break
        except Exception:
            continue
    return {
        "positions": positions,
        "position_field": position_field,
        "smoothing_length": smoothing,
        "smoothing_length_field": smoothing_field,
        "min": np.min(positions, axis=0),
        "max": np.max(positions, axis=0),
        "p1": np.percentile(positions, 1, axis=0),
        "p99": np.percentile(positions, 99, axis=0),
        "tree": cKDTree(positions),
    }


def inside_box(point, lo, hi):
    return bool(np.all(point >= lo) and np.all(point <= hi))


def segment_box_fraction(start, end, lo, hi):
    direction = end - start
    t0 = 0.0
    t1 = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-14:
            if start[axis] < lo[axis] or start[axis] > hi[axis]:
                return 0.0
            continue
        inv = 1.0 / direction[axis]
        ta = (lo[axis] - start[axis]) * inv
        tb = (hi[axis] - start[axis]) * inv
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return 0.0
    return max(0.0, t1 - t0)


def endpoint_debug(rows, ds, gas, output_dir):
    out = []
    gas_span = gas["max"] - gas["min"]
    for row in rows:
        start = np.array([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])])
        end = np.array([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])])
        center = 0.5 * (start + end)
        length = float(np.linalg.norm(end - start))
        start_dist, start_idx = gas["tree"].query(start)
        end_dist, end_idx = gas["tree"].query(end)
        center_dist, center_idx = gas["tree"].query(center)
        out.append({
            "ray_id": int(row["ray_id"]),
            "ray_kind": row.get("ray_kind", ""),
            "position_field": gas["position_field"],
            "gas_count": int(len(gas["positions"])),
            "gas_min_x": float(gas["min"][0]),
            "gas_min_y": float(gas["min"][1]),
            "gas_min_z": float(gas["min"][2]),
            "gas_max_x": float(gas["max"][0]),
            "gas_max_y": float(gas["max"][1]),
            "gas_max_z": float(gas["max"][2]),
            "gas_span_x": float(gas_span[0]),
            "gas_span_y": float(gas_span[1]),
            "gas_span_z": float(gas_span[2]),
            "ray_length_code": length,
            "ray_length_over_max_gas_span": float(length / np.max(gas_span)),
            "start_inside_gas_bbox": inside_box(start, gas["min"], gas["max"]),
            "end_inside_gas_bbox": inside_box(end, gas["min"], gas["max"]),
            "center_inside_gas_bbox": inside_box(center, gas["min"], gas["max"]),
            "start_inside_gas_p1_p99_bbox": inside_box(start, gas["p1"], gas["p99"]),
            "end_inside_gas_p1_p99_bbox": inside_box(end, gas["p1"], gas["p99"]),
            "center_inside_gas_p1_p99_bbox": inside_box(center, gas["p1"], gas["p99"]),
            "fraction_inside_gas_bbox": segment_box_fraction(start, end, gas["min"], gas["max"]),
            "fraction_inside_gas_p1_p99_bbox": segment_box_fraction(start, end, gas["p1"], gas["p99"]),
            "nearest_gas_distance_start_code": float(start_dist),
            "nearest_gas_distance_end_code": float(end_dist),
            "nearest_gas_distance_center_code": float(center_dist),
            "nearest_gas_index_start": int(start_idx),
            "nearest_gas_index_end": int(end_idx),
            "nearest_gas_index_center": int(center_idx),
        })
    csv_path = output_dir / "endpoint_debug_summary.csv"
    json_path = output_dir / "endpoint_debug_summary.json"
    fieldnames = sorted({key for row in out for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
    with json_path.open("w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    plot_endpoint_context(rows, gas, output_dir / "endpoint_context_debug.png")
    return csv_path, json_path


def plot_endpoint_context(rows, gas, output_path):
    rng = np.random.default_rng(398784)
    positions = gas["positions"]
    sample_size = min(len(positions), 25000)
    sample = positions[rng.choice(len(positions), sample_size, replace=False)]
    projections = [
        (0, 1, "x", "y"),
        (0, 2, "x", "z"),
        (1, 2, "y", "z"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (i, j, xlabel, ylabel) in zip(axes, projections):
        ax.plot(sample[:, i], sample[:, j], ".", ms=0.3, alpha=0.18, color="0.35")
        for row in rows:
            start = np.array([float(row["start_x"]), float(row["start_y"]), float(row["start_z"])])
            end = np.array([float(row["end_x"]), float(row["end_y"]), float(row["end_z"])])
            ray_id = int(row["ray_id"])
            ax.plot([start[i], end[i]], [start[j], end[j]], lw=1.2, label=f"ray {ray_id}")
            ax.plot(start[i], start[j], "o", ms=3)
            ax.plot(end[i], end[j], "s", ms=3)
        ax.set_xlabel(f"{xlabel} [code_length]")
        ax.set_ylabel(f"{ylabel} [code_length]")
        ax.set_title(f"Gas sample and ray endpoints: {xlabel}-{ylabel}")
    axes[0].legend(fontsize=7)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ray_positions_code(ds, data):
    pos_cm = np.column_stack([data["x"], data["y"], data["z"]])
    return ds.arr(pos_cm, "cm").to("code_length").d


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def finite_percentile(values, pct):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, pct))


def count_light_samples_per_mesh_segment(light, mesh):
    light_l = np.asarray(light["l"], dtype=float)
    mesh_l = np.asarray(mesh["l"], dtype=float)
    mesh_dl = np.asarray(mesh["dl"], dtype=float)
    counts = []
    for center, width in zip(mesh_l, mesh_dl):
        lo = center - 0.5 * width
        hi = center + 0.5 * width
        counts.append(int(np.count_nonzero((light_l >= lo) & (light_l <= hi))))
    return np.asarray(counts, dtype=int)


def cell_level_summary_one(ds, row, light, mesh, gas):
    light_pos_code = ray_positions_code(ds, light)
    light_dist, light_idx = gas["tree"].query(light_pos_code)
    mesh_idx = np.asarray(mesh.get("meshless_cell_index", []), dtype=int)
    mesh_idx = mesh_idx[mesh_idx >= 0]
    unique_light = np.unique(light_idx)
    unique_mesh = np.unique(mesh_idx) if mesh_idx.size else np.array([], dtype=int)
    shared = np.intersect1d(unique_light, unique_mesh)
    counts_per_mesh = count_light_samples_per_mesh_segment(light, mesh)
    out = {
        "ray_id": int(row["ray_id"]),
        "ray_kind": row.get("ray_kind", ""),
        "n_lightray_elements": int(len(light["dl"])),
        "n_meshless_elements": int(len(mesh["dl"])),
        "element_count_ratio_meshless_over_lightray": float(len(mesh["dl"]) / len(light["dl"])),
        "n_unique_lightray_nearest_sites": int(len(unique_light)),
        "n_unique_meshless_sites": int(len(unique_mesh)),
        "lightray_duplicate_factor_elements_per_nearest_site": float(len(light["dl"]) / max(len(unique_light), 1)),
        "meshless_duplicate_factor_elements_per_site": float(len(mesh["dl"]) / max(len(unique_mesh), 1)),
        "n_shared_sites_lightray_nearest_vs_meshless": int(len(shared)),
        "fraction_lightray_nearest_sites_in_meshless_path": float(len(shared) / max(len(unique_light), 1)),
        "fraction_meshless_sites_seen_by_lightray_nearest": float(len(shared) / max(len(unique_mesh), 1)),
        "median_lightray_nearest_distance_code": finite_median(light_dist),
        "p95_lightray_nearest_distance_code": finite_percentile(light_dist, 95),
        "median_lightray_samples_per_meshless_segment": finite_median(counts_per_mesh),
        "max_lightray_samples_per_meshless_segment": int(np.max(counts_per_mesh)) if counts_per_mesh.size else 0,
        "meshless_segments_with_no_lightray_samples": int(np.count_nonzero(counts_per_mesh == 0)),
        "fraction_meshless_segments_with_no_lightray_samples": float(np.count_nonzero(counts_per_mesh == 0) / max(len(counts_per_mesh), 1)),
    }
    smoothing = gas.get("smoothing_length")
    if smoothing is not None:
        light_smoothing = smoothing[light_idx]
        mesh_smoothing = smoothing[mesh_idx] if mesh_idx.size else np.array([])
        out.update({
            "smoothing_length_field": gas.get("smoothing_length_field", ""),
            "median_lightray_nearest_smoothing_code": finite_median(light_smoothing),
            "median_meshless_smoothing_code": finite_median(mesh_smoothing),
            "median_lightray_dl_over_smoothing": finite_median(ds.arr(light["dl"], "cm").to("code_length").d / light_smoothing),
            "median_meshless_dl_over_smoothing": finite_median(ds.arr(mesh["dl"], "cm").to("code_length").d / mesh_smoothing) if mesh_smoothing.size else np.nan,
        })
    return out, light_dist, light_idx, mesh_idx, counts_per_mesh


def plot_cell_level_debug(row, light, mesh, ds, gas, output_path):
    summary, light_dist, light_idx, mesh_idx, counts_per_mesh = cell_level_summary_one(
        ds, row, light, mesh, gas
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    bins = 40
    axes[0].hist(light["dl"], bins=bins, histtype="step", label="LightRay", color="C0")
    axes[0].hist(mesh["dl"], bins=bins, histtype="step", label="Meshless", color="C1")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("dl [cm]")
    axes[0].set_title("element path-length distribution")
    axes[0].legend(fontsize=8)

    order = light["order"]
    axes[1].plot(light["l"][order], light_dist[order], ".", ms=3, color="C0")
    axes[1].set_xlabel("LightRay l [cm]")
    axes[1].set_ylabel("nearest gas-site distance [code_length]")
    axes[1].set_title("LightRay sample to nearest gas generator")

    axes[2].hist(counts_per_mesh, bins=np.arange(0, max(2, np.max(counts_per_mesh) + 2)) - 0.5)
    axes[2].set_xlabel("LightRay samples inside meshless segment")
    axes[2].set_ylabel("meshless segments")
    axes[2].set_title("LightRay occupancy per Voronoi segment")

    smoothing = gas.get("smoothing_length")
    if smoothing is not None and mesh_idx.size:
        axes[3].hist(smoothing[light_idx], bins=bins, histtype="step", label="LightRay nearest sites")
        axes[3].hist(smoothing[mesh_idx], bins=bins, histtype="step", label="Meshless sites")
        axes[3].set_xlabel("smoothing length [code_length]")
        axes[3].set_title("nearest-site smoothing lengths")
        axes[3].legend(fontsize=8)
    else:
        axes[3].axis("off")
        axes[3].text(0.0, 1.0, "No smoothing-length field available", va="top")

    if mesh_idx.size:
        light_unique = np.unique(light_idx)
        mesh_unique = np.unique(mesh_idx)
        axes[4].bar(
            ["LightRay nearest", "Meshless", "shared"],
            [len(light_unique), len(mesh_unique), len(np.intersect1d(light_unique, mesh_unique))],
            color=["C0", "C1", "0.4"],
        )
        axes[4].set_title("unique gas-site overlap")
        axes[4].set_ylabel("site count")
    else:
        axes[4].axis("off")
        axes[4].text(0.0, 1.0, "Meshless cell indices unavailable", va="top")

    axes[5].axis("off")
    text = [
        f"ray {summary['ray_id']} ({summary.get('ray_kind', '')})",
        f"elements: LightRay={summary['n_lightray_elements']} Meshless={summary['n_meshless_elements']}",
        f"ratio Meshless/LightRay={summary['element_count_ratio_meshless_over_lightray']:.3f}",
        f"unique sites: LightRay-nearest={summary['n_unique_lightray_nearest_sites']} Meshless={summary['n_unique_meshless_sites']}",
        f"shared-site fractions: LR->{summary['fraction_lightray_nearest_sites_in_meshless_path']:.3f}, Mesh->{summary['fraction_meshless_sites_seen_by_lightray_nearest']:.3f}",
        f"median nearest distance={summary['median_lightray_nearest_distance_code']:.4g} code_length",
        f"median LR samples per mesh segment={summary['median_lightray_samples_per_meshless_segment']:.3g}",
        f"mesh segments with no LR sample={summary['meshless_segments_with_no_lightray_samples']}",
    ]
    axes[5].text(0.0, 1.0, "\n".join(text), va="top", family="monospace", fontsize=9)
    fig.suptitle(f"Ray {int(row['ray_id']):03d} Cell-Level Debug")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def write_interpretation_report(rows, output_dir):
    lines = [
        "# Meshless vs LightRay cell-level interpretation",
        "",
        "LightRay and meshless Voronoi rays are not currently sampling the same geometric object.",
        "The meshless path is a Voronoi cell walker: one accepted segment per generating site along a disjoint tessellation.",
        "The yt/Trident LightRay path through this TNG particle/cutout dataset is a dataset-frontend sampling of gas material, not the same Voronoi face-to-face walk.",
        "",
        "Consequences:",
        "- Element counts are not expected to agree. LightRay can create many ray samples associated with nearby gas sites or kernel/material intersections, while meshless keeps one segment per Voronoi cell.",
        "- LightRay sum(dl) can be below the requested geometric length on this particle cutout, while meshless sum(dl) is constructed to tile the requested line.",
        "- Spectral and column differences therefore mix field differences with geometric weighting differences.",
        "",
        "To make differences very small, the two spectra must use the same geometry and weights.",
        "Practical options are: compare SpectrumGenerator on two files built from the same meshless segment table; implement a LightRay-like kernel sampler for the meshless path; use a yt frontend that exposes true AREPO Voronoi cells; or restrict comparison to positions where both methods sample the same set of gas sites and path weights.",
        "",
        "Per-ray headline metrics:",
    ]
    for row in rows:
        lines.append(
            f"- ray {row['ray_id']}: n LightRay={row['n_lightray_elements']}, "
            f"n Meshless={row['n_meshless_elements']}, "
            f"unique nearest LightRay={row['n_unique_lightray_nearest_sites']}, "
            f"unique Meshless={row['n_unique_meshless_sites']}, "
            f"shared LR-nearest fraction={row['fraction_lightray_nearest_sites_in_meshless_path']:.3f}, "
            f"shared Meshless fraction={row['fraction_meshless_sites_seen_by_lightray_nearest']:.3f}"
        )
    path = output_dir / "cell_level_debug_interpretation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="dev/meshless_tests/output")
    parser.add_argument("--smoothing-factor", type=float, default=2.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    ds = yt.load(args.dataset, smoothing_factor=args.smoothing_factor)
    rows = load_summary(output_dir)
    gas = load_gas_context(ds)
    out_rows = []
    cell_rows = []
    for row in rows:
        ray_id = int(row["ray_id"])
        light = ray_data(output_dir / f"ray_{ray_id:03d}_lightray.h5")
        mesh = ray_data(output_dir / f"ray_{ray_id:03d}_meshless.h5")
        out_rows.append(summarize_one(ds, row, "lightray", light))
        out_rows.append(summarize_one(ds, row, "meshless", mesh))
        plot_one(row, light, mesh, ds, output_dir / f"ray_{ray_id:03d}_geometry_debug.png")
        cell_rows.append(
            plot_cell_level_debug(
                row, light, mesh, ds, gas,
                output_dir / f"ray_{ray_id:03d}_cell_level_debug.png",
            )
        )
    csv_path, json_path = write_outputs(out_rows, output_dir)
    endpoint_csv, endpoint_json = endpoint_debug(rows, ds, gas, output_dir)
    cell_csv, cell_json = write_table(cell_rows, output_dir, "cell_level_debug_summary")
    report_path = write_interpretation_report(cell_rows, output_dir)
    print("Geometry debug complete")
    print(f"  output_dir: {output_dir}")
    print(f"  rows: {len(out_rows)}")
    print(f"  summary_csv: {csv_path}")
    print(f"  summary_json: {json_path}")
    print(f"  endpoint_csv: {endpoint_csv}")
    print(f"  endpoint_json: {endpoint_json}")
    print(f"  cell_level_csv: {cell_csv}")
    print(f"  cell_level_json: {cell_json}")
    print(f"  interpretation_report: {report_path}")


if __name__ == "__main__":
    main()
