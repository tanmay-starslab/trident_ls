#!/usr/bin/env python
"""Benchmark LightRay and meshless Voronoi ray generation timing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def load_positions(ds):
    ad = ds.all_data()
    try:
        return ad[("gas", "coordinates")].to("code_length")
    except Exception:
        return ad[("PartType0", "Coordinates")].to("code_length")


def make_long_rays(
        ds, nrays, seed, ray_length_fraction, min_length_kpc=None,
        max_length_kpc=None, spherical=True):
    left = ds.domain_left_edge.to("code_length").d
    right = ds.domain_right_edge.to("code_length").d
    gas_pos = load_positions(ds)
    gas = gas_pos.to("code_length").d
    gas_kpc = gas_pos.to("kpc").d
    core_left = np.percentile(gas_kpc, 2, axis=0)
    core_right = np.percentile(gas_kpc, 98, axis=0)
    core_width = core_right - core_left
    gas_center = np.median(gas_kpc, axis=0)
    radius = np.linalg.norm(gas_kpc - gas_center[None, :], axis=1)
    radius_limit = float(np.percentile(radius, 95 if spherical else 98))
    if max_length_kpc is None:
        max_length_kpc = float(ray_length_fraction * min(2.0 * radius_limit, np.min(core_width)))
    if min_length_kpc is None:
        min_length_kpc = min(100.0, 0.5 * max_length_kpc)
    max_length_kpc = max(float(max_length_kpc), float(min_length_kpc))
    rng = np.random.default_rng(seed)
    sample_size = min(len(gas), max(nrays * 40, 512))
    centers = gas_kpc[rng.choice(len(gas), sample_size, replace=False)]

    base_directions = [
        ([1.0, 0.0, 0.0], "x"),
        ([0.0, 1.0, 0.0], "y"),
        ([0.0, 0.0, 1.0], "z"),
        ([1.0, 1.0, 1.0], "diagonal"),
        ([1.0, 1.0, 0.0], "xy_diagonal"),
        ([1.0, 0.0, 1.0], "xz_diagonal"),
        ([0.0, 1.0, 1.0], "yz_diagonal"),
    ]

    rays = []
    def endpoint_ok(start_kpc, end_kpc):
        if spherical:
            start_r = np.linalg.norm(start_kpc - gas_center)
            end_r = np.linalg.norm(end_kpc - gas_center)
            return start_r <= radius_limit and end_r <= radius_limit
        return bool(np.all(start_kpc >= core_left) and np.all(start_kpc <= core_right) and
                    np.all(end_kpc >= core_left) and np.all(end_kpc <= core_right))

    for ray_id in range(nrays):
        if ray_id < len(base_directions):
            direction, kind = base_directions[ray_id]
        else:
            direction = rng.normal(size=3)
            kind = "random"
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        length_kpc = float(rng.uniform(min_length_kpc, max_length_kpc))
        if ray_id < len(base_directions):
            length_kpc = max_length_kpc
        span = 0.5 * length_kpc * direction
        accepted = False
        start_kpc = end_kpc = center_kpc = None
        for attempt in range(1000):
            center_kpc = centers[(ray_id + attempt) % len(centers)]
            if spherical:
                if np.linalg.norm(center_kpc - gas_center) > max(0.0, radius_limit - 0.5 * length_kpc):
                    continue
            else:
                lo = np.minimum(core_left + np.abs(span), core_right - np.abs(span))
                hi = np.maximum(core_left + np.abs(span), core_right - np.abs(span))
                center_kpc = np.clip(center_kpc, lo, hi)
            start_kpc = center_kpc - span
            end_kpc = center_kpc + span
            if endpoint_ok(start_kpc, end_kpc):
                accepted = True
                break
        if not accepted:
            length_kpc = min_length_kpc
            span = 0.5 * length_kpc * direction
            center_kpc = gas_center
            start_kpc = center_kpc - span
            end_kpc = center_kpc + span
        start = np.clip(ds.arr(start_kpc, "kpc").to("code_length").d, left, right)
        end = np.clip(ds.arr(end_kpc, "kpc").to("code_length").d, left, right)
        rays.append({
            "ray_id": ray_id,
            "ray_kind": kind,
            "start": start.tolist(),
            "end": end.tolist(),
            "length_code": float(np.linalg.norm(end - start)),
            "length_kpc": float(np.linalg.norm(end_kpc - start_kpc)),
            "gas_center_x_kpc": float(gas_center[0]),
            "gas_center_y_kpc": float(gas_center[1]),
            "gas_center_z_kpc": float(gas_center[2]),
            "gas_radius_limit_kpc": radius_limit,
        })
    return rays


def requested_ions(lines):
    ions = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and not parts[-1].replace(".", "", 1).isdigit():
            ions.append(line)
        elif len(parts) >= 2:
            ions.append(" ".join(parts[:-1]))
        else:
            ions.append(line)
    return sorted(set(ions))


def add_fields(ds, lines):
    if not lines:
        return ""
    ions = requested_ions(lines)
    try:
        trident.add_ion_fields(ds, ions=ions)
        return ""
    except Exception as exc:
        return f"trident.add_ion_fields failed for {ions}: {exc}"


def ray_stats(ray):
    ad = ray.all_data()
    dl = ad[("gas", "dl")].to("cm").d
    return {
        "n_elements": int(dl.size),
        "sum_dl_cm": float(np.sum(dl)),
        "median_dl_cm": float(np.median(dl)) if dl.size else math.nan,
    }


def run_chunk(method, dataset, smoothing_factor, lines, periodic, output_dir, chunk_id, rays):
    import yt
    import trident

    yt.set_log_level(40)
    t_setup0 = time.perf_counter()
    ds = yt.load(dataset, smoothing_factor=smoothing_factor)
    field_warning = add_fields(ds, lines)
    setup_s = time.perf_counter() - t_setup0
    method_dir = Path(output_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ray in rays:
        ray_id = int(ray["ray_id"])
        filename = method_dir / f"ray_{ray_id:04d}_{method}.h5"
        start = ds.arr(ray["start"], "code_length")
        end = ds.arr(ray["end"], "code_length")
        t0 = time.perf_counter()
        status = "ok"
        error = ""
        try:
            if method == "lightray":
                out_ray = trident.make_simple_ray(
                    ds,
                    start_position=start,
                    end_position=end,
                    lines=lines,
                    data_filename=str(filename),
                )
            elif method == "meshless":
                out_ray = trident.make_meshless_voronoi_ray(
                    ds,
                    start_position=start,
                    end_position=end,
                    lines=lines,
                    data_filename=str(filename),
                    periodic=periodic,
                )
            else:
                raise ValueError(f"unknown method: {method}")
            stats = ray_stats(out_ray)
        except Exception:
            status = "failed"
            error = traceback.format_exc()
            stats = {"n_elements": 0, "sum_dl_cm": math.nan, "median_dl_cm": math.nan}
        elapsed_s = time.perf_counter() - t0
        file_size_mb = filename.stat().st_size / 1024**2 if filename.exists() else 0.0
        records.append({
            "method": method,
            "ray_id": ray_id,
            "ray_kind": ray["ray_kind"],
            "chunk_id": chunk_id,
            "worker_setup_s": setup_s,
            "elapsed_s": elapsed_s,
            "status": status,
            "error": error or field_warning,
            "output_file": str(filename),
            "file_size_mb": file_size_mb,
            "start_x": ray["start"][0],
            "start_y": ray["start"][1],
            "start_z": ray["start"][2],
            "end_x": ray["end"][0],
            "end_y": ray["end"][1],
            "end_z": ray["end"][2],
            "length_code": ray["length_code"],
            "length_kpc": ray.get("length_kpc", math.nan),
            **stats,
        })
    return records


def chunked(values, nchunks):
    chunks = [[] for _ in range(nchunks)]
    for index, value in enumerate(values):
        chunks[index % nchunks].append(value)
    return [chunk for chunk in chunks if chunk]


def run_method(method, args, output_dir, rays):
    chunks = chunked(rays, args.workers)
    t0 = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_chunk,
                method,
                args.dataset,
                args.smoothing_factor,
                args.lines,
                args.periodic,
                str(output_dir),
                chunk_id,
                chunk,
            )
            for chunk_id, chunk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    wall_s = time.perf_counter() - t0
    for record in records:
        record["method_wall_s"] = wall_s
        record["workers"] = args.workers
        record["smoothing_factor"] = args.smoothing_factor
    return sorted(records, key=lambda row: int(row["ray_id"])), wall_s


def write_table(rows, path):
    fieldnames = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def method_summary(rows, wall_times):
    out = []
    for method in ["lightray", "meshless"]:
        subset = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        elapsed = np.array([float(row["elapsed_s"]) for row in subset])
        elements = np.array([float(row["n_elements"]) for row in subset])
        out.append({
            "method": method,
            "successful_rays": len(subset),
            "failed_rays": len([row for row in rows if row["method"] == method and row["status"] != "ok"]),
            "wall_s": wall_times[method],
            "rays_per_wall_second": len(subset) / wall_times[method] if wall_times[method] > 0 else math.nan,
            "mean_ray_s": float(np.mean(elapsed)) if elapsed.size else math.nan,
            "median_ray_s": float(np.median(elapsed)) if elapsed.size else math.nan,
            "p95_ray_s": float(np.percentile(elapsed, 95)) if elapsed.size else math.nan,
            "mean_elements": float(np.mean(elements)) if elements.size else math.nan,
            "median_elements": float(np.median(elements)) if elements.size else math.nan,
        })
    if len(out) == 2:
        light = out[0]
        mesh = out[1]
        light["meshless_wall_speedup_vs_lightray"] = (
            light["wall_s"] / mesh["wall_s"] if mesh["wall_s"] > 0 else math.nan
        )
        mesh["meshless_wall_speedup_vs_lightray"] = light["meshless_wall_speedup_vs_lightray"]
        light["meshless_median_ray_speedup_vs_lightray"] = (
            light["median_ray_s"] / mesh["median_ray_s"] if mesh["median_ray_s"] > 0 else math.nan
        )
        mesh["meshless_median_ray_speedup_vs_lightray"] = light["meshless_median_ray_speedup_vs_lightray"]
    return out


def plot_timing(rows, summary, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    methods = ["lightray", "meshless"]
    colors = {"lightray": "C0", "meshless": "C1"}
    wall = [float(next(row["wall_s"] for row in summary if row["method"] == method)) for method in methods]
    axes[0].bar(methods, wall, color=[colors[m] for m in methods])
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("parallel wall-clock time")

    data = [
        [float(row["elapsed_s"]) for row in rows if row["method"] == method and row["status"] == "ok"]
        for method in methods
    ]
    axes[1].boxplot(data, tick_labels=methods, showfliers=False)
    axes[1].set_ylabel("per-ray generation time [s]")
    axes[1].set_title("per-ray timing distribution")

    for method in methods:
        subset = sorted([row for row in rows if row["method"] == method and row["status"] == "ok"], key=lambda row: int(row["ray_id"]))
        axes[2].plot(
            [int(row["ray_id"]) for row in subset],
            [float(row["elapsed_s"]) for row in subset],
            ".-",
            ms=3,
            label=method,
            color=colors[method],
        )
    axes[2].set_xlabel("ray id")
    axes[2].set_ylabel("time [s]")
    axes[2].set_title("per-ray time")
    axes[2].legend()

    for method in methods:
        subset = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        axes[3].scatter(
            [float(row["n_elements"]) for row in subset],
            [float(row["elapsed_s"]) for row in subset],
            s=18,
            alpha=0.7,
            label=method,
            color=colors[method],
        )
    axes[3].set_xlabel("ray elements")
    axes[3].set_ylabel("time [s]")
    axes[3].set_title("elements vs generation time")
    axes[3].legend()
    for method in methods:
        subset = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        axes[4].scatter(
            [float(row["length_kpc"]) for row in subset],
            [float(row["elapsed_s"]) for row in subset],
            s=18,
            alpha=0.7,
            label=method,
            color=colors[method],
        )
    axes[4].set_xlabel("sightline length [kpc]")
    axes[4].set_ylabel("time [s]")
    axes[4].set_title("timing vs sightline length")
    axes[4].legend()

    for method in methods:
        subset = sorted([row for row in rows if row["method"] == method and row["status"] == "ok"], key=lambda row: float(row["length_kpc"]))
        if not subset:
            continue
        lengths = np.array([float(row["length_kpc"]) for row in subset])
        times = np.array([float(row["elapsed_s"]) for row in subset])
        bins = np.linspace(np.min(lengths), np.max(lengths), 8)
        centers = []
        medians = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (lengths >= lo) & (lengths <= hi)
            if np.any(mask):
                centers.append(0.5 * (lo + hi))
                medians.append(float(np.median(times[mask])))
        axes[5].plot(centers, medians, "o-", label=method, color=colors[method])
    axes[5].set_xlabel("sightline length [kpc]")
    axes[5].set_ylabel("median time [s]")
    axes[5].set_title("binned median timing vs length")
    axes[5].legend()
    fig.suptitle("Ray Generation Timing Benchmark")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ray_lengths(rays, output_path):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist([ray["length_code"] for ray in rays], bins=30)
    ax.set_xlabel("ray length [code_length]")
    ax.set_ylabel("count")
    ax.set_title("Benchmark sightline lengths")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="dev/meshless_tests/output/timing_benchmark")
    parser.add_argument("--nrays", type=int, default=100)
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--ray-length-fraction", type=float, default=0.85)
    parser.add_argument("--min-length-kpc", type=float, default=None)
    parser.add_argument("--max-length-kpc", type=float, default=None)
    parser.add_argument("--spherical", action="store_true", default=True)
    parser.add_argument("--box-rays", dest="spherical", action="store_false")
    parser.add_argument("--smoothing-factor", type=float, default=2.0)
    parser.add_argument("--lines", nargs="+", default=["H I"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yt.set_log_level(40)
    ds = yt.load(args.dataset, smoothing_factor=args.smoothing_factor)
    rays = make_long_rays(
        ds, args.nrays, args.seed, args.ray_length_fraction,
        min_length_kpc=args.min_length_kpc,
        max_length_kpc=args.max_length_kpc,
        spherical=args.spherical,
    )
    write_table(rays, output_dir / "benchmark_rays")
    plot_ray_lengths(rays, output_dir / "benchmark_ray_lengths.png")

    all_rows = []
    wall_times = {}
    for method in ["lightray", "meshless"]:
        print(f"Running {method} benchmark for {args.nrays} rays with {args.workers} workers")
        rows, wall_s = run_method(method, args, output_dir, rays)
        all_rows.extend(rows)
        wall_times[method] = wall_s
        print(f"  {method}: {wall_s:.3f} s")

    summary = method_summary(all_rows, wall_times)
    write_table(all_rows, output_dir / "ray_generation_timing")
    write_table(summary, output_dir / "ray_generation_timing_summary")
    plot_timing(all_rows, summary, output_dir / "ray_generation_timing_summary.png")

    print("Ray-generation timing benchmark complete")
    print(f"  output_dir: {output_dir}")
    for row in summary:
        print(
            f"  {row['method']}: wall={row['wall_s']:.3f}s, "
            f"median_ray={row['median_ray_s']:.3f}s, "
            f"success={row['successful_rays']}, failed={row['failed_rays']}"
        )
    if summary:
        print(f"  meshless wall speedup vs LightRay: {summary[0]['meshless_wall_speedup_vs_lightray']:.3f}x")


if __name__ == "__main__":
    main()
