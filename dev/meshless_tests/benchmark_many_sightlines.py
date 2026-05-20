#!/usr/bin/env python
"""Benchmark many meshless sightlines and optional yt/Trident LightRay runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CONTEXT = Path("/private/tmp/trident_meshless_benchmark_context")
_RUN_CONTEXT.mkdir(parents=True, exist_ok=True)
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(_RUN_CONTEXT)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
import yt

import trident
from trident.meshless_ray_io import write_meshless_ray_catalog_hdf5
from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def synthetic_inputs(npoints, nrays, seed):
    rng = np.random.default_rng(seed)
    points = np.ascontiguousarray(rng.random((npoints, 3)))
    starts, ends = box_rays(nrays, seed + nrays)
    return points, starts, ends, None


def tng_inputs(dataset, nrays, seed, periodic):
    t0 = time.perf_counter()
    ds = yt.load(dataset)
    yt_load_s = time.perf_counter() - t0
    ad = ds.all_data()
    t0 = time.perf_counter()
    try:
        positions = ad[("gas", "coordinates")].to("code_length")
    except Exception:
        positions = ad[("PartType0", "Coordinates")].to("code_length")
    positions_np = np.ascontiguousarray(positions.d)
    position_extract_s = time.perf_counter() - t0
    gas_left = np.percentile(positions_np, 5, axis=0)
    gas_right = np.percentile(positions_np, 95, axis=0)
    starts, ends = box_rays(
        nrays,
        seed + nrays,
        gas_left,
        gas_right,
    )
    return (
        positions_np,
        starts,
        ends,
        {
            "ds": ds,
            "yt_load_s": yt_load_s,
            "position_extract_s": position_extract_s,
            "box_size": ds.domain_width.to("code_length").d if periodic else None,
        },
    )


def box_rays(nrays, seed, left=None, right=None):
    rng = np.random.default_rng(seed)
    left = np.zeros(3) if left is None else np.asarray(left, dtype=float)
    right = np.ones(3) if right is None else np.asarray(right, dtype=float)
    center = 0.5 * (left + right)
    width = right - left
    lo = left + 0.05 * width
    hi = right - 0.05 * width
    base = [
        ([lo[0], center[1], center[2]], [hi[0], center[1], center[2]]),
        ([center[0], lo[1], center[2]], [center[0], hi[1], center[2]]),
        ([center[0], center[1], lo[2]], [center[0], center[1], hi[2]]),
        (lo, hi),
    ]
    starts = []
    ends = []
    for i in range(nrays):
        if i < len(base):
            start, end = base[i]
        else:
            start = lo + rng.random(3) * (hi - lo)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            length = rng.uniform(0.35, 0.9) * np.min(width)
            end = np.clip(start + length * direction, lo, hi)
            if np.linalg.norm(end - start) < 0.25 * np.min(width):
                end = lo + rng.random(3) * (hi - lo)
        starts.append(start)
        ends.append(end)
    return np.ascontiguousarray(starts), np.ascontiguousarray(ends)


def ray_summary(method, nrays, elapsed_s, rays=None, batch=None, extra=None):
    if batch is not None:
        nseg = batch.n_segments
        fallbacks = batch.fallback_counts
        nudges = batch.nudge_counts
        failed = batch.failed_stack_recoveries
        lengths = batch.lengths
        sum_dl = np.array([batch.ray(i).dl.sum() for i in range(batch.n_rays)])
    else:
        nseg = np.array([len(ray.indices) for ray in rays], dtype=float)
        fallbacks = np.array([ray.fallback_count for ray in rays], dtype=float)
        nudges = np.array([ray.nudge_count for ray in rays], dtype=float)
        failed = np.array([ray.failed_stack_recoveries for ray in rays], dtype=float)
        lengths = np.array([ray.length for ray in rays], dtype=float)
        sum_dl = np.array([ray.dl.sum() for ray in rays], dtype=float)
    row = {
        "method": method,
        "nrays": int(nrays),
        "total_runtime_s": float(elapsed_s),
        "runtime_per_ray_s": float(elapsed_s / nrays),
        "n_segments_total": int(np.sum(nseg)),
        "n_segments_mean": float(np.mean(nseg)),
        "fallback_count_total": int(np.sum(fallbacks)),
        "nudge_count_total": int(np.sum(nudges)),
        "failed_stack_recoveries_total": int(np.sum(failed)),
        "max_sum_dl_abs_error": float(np.max(np.abs(sum_dl - lengths))),
        "status": "ok",
        "error": "",
    }
    if extra:
        row.update(extra)
    return row


def benchmark_meshless_rebuild(points, starts, ends, box_size):
    rays = []
    t0 = time.perf_counter()
    for start, end in zip(starts, ends):
        tracer = MeshlessVoronoiRayTracer(points, box_size=box_size)
        rays.append(tracer.trace_ray(start, end_position=end))
    return rays, time.perf_counter() - t0


def benchmark_meshless_batch(points, starts, ends, box_size, parallel, n_jobs, chunksize):
    t0 = time.perf_counter()
    tracer = MeshlessVoronoiRayTracer(points, box_size=box_size)
    tree_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch = tracer.trace_rays(
        starts,
        end_positions=ends,
        parallel=parallel,
        n_jobs=n_jobs,
        chunksize=chunksize,
        return_format="ragged",
    )
    trace_s = time.perf_counter() - t0
    return batch, tree_s, trace_s


def benchmark_lightray(ds, starts, ends, output_dir, lines):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    t0_all = time.perf_counter()
    for i, (start, end) in enumerate(zip(starts, ends)):
        filename = output_dir / f"lightray_{i:05d}.h5"
        t0 = time.perf_counter()
        try:
            ray = trident.make_simple_ray(
                ds,
                start_position=ds.arr(start, "code_length"),
                end_position=ds.arr(end, "code_length"),
                lines=lines,
                data_filename=str(filename),
            )
            nseg = len(ray.all_data()[("gas", "dl")])
            status = "ok"
            error = ""
        except Exception:
            nseg = 0
            status = "failed"
            error = traceback.format_exc()
            errors.append(error)
        rows.append({"ray_id": i, "elapsed_s": time.perf_counter() - t0, "n_segments": nseg, "status": status, "error": error})
    elapsed = time.perf_counter() - t0_all
    return {
        "method": "lightray",
        "nrays": len(starts),
        "total_runtime_s": elapsed,
        "runtime_per_ray_s": elapsed / len(starts),
        "n_segments_total": int(sum(row["n_segments"] for row in rows)),
        "n_segments_mean": float(np.mean([row["n_segments"] for row in rows])),
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "partial",
        "error": errors[0] if errors else "",
    }


def write_tables(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"
    json_path = output_dir / "benchmark_summary.json"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def write_plots(rows, output_dir):
    output_dir = Path(output_dir)
    ok = [row for row in rows if row.get("status") == "ok" and "runtime_per_ray_s" in row]
    if not ok:
        return []
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted({row["method"] for row in ok}):
        subset = sorted([row for row in ok if row["method"] == method], key=lambda row: row["nrays"])
        ax.plot([row["nrays"] for row in subset], [row["total_runtime_s"] for row in subset], marker="o", label=method)
    ax.set_xlabel("number of rays")
    ax.set_ylabel("total runtime [s]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "runtime_vs_nrays.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted({row["method"] for row in ok}):
        subset = sorted([row for row in ok if row["method"] == method], key=lambda row: row["nrays"])
        ax.plot([row["nrays"] for row in subset], [row["runtime_per_ray_s"] for row in subset], marker="o", label=method)
    ax.set_xlabel("number of rays")
    ax.set_ylabel("runtime per ray [s]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "runtime_per_ray_vs_nrays.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "tng"], default="synthetic")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument("--nrays", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--output-dir", default="/private/tmp/trident_meshless_benchmarks")
    parser.add_argument("--periodic", type=parse_bool, default=False)
    parser.add_argument("--benchmark-lightray", type=parse_bool, default=False)
    parser.add_argument("--lines", nargs="*", default=["H I"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--write-catalog", type=parse_bool, default=True)
    args = parser.parse_args()

    rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for nrays in args.nrays:
        if args.mode == "synthetic":
            points, starts, ends, context = synthetic_inputs(args.npoints, nrays, args.seed)
            box_size = 1.0 if args.periodic else None
            common = {"mode": "synthetic", "npoints": len(points)}
        else:
            points, starts, ends, context = tng_inputs(args.dataset, nrays, args.seed, args.periodic)
            box_size = context["box_size"]
            common = {
                "mode": "tng",
                "npoints": len(points),
                "yt_load_s": context["yt_load_s"],
                "position_extract_s": context["position_extract_s"],
            }

        rays, elapsed = benchmark_meshless_rebuild(points, starts, ends, box_size)
        rows.append(ray_summary("meshless_rebuild_per_ray", nrays, elapsed, rays=rays, extra=common))

        batch, tree_s, trace_s = benchmark_meshless_batch(
            points, starts, ends, box_size, "none", 1, args.chunksize
        )
        rows.append(ray_summary(
            "meshless_batch_serial",
            nrays,
            tree_s + trace_s,
            batch=batch,
            extra={**common, "tree_build_s": tree_s, "trace_s": trace_s},
        ))

        batch_thread, tree_thread_s, trace_thread_s = benchmark_meshless_batch(
            points, starts, ends, box_size, "threads", args.threads, args.chunksize
        )
        rows.append(ray_summary(
            "meshless_batch_threads",
            nrays,
            tree_thread_s + trace_thread_s,
            batch=batch_thread,
            extra={**common, "tree_build_s": tree_thread_s, "trace_s": trace_thread_s, "threads": args.threads},
        ))

        if args.write_catalog:
            catalog_path = output_dir / f"catalog_{args.mode}_{nrays:05d}.h5"
            t0 = time.perf_counter()
            write_meshless_ray_catalog_hdf5(catalog_path, batch, overwrite=True)
            elapsed = time.perf_counter() - t0
            rows.append({
                **common,
                "method": "meshless_catalog_write",
                "nrays": nrays,
                "total_runtime_s": elapsed,
                "runtime_per_ray_s": elapsed / nrays,
                "catalog_file": str(catalog_path),
                "status": "ok",
                "error": "",
            })

        if args.benchmark_lightray and args.mode == "tng":
            rows.append({
                **common,
                **benchmark_lightray(
                    context["ds"], starts, ends, output_dir / f"lightray_{nrays:05d}", args.lines
                ),
            })

    csv_path, json_path = write_tables(rows, output_dir)
    plot_paths = write_plots(rows, output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    for row in rows:
        if row.get("status") == "ok":
            print(
                f"{row['method']} nrays={row['nrays']} total={row['total_runtime_s']:.4f}s "
                f"per_ray={row['runtime_per_ray_s']:.6f}s"
            )


if __name__ == "__main__":
    main()
