#!/usr/bin/env python
"""Profile meshless Voronoi ray tracing costs."""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_CONTEXT = Path("/private/tmp/trident_meshless_profile_context")
_RUN_CONTEXT.mkdir(parents=True, exist_ok=True)
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(_RUN_CONTEXT)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yt

from trident.meshless_voronoi_ray import MeshlessVoronoiRayTracer


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)


def parse_bool(value):
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def generate_points(npoints, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((npoints, 3)))


def generate_rays(nrays, seed, left=None, right=None):
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
            end = lo + rng.random(3) * (hi - lo)
        starts.append(start)
        ends.append(end)
    return np.ascontiguousarray(starts), np.ascontiguousarray(ends)


def load_tng(dataset, periodic):
    t0 = time.perf_counter()
    ds = yt.load(dataset)
    yt_load_s = time.perf_counter() - t0
    ad = ds.all_data()
    t0 = time.perf_counter()
    try:
        positions = ad[("gas", "coordinates")].to("code_length")
    except Exception:
        positions = ad[("PartType0", "Coordinates")].to("code_length")
    position_extract_s = time.perf_counter() - t0
    box_size = ds.domain_width.to("code_length").d if periodic else None
    return (
        ds,
        np.ascontiguousarray(positions.d),
        ds.domain_left_edge.to("code_length").d,
        ds.domain_right_edge.to("code_length").d,
        box_size,
        yt_load_s,
        position_extract_s,
    )


def trace_loop(tracer, starts, ends):
    return [
        tracer.trace_ray(start_position=start, end_position=end)
        for start, end in zip(starts, ends)
    ]


def summarize(rays, elapsed_s):
    nseg = np.array([len(ray.indices) for ray in rays], dtype=float)
    fallbacks = np.array([ray.fallback_count for ray in rays], dtype=float)
    nudges = np.array([ray.nudge_count for ray in rays], dtype=float)
    failed = np.array([ray.failed_stack_recoveries for ray in rays], dtype=float)
    lengths = np.array([ray.length for ray in rays], dtype=float)
    sums = np.array([ray.dl.sum() for ray in rays], dtype=float)
    return {
        "trace_s": elapsed_s,
        "trace_per_ray_s": elapsed_s / len(rays),
        "n_segments_total": int(np.sum(nseg)),
        "n_segments_mean": float(np.mean(nseg)),
        "n_segments_median": float(np.median(nseg)),
        "fallback_count_total": int(np.sum(fallbacks)),
        "nudge_count_total": int(np.sum(nudges)),
        "failed_stack_recoveries_total": int(np.sum(failed)),
        "max_sum_dl_abs_error": float(np.max(np.abs(sums - lengths))),
    }


def profile_call(profile_mode, output_path, func, *args):
    if profile_mode == "none":
        return func(*args), ""
    profiler = cProfile.Profile()
    result = profiler.runcall(func, *args)
    with output_path.open("w") as handle:
        pstats.Stats(profiler, stream=handle).sort_stats("cumtime").print_stats(80)
    return result, str(output_path)


def run_case(args, npoints, nrays):
    row = {
        "mode": args.mode,
        "npoints": int(npoints),
        "nrays": int(nrays),
        "periodic": bool(args.periodic),
    }
    if args.mode == "synthetic":
        points = generate_points(npoints, args.seed)
        starts, ends = generate_rays(nrays, args.seed + nrays)
        box_size = 1.0 if args.periodic else None
        row["yt_load_s"] = 0.0
        row["position_extract_s"] = 0.0
    else:
        _, points, left, right, box_size, yt_load_s, position_extract_s = load_tng(
            args.dataset, args.periodic
        )
        starts, ends = generate_rays(nrays, args.seed + nrays, left=left, right=right)
        row["npoints"] = int(len(points))
        row["yt_load_s"] = yt_load_s
        row["position_extract_s"] = position_extract_s

    t0 = time.perf_counter()
    tracer = MeshlessVoronoiRayTracer(points, box_size=box_size)
    row["tree_build_s"] = time.perf_counter() - t0
    profile_path = Path(args.output_dir) / f"cprofile_{args.mode}_n{row['npoints']}_r{nrays}.txt"
    t0 = time.perf_counter()
    rays, cprofile_file = profile_call(args.profile, profile_path, trace_loop, tracer, starts, ends)
    row.update(summarize(rays, time.perf_counter() - t0))
    row["cprofile_file"] = cprofile_file
    return row


def write_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "profile_summary.csv"
    json_path = output_dir / "profile_summary.json"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "tng"], default="synthetic")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--nrays", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--npoints", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--output-dir", default="/private/tmp/trident_meshless_profile")
    parser.add_argument("--seed", type=int, default=398784)
    parser.add_argument("--periodic", type=parse_bool, default=False)
    parser.add_argument("--profile", choices=["cProfile", "none"], default="none")
    parser.add_argument("--line-profiler", type=parse_bool, default=False)
    parser.add_argument("--benchmark-lightray", type=parse_bool, default=False)
    parser.add_argument("--write-rays", type=parse_bool, default=False)
    args = parser.parse_args()

    if args.line_profiler:
        print("line-profiler is not wired into this script; use cProfile for now.")
    if args.benchmark_lightray:
        print("LightRay benchmarking is handled by benchmark_many_sightlines.py.")
    if args.write_rays:
        print("This profiler measures geometry only; no ray files will be written.")

    rows = []
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    npoint_values = args.npoints if args.mode == "synthetic" else [0]
    for npoints in npoint_values:
        for nrays in args.nrays:
            row = run_case(args, npoints, nrays)
            rows.append(row)
            print(
                "mode={mode} npoints={npoints} nrays={nrays} "
                "tree={tree_build_s:.4f}s trace={trace_s:.4f}s "
                "per_ray={trace_per_ray_s:.6f}s segments_mean={n_segments_mean:.2f}".format(
                    **row
                )
            )
    csv_path, json_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
