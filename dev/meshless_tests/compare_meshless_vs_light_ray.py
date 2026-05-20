#!/usr/bin/env python
"""Compare yt LightRay and meshless Voronoi rays on a local TNG cutout."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(Path(__file__).resolve().parent)

import trident


DEFAULT_DATASET = (
    "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5"
)
SPECTRUM_LINES = ["H I 1216", "O VI 1032", "O VI 1038", "Mg II 2796", "Mg II 2803"]
FIELD_LINES = ["H I", "O VI", "Mg II"]


def _ray_endpoints(ds, n_random, seed):
    left = ds.domain_left_edge.to("code_length")
    right = ds.domain_right_edge.to("code_length")
    width = right - left
    pad = 0.1 * width
    lo = left + pad
    hi = right - pad
    center = 0.5 * (left + right)

    rays = [
        (
            ds.arr([lo.d[0], center.d[1], center.d[2]], "code_length"),
            ds.arr([hi.d[0], center.d[1], center.d[2]], "code_length"),
        ),
        (
            ds.arr([center.d[0], lo.d[1], center.d[2]], "code_length"),
            ds.arr([center.d[0], hi.d[1], center.d[2]], "code_length"),
        ),
        (
            ds.arr([center.d[0], center.d[1], lo.d[2]], "code_length"),
            ds.arr([center.d[0], center.d[1], hi.d[2]], "code_length"),
        ),
        (lo, hi),
    ]

    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        start = lo + rng.random(3) * (hi - lo)
        end = lo + rng.random(3) * (hi - lo)
        rays.append((start, end))
    return rays


def _field_stats(ray):
    ad = ray.all_data()
    stats = {
        "n_elements": int(len(ad[("gas", "dl")])),
        "dl_sum_cm": float(ad[("gas", "dl")].sum().to("cm").d),
        "dl_min_cm": float(ad[("gas", "dl")].min().to("cm").d),
        "dl_max_cm": float(ad[("gas", "dl")].max().to("cm").d),
        "dl_median_cm": float(np.median(ad[("gas", "dl")].to("cm").d)),
    }
    for field in [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity"),
        ("gas", "velocity_los"),
        ("gas", "redshift_dopp"),
        ("gas", "redshift_eff"),
    ]:
        try:
            values = ad[field]
        except Exception:
            continue
        stats[str(field) + "_median"] = float(np.median(values.d))
    for field in [
        ("gas", "H_p0_number_density"),
        ("gas", "O_p5_number_density"),
        ("gas", "Mg_p1_number_density"),
    ]:
        try:
            stats[str(field) + "_column"] = float((ad[field] * ad[("gas", "dl")]).sum().d)
        except Exception:
            continue
    return stats


def _spectrum_stats(ray, prefix, output_dir):
    sg = trident.SpectrumGenerator("COS-G130M")
    sg.make_spectrum(ray, lines=SPECTRUM_LINES)
    spectrum_path = output_dir / f"{prefix}_spectrum.h5"
    sg.save_spectrum(str(spectrum_path))
    return {
        "spectrum_file": str(spectrum_path),
        "flux_min": float(np.min(sg.flux_field)),
        "flux_mean": float(np.mean(sg.flux_field)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="comparison_outputs")
    parser.add_argument("--n-random", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--periodic", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = yt.load(args.dataset)
    rows = []
    for i, (start, end) in enumerate(_ray_endpoints(ds, args.n_random, args.seed)):
        light_file = output_dir / f"ray_{i:03d}_light.h5"
        meshless_file = output_dir / f"ray_{i:03d}_meshless.h5"

        light_ray = trident.make_simple_ray(
            ds,
            start_position=start,
            end_position=end,
            lines=FIELD_LINES,
            data_filename=str(light_file),
        )
        meshless_ray = trident.make_meshless_voronoi_ray(
            ds,
            start_position=start,
            end_position=end,
            lines=FIELD_LINES,
            data_filename=str(meshless_file),
            periodic=args.periodic,
        )

        row = {"ray": i}
        for label, ray in [("light", light_ray), ("meshless", meshless_ray)]:
            for key, value in _field_stats(ray).items():
                row[f"{label}_{key}"] = value
            for key, value in _spectrum_stats(ray, f"ray_{i:03d}_{label}", output_dir).items():
                row[f"{label}_{key}"] = value
        row["dl_sum_frac_diff"] = (
            row["meshless_dl_sum_cm"] - row["light_dl_sum_cm"]
        ) / row["light_dl_sum_cm"]
        rows.append(row)

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
