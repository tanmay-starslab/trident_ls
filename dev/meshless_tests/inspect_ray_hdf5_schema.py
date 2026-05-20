#!/usr/bin/env python
"""Inspect a Trident ray HDF5 schema.

If no input ray is supplied, this script creates a tiny standard
``make_simple_ray`` first and then prints the HDF5 layout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if Path.cwd().resolve() == _REPO_ROOT:
    os.chdir(Path(__file__).resolve().parent)

import trident
from trident.meshless_ray_io import inspect_existing_trident_ray_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ray", nargs="?", help="Existing Trident ray HDF5 file")
    parser.add_argument("--output-dir", default="internal_meshless_outputs")
    args = parser.parse_args()

    if args.ray:
        ray_path = Path(args.ray)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ray_path = output_dir / "standard_simple_ray.h5"
        ds = trident.make_onezone_dataset()
        trident.make_simple_ray(
            ds,
            start_position=ds.arr([0.0, 0.0, 0.0], "unitary"),
            end_position=ds.arr([1.0, 1.0, 1.0], "unitary"),
            lines=["H I"],
            data_filename=str(ray_path),
        )

    inspect_existing_trident_ray_schema(str(ray_path), print_schema=True)


if __name__ == "__main__":
    main()
