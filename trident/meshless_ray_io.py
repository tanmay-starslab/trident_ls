"""
I/O helpers for meshless Voronoi rays.

These helpers keep the meshless ray writer aligned with Trident's existing
LightRay convention: write a yt data dataset tagged as ``yt_light_ray`` and
load it back through yt before passing it to SpectrumGenerator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from yt.frontends.ytdata.utilities import save_as_dataset
from yt.loaders import load
from yt.utilities.logger import ytLogger as mylog


def _field_in_data(data, name):
    if name in data:
        return name
    gas_name = ("gas", name)
    if gas_name in data:
        return gas_name
    return None


def write_meshless_ray_hdf5(
    source_ds,
    filename,
    data,
    extra_attrs=None,
    fail_empty=True,
):
    """Write a meshless ray as a Trident/yt LightRay-compatible dataset.

    Parameters
    ----------
    source_ds : yt Dataset or dict
        Source dataset metadata passed through to yt's ``save_as_dataset``.
    filename : str
        Destination HDF5 filename.
    data : dict
        Ray field dictionary.  Keys may be strings or field tuples, matching
        the conventions used by Trident's LightRay writer.
    extra_attrs : dict, optional
        Additional metadata stored on the yt data dataset.
    fail_empty : bool, optional
        Raise if no positive-length ray elements remain.
    """
    out_data = dict(data)
    attrs = {"data_type": "yt_light_ray", "dimensionality": 3}
    if extra_attrs:
        attrs.update(extra_attrs)

    dl_key = _field_in_data(out_data, "dl")
    if dl_key is None:
        raise RuntimeError("Meshless ray data must include a dl field.")

    mask = out_data[dl_key] > 0
    temp_key = _field_in_data(out_data, "temperature")
    if temp_key is not None:
        mask = mask & (out_data[temp_key] > 0)

    if not mask.any():
        err = "No zones along meshless ray with positive dl"
        if temp_key is not None:
            err += " and nonzero temperature"
        err += ". Modify your ray trajectory."
        if fail_empty:
            raise RuntimeError(err)
        mylog.warning(err)
        attrs["empty"] = True

    for key in list(out_data.keys()):
        if key == ("gas", "extra_data"):
            continue
        out_data[key] = out_data[key][mask]

    field_types = dict((field, "grid") for field in out_data.keys())
    save_as_dataset(
        source_ds,
        filename,
        out_data,
        field_types=field_types,
        extra_attrs=attrs,
    )
    return filename


def load_meshless_ray(filename):
    """Load a meshless ray dataset written by ``write_meshless_ray_hdf5``."""
    ray = load(filename)
    ray.domain_left_edge = ray.domain_left_edge.to("code_length")
    ray.domain_right_edge = ray.domain_right_edge.to("code_length")
    return ray


def _catalog_field_name(field):
    if isinstance(field, tuple):
        return "__".join(str(part) for part in field)
    return str(field).replace("/", "_")


def _as_plain_array(values):
    unit = ""
    if hasattr(values, "units"):
        unit = str(values.units)
    if hasattr(values, "d"):
        values = values.d
    return np.asarray(values), unit


def write_meshless_ray_catalog_hdf5(
    filename,
    ray_batch,
    fields=None,
    metadata=None,
    overwrite=False,
    compression=None,
):
    """Write many meshless rays in a compact ragged HDF5 catalog.

    Parameters
    ----------
    filename : str
        Output HDF5 path.
    ray_batch : MeshlessVoronoiRayBatch
        Ragged batch returned by ``MeshlessVoronoiRayTracer.trace_rays`` with
        ``return_format="ragged"``.
    fields : dict, optional
        Optional flat per-segment field arrays.  Each array must have first
        dimension equal to the total number of segments.
    metadata : dict, optional
        Extra scalar/string metadata stored as file attributes.
    overwrite : bool, optional
        Overwrite an existing catalog.
    compression : str, optional
        HDF5 compression filter, for example ``"gzip"`` or ``"lzf"``.
    """
    import h5py

    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise FileExistsError(f"{filename} already exists. Pass overwrite=True.")
    filename.parent.mkdir(parents=True, exist_ok=True)

    fields = {} if fields is None else dict(fields)
    metadata = {} if metadata is None else dict(metadata)
    nseg = int(len(ray_batch.indices))
    for key, values in fields.items():
        arr, _ = _as_plain_array(values)
        if len(arr) != nseg:
            raise ValueError(
                f"Field {key!r} has length {len(arr)}, expected {nseg}."
            )

    with h5py.File(filename, "w") as handle:
        handle.attrs["data_type"] = "meshless_voronoi_ray_catalog"
        handle.attrs["algorithm"] = "meshless_voronoi_salsa"
        handle.attrs["algorithm_version"] = ray_batch.algorithm_version
        handle.attrs["periodic"] = bool(ray_batch.periodic)
        handle.attrs["epsilon"] = float(ray_batch.eps)
        for key, value in metadata.items():
            try:
                handle.attrs[key] = value
            except TypeError:
                handle.attrs[key] = str(value)

        rays = handle.create_group("rays")
        rays.create_dataset("start_positions", data=ray_batch.start_positions, compression=compression)
        rays.create_dataset("end_positions", data=ray_batch.end_positions, compression=compression)
        rays.create_dataset("directions", data=ray_batch.directions, compression=compression)
        rays.create_dataset("lengths", data=ray_batch.lengths, compression=compression)
        rays.create_dataset("offsets", data=ray_batch.ray_offsets, compression=compression)
        rays.create_dataset("n_segments", data=ray_batch.n_segments, compression=compression)
        rays.create_dataset("start_indices", data=ray_batch.start_indices, compression=compression)
        rays.create_dataset("end_indices", data=ray_batch.end_indices, compression=compression)
        rays.create_dataset(
            "failed_stack_recoveries",
            data=ray_batch.failed_stack_recoveries,
            compression=compression,
        )
        rays.create_dataset("fallback_counts", data=ray_batch.fallback_counts, compression=compression)
        rays.create_dataset("nudge_counts", data=ray_batch.nudge_counts, compression=compression)
        if ray_batch.box_size is not None:
            rays.create_dataset("box_size", data=ray_batch.box_size)

        segments = handle.create_group("segments")
        segments.create_dataset("cell_indices", data=ray_batch.indices, compression=compression)
        segments.create_dataset("dl", data=ray_batch.dl, compression=compression)
        segments.create_dataset("cumulative_dl", data=ray_batch.cumulative_dl, compression=compression)

        field_group = handle.create_group("fields")
        for key, values in fields.items():
            arr, unit = _as_plain_array(values)
            dset = field_group.create_dataset(
                _catalog_field_name(key),
                data=arr,
                compression=compression,
            )
            dset.attrs["field_name"] = str(key)
            dset.attrs["unit"] = unit

    return str(filename)


def load_meshless_ray_catalog_hdf5(filename):
    """Load a compact meshless ray catalog into plain numpy arrays."""
    import h5py

    out = {"attrs": {}, "rays": {}, "segments": {}, "fields": {}}
    with h5py.File(filename, "r") as handle:
        out["attrs"] = dict(handle.attrs)
        for name, dset in handle["rays"].items():
            out["rays"][name] = dset[()]
        for name, dset in handle["segments"].items():
            out["segments"][name] = dset[()]
        if "fields" in handle:
            for name, dset in handle["fields"].items():
                out["fields"][name] = {
                    "data": dset[()],
                    "attrs": dict(dset.attrs),
                }
    return out


def read_meshless_catalog_ray(filename, ray_id):
    """Read one ray's geometry and per-segment fields from a catalog."""
    catalog = load_meshless_ray_catalog_hdf5(filename)
    offsets = catalog["rays"]["offsets"]
    if ray_id < 0 or ray_id >= len(offsets) - 1:
        raise IndexError("ray_id out of range")
    first = int(offsets[ray_id])
    last = int(offsets[ray_id + 1])
    ray = {
        "ray_id": int(ray_id),
        "start_position": catalog["rays"]["start_positions"][ray_id],
        "end_position": catalog["rays"]["end_positions"][ray_id],
        "direction": catalog["rays"]["directions"][ray_id],
        "length": catalog["rays"]["lengths"][ray_id],
        "indices": catalog["segments"]["cell_indices"][first:last],
        "dl": catalog["segments"]["dl"][first:last],
        "cumulative_dl": catalog["segments"]["cumulative_dl"][first:last],
        "fields": {},
    }
    for name, payload in catalog["fields"].items():
        ray["fields"][name] = {
            "data": payload["data"][first:last],
            "attrs": payload["attrs"],
        }
    return ray


def inspect_existing_trident_ray_schema(filename, print_schema=True):
    """Inspect an existing Trident/yt ray HDF5 file.

    Returns a dictionary containing file attributes and dataset/group entries.
    This helper intentionally depends on h5py only at call time so importing
    Trident does not require schema-inspection dependencies beyond yt itself.
    """
    import h5py

    schema = {"attrs": {}, "items": []}
    with h5py.File(filename, "r") as handle:
        schema["attrs"] = dict(handle.attrs)

        def visitor(name, obj):
            item = {"name": name, "attrs": dict(obj.attrs)}
            if hasattr(obj, "shape"):
                item["kind"] = "dataset"
                item["shape"] = obj.shape
                item["dtype"] = str(obj.dtype)
            else:
                item["kind"] = "group"
            schema["items"].append(item)

        handle.visititems(visitor)

    if print_schema:
        print("attrs:")
        for key in sorted(schema["attrs"]):
            print(f"  {key}: {schema['attrs'][key]}")
        for item in schema["items"]:
            if item["kind"] == "dataset":
                print(
                    f"dataset {item['name']} shape={item['shape']} "
                    f"dtype={item['dtype']} attrs={item['attrs']}"
                )
            else:
                print(f"group {item['name']} attrs={item['attrs']}")

    return schema
