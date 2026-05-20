"""
I/O helpers for meshless Voronoi rays.

These helpers keep the meshless ray writer aligned with Trident's existing
LightRay convention: write a yt data dataset tagged as ``yt_light_ray`` and
load it back through yt before passing it to SpectrumGenerator.
"""

from __future__ import annotations

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
