"""Sightline generation and batch tracing helpers for meshless Voronoi rays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MeshlessSightlineSet:
    """Container for generated sightline start/end coordinates and metadata."""

    starts: np.ndarray
    ends: np.ndarray
    metadata: dict

    @property
    def directions(self) -> np.ndarray:
        delta = self.ends - self.starts
        lengths = np.linalg.norm(delta, axis=1)
        return delta / lengths[:, None]

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.ends - self.starts, axis=1)


def _as_vector(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector.")
    return arr


def _unit_vector(value, name):
    arr = _as_vector(value, name)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError(f"{name} must be non-zero.")
    return arr / norm


def _basis_from_plane_or_normal(
    plane="xy",
    normal_axis: Optional[str] = None,
    normal_vector=None,
    up_vector=None,
):
    if normal_vector is None:
        if normal_axis is not None:
            axis = normal_axis.lower()
            mapping = {
                "x": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
                "y": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
                "z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            }
            if axis not in mapping:
                raise ValueError("normal_axis must be 'x', 'y', or 'z'.")
            return mapping[axis]

        plane = plane.lower()
        mapping = {
            "xy": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "xz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            "yz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        }
        if plane not in mapping:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'.")
        return mapping[plane]

    normal = _unit_vector(normal_vector, "normal_vector")
    if up_vector is None:
        trial = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(trial, normal))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
    else:
        trial = _unit_vector(up_vector, "up_vector")
    v = trial - np.dot(trial, normal) * normal
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 0:
        raise ValueError("up_vector cannot be parallel to normal_vector.")
    v = v / v_norm
    u = np.cross(v, normal)
    u = u / np.linalg.norm(u)
    return u, v, normal


def _line_offsets(length, start_offset, end_offset):
    if start_offset is None and end_offset is None:
        if length is None:
            raise ValueError("Provide length or start_offset/end_offset.")
        half = 0.5 * float(length)
        return -half, half
    if start_offset is None or end_offset is None:
        raise ValueError("start_offset and end_offset must be provided together.")
    return float(start_offset), float(end_offset)


def generate_uniform_grid_sightlines(
    center,
    width,
    height,
    nx,
    ny,
    normal_axis=None,
    normal_vector=None,
    up_vector=None,
    length=None,
    start_offset=None,
    end_offset=None,
    plane="xy",
    ordering="row-major",
):
    """Generate parallel sightlines through a rectangular uniform grid."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    center = _as_vector(center, "center")
    u, v, normal = _basis_from_plane_or_normal(
        plane=plane,
        normal_axis=normal_axis,
        normal_vector=normal_vector,
        up_vector=up_vector,
    )
    start_offset, end_offset = _line_offsets(length, start_offset, end_offset)
    xs = np.linspace(-0.5 * float(width), 0.5 * float(width), int(nx))
    ys = np.linspace(-0.5 * float(height), 0.5 * float(height), int(ny))

    starts = []
    ends = []
    grid_i = []
    grid_j = []
    projected_x = []
    projected_y = []
    if ordering not in ("row-major", "column-major"):
        raise ValueError("ordering must be 'row-major' or 'column-major'.")
    iterator = (
        ((i, j) for j in range(int(ny)) for i in range(int(nx)))
        if ordering == "row-major"
        else ((i, j) for i in range(int(nx)) for j in range(int(ny)))
    )
    for i, j in iterator:
        point = center + xs[i] * u + ys[j] * v
        starts.append(point + start_offset * normal)
        ends.append(point + end_offset * normal)
        grid_i.append(i)
        grid_j.append(j)
        projected_x.append(xs[i])
        projected_y.append(ys[j])

    metadata = {
        "type": "uniform_grid",
        "grid_i": np.asarray(grid_i, dtype=np.int64),
        "grid_j": np.asarray(grid_j, dtype=np.int64),
        "projected_x": np.asarray(projected_x, dtype=np.float64),
        "projected_y": np.asarray(projected_y, dtype=np.float64),
        "center": center,
        "basis_u": u,
        "basis_v": v,
        "normal": normal,
        "ordering": ordering,
    }
    return MeshlessSightlineSet(np.asarray(starts), np.asarray(ends), metadata)


def generate_radial_sightlines(
    center,
    radius,
    nrays,
    normal_axis=None,
    normal_vector=None,
    length=None,
    radial_distribution="area-uniform",
    phi_distribution="uniform",
    r_min=0.0,
    seed=None,
    plane="xy",
    up_vector=None,
):
    """Generate parallel sightlines with radial impact parameters."""
    if nrays <= 0:
        raise ValueError("nrays must be positive.")
    if phi_distribution != "uniform":
        raise ValueError("Only phi_distribution='uniform' is supported.")
    center = _as_vector(center, "center")
    radius = float(radius)
    r_min = float(r_min)
    if radius <= 0 or r_min < 0 or r_min > radius:
        raise ValueError("Require 0 <= r_min <= radius and radius > 0.")
    u_basis, v_basis, normal = _basis_from_plane_or_normal(
        plane=plane,
        normal_axis=normal_axis,
        normal_vector=normal_vector,
        up_vector=up_vector,
    )
    start_offset, end_offset = _line_offsets(length, None, None)
    rng = np.random.default_rng(seed)
    u = rng.random(int(nrays))
    phi_u = rng.random(int(nrays))
    if radial_distribution == "area-uniform":
        impact = np.sqrt(u * (radius**2 - r_min**2) + r_min**2)
    elif radial_distribution == "radius-uniform":
        impact = r_min + u * (radius - r_min)
    else:
        raise ValueError("radial_distribution must be 'area-uniform' or 'radius-uniform'.")
    phi = 2.0 * np.pi * phi_u
    projected_x = impact * np.cos(phi)
    projected_y = impact * np.sin(phi)
    points = center + projected_x[:, None] * u_basis + projected_y[:, None] * v_basis
    starts = points + start_offset * normal
    ends = points + end_offset * normal
    metadata = {
        "type": "radial",
        "impact_parameter": impact,
        "phi": phi,
        "projected_x": projected_x,
        "projected_y": projected_y,
        "center": center,
        "basis_u": u_basis,
        "basis_v": v_basis,
        "normal": normal,
        "radial_distribution": radial_distribution,
        "seed": seed,
    }
    return MeshlessSightlineSet(np.asarray(starts), np.asarray(ends), metadata)


def generate_random_parallel_sightlines(
    center,
    width,
    height,
    nrays,
    normal_vector,
    length,
    seed=None,
    up_vector=None,
):
    """Generate random parallel sightlines inside a rectangular extent."""
    if nrays <= 0:
        raise ValueError("nrays must be positive.")
    center = _as_vector(center, "center")
    u_basis, v_basis, normal = _basis_from_plane_or_normal(
        normal_vector=normal_vector, up_vector=up_vector
    )
    rng = np.random.default_rng(seed)
    projected_x = (rng.random(int(nrays)) - 0.5) * float(width)
    projected_y = (rng.random(int(nrays)) - 0.5) * float(height)
    half = 0.5 * float(length)
    points = center + projected_x[:, None] * u_basis + projected_y[:, None] * v_basis
    starts = points - half * normal
    ends = points + half * normal
    metadata = {
        "type": "random_parallel",
        "projected_x": projected_x,
        "projected_y": projected_y,
        "center": center,
        "basis_u": u_basis,
        "basis_v": v_basis,
        "normal": normal,
        "seed": seed,
    }
    return MeshlessSightlineSet(np.asarray(starts), np.asarray(ends), metadata)


def trace_meshless_sightline_batch(
    tracer,
    starts,
    ends=None,
    directions=None,
    lengths=None,
    parallel="none",
    n_jobs=1,
    chunksize=1000,
    return_format="list",
):
    """Trace a batch of sightlines with an existing meshless tracer."""
    return tracer.trace_rays(
        starts,
        end_positions=ends,
        directions=directions,
        lengths=lengths,
        parallel=parallel,
        n_jobs=n_jobs,
        chunksize=chunksize,
        return_format=return_format,
    )
