"""
Meshless Voronoi ray tracing utilities.

This module implements a mesh-free ray walker for Voronoi tessellations.  It
is intended for particle/cell based simulations where the gas elements are
represented by mesh-generating sites, for example AREPO gas cells or SPH/MFM
particles interpreted as Voronoi generators.

The implementation follows the SALSA meshless Voronoi ray-tracing idea: do not
construct the full Voronoi mesh.  Instead, build a nearest-neighbour tree over
site positions and advance a straight ray from one Voronoi cell to the next by
finding the next face crossing from nearest-site queries and ray/face-plane
intersections.

The output is deliberately simple: ordered cell/site indices and the geometric
path length through each site's Voronoi cell.  The caller can then sample any
simulation field at those indices and build spectra using Trident's existing
spectrum-generation machinery.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - exercised only without scipy
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class MeshlessVoronoiRay:
    """Container for a traced Voronoi ray.

    Parameters
    ----------
    indices : numpy.ndarray
        Ordered integer indices of Voronoi generating sites intersected by the
        ray.
    dl : numpy.ndarray
        Ordered path lengths through each corresponding Voronoi cell, in the
        same length units as the input positions.
    start_position : numpy.ndarray
        Ray start position.
    end_position : numpy.ndarray
        Ray end position.
    direction : numpy.ndarray
        Unit direction vector from start to end.
    length : float
        Total requested ray length.
    start_index, end_index : int
        Nearest Voronoi site indices at the ray start and final point.
    failed_stack_recoveries : int
        Number of times a failed candidate seeded a later crossing search.
    fallback_count : int
        Number of conservative bisection fallbacks used.
    nudge_count : int
        Number of numerical near-face nudges used to avoid zero-length loops.
    """

    indices: np.ndarray
    dl: np.ndarray
    start_position: np.ndarray
    end_position: np.ndarray
    direction: np.ndarray
    length: float
    start_index: int = -1
    end_index: int = -1
    failed_stack_recoveries: int = 0
    fallback_count: int = 0
    nudge_count: int = 0
    eps: float = np.nan
    periodic: bool = False
    box_size: Optional[np.ndarray] = None
    algorithm_version: str = "salsa_meshless_voronoi_v1"

    @property
    def cumulative_dl(self) -> np.ndarray:
        """Cumulative distance at the end of each ray segment."""
        return np.cumsum(self.dl)

    @property
    def metadata(self) -> dict:
        """Return scalar diagnostics suitable for storing with a ray file."""
        return {
            "algorithm_version": self.algorithm_version,
            "ray_start": self.start_position,
            "ray_end": self.end_position,
            "ray_direction": self.direction,
            "ray_length": self.length,
            "nearest_index_start": self.start_index,
            "nearest_index_end": self.end_index,
            "number_of_cell_crossings": max(len(self.indices) - 1, 0),
            "failed_stack_recoveries": self.failed_stack_recoveries,
            "fallback_count": self.fallback_count,
            "nudge_count": self.nudge_count,
            "epsilon": self.eps,
            "periodic": self.periodic,
            "box_size": self.box_size,
        }


@dataclass(frozen=True)
class MeshlessVoronoiRayBatch:
    """Compact ragged container for many meshless Voronoi rays."""

    ray_offsets: np.ndarray
    indices: np.ndarray
    dl: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray
    failed_stack_recoveries: np.ndarray
    fallback_counts: np.ndarray
    nudge_counts: np.ndarray
    eps: float = np.nan
    periodic: bool = False
    box_size: Optional[np.ndarray] = None
    algorithm_version: str = "salsa_meshless_voronoi_v1"

    def __len__(self) -> int:
        return max(len(self.ray_offsets) - 1, 0)

    @property
    def n_rays(self) -> int:
        """Number of rays stored in the batch."""
        return len(self)

    @property
    def n_segments(self) -> np.ndarray:
        """Number of flattened segments belonging to each ray."""
        return np.diff(self.ray_offsets)

    @property
    def cumulative_dl(self) -> np.ndarray:
        """Per-ray cumulative segment distances in flattened order."""
        cumulative = np.empty_like(self.dl)
        for ray_id in range(len(self)):
            start = self.ray_offsets[ray_id]
            end = self.ray_offsets[ray_id + 1]
            cumulative[start:end] = np.cumsum(self.dl[start:end])
        return cumulative

    def ray(self, ray_id: int) -> MeshlessVoronoiRay:
        """Return a single ray from the batch as ``MeshlessVoronoiRay``."""
        if ray_id < 0 or ray_id >= len(self):
            raise IndexError("ray_id out of range")
        start = self.ray_offsets[ray_id]
        end = self.ray_offsets[ray_id + 1]
        return MeshlessVoronoiRay(
            indices=np.asarray(self.indices[start:end], dtype=np.int64),
            dl=np.asarray(self.dl[start:end], dtype=np.float64),
            start_position=np.asarray(self.start_positions[ray_id], dtype=np.float64),
            end_position=np.asarray(self.end_positions[ray_id], dtype=np.float64),
            direction=np.asarray(self.directions[ray_id], dtype=np.float64),
            length=float(self.lengths[ray_id]),
            start_index=int(self.start_indices[ray_id]),
            end_index=int(self.end_indices[ray_id]),
            failed_stack_recoveries=int(self.failed_stack_recoveries[ray_id]),
            fallback_count=int(self.fallback_counts[ray_id]),
            nudge_count=int(self.nudge_counts[ray_id]),
            eps=float(self.eps),
            periodic=bool(self.periodic),
            box_size=None if self.box_size is None else np.asarray(self.box_size),
            algorithm_version=self.algorithm_version,
        )

    def to_list(self) -> list[MeshlessVoronoiRay]:
        """Return all rays as a list of ``MeshlessVoronoiRay`` objects."""
        return [self.ray(ray_id) for ray_id in range(len(self))]


class MeshlessVoronoiRayTracer:
    """Mesh-free ray tracer for Voronoi tessellations.

    Parameters
    ----------
    positions : array_like, shape (N, 3)
        Voronoi generating-site positions.
    box_size : float or array_like, optional
        Periodic box size.  If supplied, nearest-neighbour queries are periodic
        and displacement vectors use the minimum-image convention.  This maps
        directly to scipy.spatial.cKDTree(boxsize=...).  All positions should
        lie in [0, box_size) for periodic runs.
    leafsize : int, optional
        KD-tree leaf size.
    eps : float, optional
        Absolute geometric tolerance used around Voronoi faces.  If omitted, a
        scale-aware default is chosen from the input coordinate range.
    max_iter : int, optional
        Safety limit on the number of Voronoi-cell crossings per ray.
    max_bisect_iter : int, optional
        Safety limit on the internal search for the next natural neighbour.
    """

    def __init__(
        self,
        positions: np.ndarray,
        box_size: Optional[Iterable[float]] = None,
        leafsize: int = 32,
        eps: Optional[float] = None,
        max_iter: int = 1_000_000,
        max_bisect_iter: int = 256,
    ) -> None:
        if cKDTree is None:
            raise ImportError(
                "MeshlessVoronoiRayTracer requires scipy.spatial.cKDTree."
            ) from _SCIPY_IMPORT_ERROR

        self.positions = np.asarray(positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")
        if len(self.positions) == 0:
            raise ValueError("positions cannot be empty.")

        if box_size is None:
            self.box_size = None
            tree_box = None
        else:
            self.box_size = np.asarray(box_size, dtype=np.float64)
            if self.box_size.ndim == 0:
                self.box_size = np.repeat(float(self.box_size), 3)
            if self.box_size.shape != (3,):
                raise ValueError("box_size must be scalar or length-3.")
            if np.any(self.box_size <= 0):
                raise ValueError("box_size entries must be positive.")
            self.positions = np.mod(self.positions, self.box_size)
            tree_box = self.box_size

        span = np.ptp(self.positions, axis=0)
        scale = float(np.max(span)) if np.max(span) > 0 else 1.0
        self.eps = float(eps) if eps is not None else max(1.0e-10 * scale, 1.0e-12)
        self.max_iter = int(max_iter)
        self.max_bisect_iter = int(max_bisect_iter)
        self.tree = cKDTree(self.positions, leafsize=leafsize, boxsize=tree_box)

    def nearest_index(self, point: np.ndarray) -> int:
        """Return index of the Voronoi site nearest to point."""
        p = self._wrap_point(np.asarray(point, dtype=np.float64))
        return int(self.tree.query(p, k=1)[1])

    def nearest_indices(self, points: np.ndarray, workers: int = 1) -> np.ndarray:
        """Return nearest Voronoi site indices for many points."""
        p = self._wrap_point(np.asarray(points, dtype=np.float64))
        try:
            return np.asarray(self.tree.query(p, k=1, workers=workers)[1], dtype=np.int64)
        except TypeError:  # scipy versions before the workers keyword
            return np.asarray(self.tree.query(p, k=1)[1], dtype=np.int64)

    def trace_rays(
        self,
        start_positions: np.ndarray,
        end_positions: Optional[np.ndarray] = None,
        directions: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,
        parallel: str = "none",
        n_jobs: int = 1,
        chunksize: Optional[int] = None,
        return_format: str = "list",
    ):
        """Trace many rays while reusing this tracer's nearest-neighbor tree."""
        using_end_positions = end_positions is not None
        starts, ends, dirs, lens = self._prepare_batch_inputs(
            start_positions, end_positions, directions, lengths
        )
        nrays = len(starts)
        if chunksize is None or chunksize <= 0:
            chunksize = nrays if nrays else 1

        all_rays = []
        for first in range(0, nrays, chunksize):
            last = min(first + chunksize, nrays)
            if parallel == "none" or n_jobs == 1:
                if using_end_positions:
                    rays = [
                        self.trace_ray(starts[i], end_position=ends[i])
                        for i in range(first, last)
                    ]
                else:
                    rays = [
                        self.trace_ray(starts[i], direction=dirs[i], length=lens[i])
                        for i in range(first, last)
                    ]
            elif parallel == "threads":
                if using_end_positions:
                    def _trace(item):
                        return self.trace_ray(item[0], end_position=item[1])
                    iterable = zip(starts[first:last], ends[first:last])
                else:
                    def _trace(item):
                        return self.trace_ray(item[0], direction=item[1], length=item[2])
                    iterable = zip(starts[first:last], dirs[first:last], lens[first:last])
                with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
                    rays = list(executor.map(_trace, iterable))
            else:
                raise ValueError("parallel must be 'none' or 'threads'.")
            all_rays.extend(rays)

        if return_format == "list":
            return all_rays
        if return_format == "ragged":
            return self._rays_to_batch(all_rays, starts, ends, dirs, lens)
        raise ValueError("return_format must be 'list' or 'ragged'.")

    def trace_ray(
        self,
        start_position: np.ndarray,
        end_position: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        length: Optional[float] = None,
    ) -> MeshlessVoronoiRay:
        """Trace a straight ray through the implicit Voronoi tessellation.

        Supply either ``start_position`` and ``end_position`` or
        ``start_position``, ``direction``, and ``length``.
        """
        r0 = np.asarray(start_position, dtype=np.float64)
        if r0.shape != (3,):
            raise ValueError("start_position must be length-3.")

        if end_position is not None:
            r1 = np.asarray(end_position, dtype=np.float64)
            if r1.shape != (3,):
                raise ValueError("end_position must be length-3.")
            delta = self._minimum_image(r1 - r0)
            total_length = float(np.linalg.norm(delta))
            if total_length <= 0:
                raise ValueError("Ray length must be positive.")
            ray_hat = delta / total_length
        else:
            if direction is None or length is None:
                raise ValueError(
                    "Provide end_position or both direction and length."
                )
            ray_hat = np.asarray(direction, dtype=np.float64)
            if ray_hat.shape != (3,):
                raise ValueError("direction must be length-3.")
            norm = float(np.linalg.norm(ray_hat))
            if norm <= 0:
                raise ValueError("direction must be non-zero.")
            ray_hat = ray_hat / norm
            total_length = float(length)
            if total_length <= 0:
                raise ValueError("length must be positive.")
            r1 = r0 + ray_hat * total_length

        r = self._wrap_point(r0)
        rf = self._wrap_point(r0 + ray_hat * total_length)
        i_cur = self.nearest_index(r)
        i_final = self.nearest_index(rf)

        indices = []
        lengths = []
        travelled = 0.0
        failed_stack: list[Tuple[int, float]] = []
        failed_stack_recoveries = 0
        fallback_count = 0
        nudge_count = 0

        for _ in range(self.max_iter):
            remaining = total_length - travelled
            if remaining <= self.eps:
                break

            if i_cur == i_final:
                indices.append(i_cur)
                lengths.append(remaining)
                travelled = total_length
                break

            step, i_next, used_failed_stack, used_fallback = self._next_cell_crossing(
                r, ray_hat, i_cur, remaining, failed_stack
            )
            failed_stack_recoveries += int(used_failed_stack)
            fallback_count += int(used_fallback)

            if not np.isfinite(step) or step <= self.eps:
                # Numerical degeneracy near a face or vertex.  Nudge forward
                # and re-query rather than allowing a zero-length loop.
                nudge = min(max(10.0 * self.eps, 1.0e-12), remaining)
                r = self._wrap_point(r + ray_hat * nudge)
                travelled += nudge
                i_cur = self.nearest_index(r)
                nudge_count += 1
                continue

            step = min(step, remaining)
            indices.append(i_cur)
            lengths.append(step)
            r = self._wrap_point(r + ray_hat * step)
            travelled += step
            i_cur = i_next

        else:
            raise RuntimeError(
                "Meshless Voronoi ray tracing exceeded max_iter. "
                "Increase max_iter or inspect the geometry/tolerance."
            )

        if travelled < total_length - self.eps:
            remaining = total_length - travelled
            indices.append(i_cur)
            lengths.append(remaining)
            travelled = total_length

        # Merge adjacent duplicate cells created only by numerical nudges.
        out_i, out_dl = self._merge_segments(indices, lengths)
        return MeshlessVoronoiRay(
            indices=np.asarray(out_i, dtype=np.int64),
            dl=np.asarray(out_dl, dtype=np.float64),
            start_position=np.asarray(r0, dtype=np.float64),
            end_position=np.asarray(r0 + ray_hat * total_length, dtype=np.float64),
            direction=np.asarray(ray_hat, dtype=np.float64),
            length=float(total_length),
            start_index=int(self.nearest_index(r0)),
            end_index=int(i_final),
            failed_stack_recoveries=int(failed_stack_recoveries),
            fallback_count=int(fallback_count),
            nudge_count=int(nudge_count),
            eps=float(self.eps),
            periodic=self.box_size is not None,
            box_size=None if self.box_size is None else np.asarray(self.box_size),
        )

    def _prepare_batch_inputs(
        self,
        start_positions: np.ndarray,
        end_positions: Optional[np.ndarray],
        directions: Optional[np.ndarray],
        lengths: Optional[np.ndarray],
    ):
        starts = np.ascontiguousarray(start_positions, dtype=np.float64)
        if starts.ndim != 2 or starts.shape[1] != 3:
            raise ValueError("start_positions must have shape (M, 3).")
        if end_positions is not None:
            ends = np.ascontiguousarray(end_positions, dtype=np.float64)
            if ends.shape != starts.shape:
                raise ValueError("end_positions must have shape (M, 3).")
            deltas = self._minimum_image(ends - starts)
            lens = np.linalg.norm(deltas, axis=1)
            if np.any(lens <= 0):
                raise ValueError("All ray lengths must be positive.")
            dirs = deltas / lens[:, None]
            return starts, starts + dirs * lens[:, None], dirs, lens

        if directions is None or lengths is None:
            raise ValueError("Provide end_positions or both directions and lengths.")
        dirs = np.ascontiguousarray(directions, dtype=np.float64)
        if dirs.shape != starts.shape:
            raise ValueError("directions must have shape (M, 3).")
        norm = np.linalg.norm(dirs, axis=1)
        if np.any(norm <= 0):
            raise ValueError("All directions must be non-zero.")
        dirs = dirs / norm[:, None]
        lens = np.ascontiguousarray(lengths, dtype=np.float64)
        if lens.shape != (len(starts),):
            raise ValueError("lengths must have shape (M,).")
        if np.any(lens <= 0):
            raise ValueError("All lengths must be positive.")
        ends = starts + dirs * lens[:, None]
        return starts, ends, dirs, lens

    def _rays_to_batch(self, rays, starts, ends, directions, lengths):
        offsets = np.zeros(len(rays) + 1, dtype=np.int64)
        for i, ray in enumerate(rays):
            offsets[i + 1] = offsets[i] + len(ray.indices)
        total = int(offsets[-1])
        indices = np.empty(total, dtype=np.int64)
        dl = np.empty(total, dtype=np.float64)
        for i, ray in enumerate(rays):
            first = offsets[i]
            last = offsets[i + 1]
            indices[first:last] = ray.indices
            dl[first:last] = ray.dl

        return MeshlessVoronoiRayBatch(
            ray_offsets=offsets,
            indices=indices,
            dl=dl,
            start_positions=np.asarray(starts, dtype=np.float64),
            end_positions=np.asarray(ends, dtype=np.float64),
            directions=np.asarray(directions, dtype=np.float64),
            lengths=np.asarray(lengths, dtype=np.float64),
            start_indices=np.asarray([ray.start_index for ray in rays], dtype=np.int64),
            end_indices=np.asarray([ray.end_index for ray in rays], dtype=np.int64),
            failed_stack_recoveries=np.asarray(
                [ray.failed_stack_recoveries for ray in rays], dtype=np.int64
            ),
            fallback_counts=np.asarray([ray.fallback_count for ray in rays], dtype=np.int64),
            nudge_counts=np.asarray([ray.nudge_count for ray in rays], dtype=np.int64),
            eps=float(self.eps),
            periodic=self.box_size is not None,
            box_size=None if self.box_size is None else np.asarray(self.box_size),
        )

    def _next_cell_crossing(
        self,
        r: np.ndarray,
        ray_hat: np.ndarray,
        i_cur: int,
        remaining: float,
        failed_stack: list[Tuple[int, float]],
    ) -> Tuple[float, int, bool, bool]:
        """Find the next Voronoi face crossing without mesh construction."""
        x_cur = self.positions[i_cur]
        L = 0.0
        R = remaining
        i_end = -1
        used_failed_stack = False

        # SALSA keeps failed candidates in a stack to seed future searches.
        # Remove self-candidates and try the nearest remaining failed distance.
        failed_stack[:] = [(idx, dist) for idx, dist in failed_stack if idx != i_cur]
        if failed_stack:
            i_end, dist = min(failed_stack, key=lambda item: item[1])
            failed_stack.remove((i_end, dist))
            R = min(remaining, max(2.0 * dist, self.eps))
            used_failed_stack = True

        dl_local = np.inf
        for n in range(self.max_bisect_iter):
            l_cen = 0.5 * (L + R)
            r_end = self._wrap_point(r + ray_hat * l_cen)

            if n > 0 or i_end < 0:
                i_end = self.nearest_index(r_end)

            if i_end == i_cur:
                if R - L > self.eps:
                    L = l_cen
                else:
                    R = min(remaining, R + self.eps)
                    if R <= L + self.eps:
                        R = min(remaining, L + 2.0 * self.eps)
                continue

            x_end = self.positions[i_end]
            dl_local = self._intersect_face_plane(r, ray_hat, x_cur, x_end, dl_local)
            if not np.isfinite(dl_local):
                L = l_cen
                continue

            r_cand = self._wrap_point(r + ray_hat * dl_local)
            i_cand = self.nearest_index(r_cand)

            if i_cand not in (i_cur, i_end):
                l_search = float(np.dot(self._minimum_image(r_cand - r), ray_hat))
                if l_search > L + self.eps:
                    R = min(remaining, 2.0 * l_search - L)
                    failed_stack.append((i_cand, min(remaining, l_cen + self.eps)))
                    dl_local = np.inf
                    continue

            if dl_local < -self.eps:
                L = l_cen
                dl_local = np.inf
                continue

            return max(dl_local, 0.0), i_end, used_failed_stack, False

        # Conservative fallback: binary search the first point that no longer
        # belongs to the current cell, then return the cell at that boundary.
        lo, hi = 0.0, remaining
        for _ in range(128):
            mid = 0.5 * (lo + hi)
            i_mid = self.nearest_index(r + ray_hat * mid)
            if i_mid == i_cur:
                lo = mid
            else:
                hi = mid
        i_next = self.nearest_index(r + ray_hat * hi)
        return hi, i_next, used_failed_stack, True

    def _intersect_face_plane(
        self,
        r: np.ndarray,
        ray_hat: np.ndarray,
        x_cur: np.ndarray,
        x_end: np.ndarray,
        dl_local: float,
    ) -> float:
        """Ray intersection with the candidate Voronoi face plane.

        The face between two Voronoi sites lies on the perpendicular bisector
        plane.  The midpoint is on the plane and ``x_end - x_cur`` is its
        normal.  This helper mirrors Algorithm 2 in the SALSA paper.
        """
        q = self._minimum_image(x_end - x_cur)
        midpoint = self._wrap_point(x_cur + 0.5 * q)
        c = self._minimum_image(midpoint - r)
        cq = float(np.dot(c, q))
        hq = float(np.dot(ray_hat, q))

        if abs(hq) <= self.eps:
            return dl_local

        if cq > 0:
            s = cq / hq
        else:
            if hq > 0:
                s = 0.0
            else:
                s = np.inf

        if 0.0 <= s <= dl_local:
            return float(s)
        return dl_local

    def _wrap_point(self, point: np.ndarray) -> np.ndarray:
        if self.box_size is None:
            return np.asarray(point, dtype=np.float64)
        return np.mod(point, self.box_size)

    def _minimum_image(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float64)
        if self.box_size is None:
            return delta
        return delta - self.box_size * np.round(delta / self.box_size)

    @staticmethod
    def _merge_segments(indices, lengths):
        out_i = []
        out_dl = []
        for idx, dl in zip(indices, lengths):
            if dl <= 0:
                continue
            if out_i and idx == out_i[-1]:
                out_dl[-1] += float(dl)
            else:
                out_i.append(int(idx))
                out_dl.append(float(dl))
        return out_i, out_dl


def meshless_voronoi_ray(
    positions: np.ndarray,
    start_position: np.ndarray,
    end_position: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    length: Optional[float] = None,
    box_size: Optional[Iterable[float]] = None,
    **tracer_kwargs,
) -> MeshlessVoronoiRay:
    """Convenience function for tracing a single meshless Voronoi ray.

    Examples
    --------
    >>> ray = meshless_voronoi_ray(pos, [0, 0, 0], [1, 1, 1])
    >>> ray.indices
    array([...])
    >>> ray.dl
    array([...])
    """
    tracer = MeshlessVoronoiRayTracer(
        positions=positions, box_size=box_size, **tracer_kwargs
    )
    return tracer.trace_ray(
        start_position=start_position,
        end_position=end_position,
        direction=direction,
        length=length,
    )
