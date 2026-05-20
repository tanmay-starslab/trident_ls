# Meshless Voronoi Diagnostics

These scripts compare Trident's original yt `LightRay` ray construction with
the SALSA-style meshless Voronoi ray construction implemented in this fork.
They keep Trident's existing `SpectrumGenerator` unchanged and focus on
whether the sampled ray fields, derived redshift/velocity quantities, ion
columns, and final spectra look physically sensible.

## Main Diagnostic Command

Run from the repository root:

```bash
/Users/wavefunction/github_repos/m61-tng/.venv/bin/python \
    dev/meshless_tests/plot_meshless_vs_lightray_diagnostics.py \
    --dataset "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5" \
    --output-dir /private/tmp/trident_meshless_diagnostics \
    --nrays 8 \
    --seed 398784 \
    --instrument COS-G130M \
    --periodic false \
    --overwrite
```

For a quicker smoke test, use `--nrays 3`.

## Expected Outputs

The script writes all diagnostics to the selected output directory:

- `ray_000_ray_diagnostics.png`, etc.: ray-field comparisons versus cumulative path length.
- `ray_000_spectrum_diagnostics.png`, etc.: spectrum and residual comparisons.
- `all_rays_summary.png`: aggregate path length, element count, column-like, EW, and flux-difference comparisons.
- `diagnostics_summary.csv` and `diagnostics_summary.json`: machine-readable metrics.
- Intermediate LightRay, meshless ray, and spectrum HDF5 files.

## Interpreting Differences

LightRay and meshless Voronoi rays are not expected to match element-by-element.
The yt LightRay samples the dataset-native geometry, while the meshless path
walks the implicit Voronoi tessellation defined by gas generating sites.  Focus
on whether total path lengths are consistent, field trends are plausible, ion
column-like integrals are reasonable, and spectra are generated successfully by
the same `SpectrumGenerator` path.

Large spectral or field differences can be scientifically meaningful rather
than implementation errors, especially when the number of ray elements differs.
Use the per-ray diagnostic plots to inspect whether those differences come from
different sampled gas cells, path length distributions, velocity structure, or
ion density fields.

## Output Hygiene

Generated HDF5 files, plots, logs, summaries, and caches are ignored by git.
Do not commit generated diagnostics or the local TNG cutout.

## Profiling

Geometry-only profiling separates dataset load, position extraction, KDTree
construction, and ray walking:

```bash
/Users/wavefunction/github_repos/m61-tng/.venv/bin/python \
    dev/meshless_tests/profile_meshless_raytracing.py \
    --mode synthetic \
    --npoints 1000 10000 \
    --nrays 1 10 100 \
    --output-dir /private/tmp/trident_meshless_profile \
    --profile none
```

For the local TNG cutout:

```bash
/Users/wavefunction/github_repos/m61-tng/.venv/bin/python \
    dev/meshless_tests/profile_meshless_raytracing.py \
    --mode tng \
    --dataset "/Users/wavefunction/ASU Dropbox/Tanmay Singh/M61/data/cutout_398784.hdf5" \
    --nrays 1 5 \
    --output-dir /private/tmp/trident_meshless_profile_tng \
    --periodic false
```

## Batch Sightlines And Catalogs

The optimized path keeps the single-ray API unchanged and adds tree-reusing
batch helpers:

```python
import trident

sightlines = trident.generate_uniform_grid_sightlines(
    center=[0, 0, 0],
    width=200,
    height=200,
    nx=20,
    ny=20,
    plane="xy",
    length=300,
)

catalog = trident.make_meshless_voronoi_ray_catalog(
    ds,
    sightlines.starts,
    ends=sightlines.ends,
    lines=["H I", "O VI"],
    output_filename="/private/tmp/meshless_catalog.h5",
    periodic=False,
    overwrite=True,
)
```

Radial sightlines around a galaxy center use area-uniform impact parameters by
default:

```python
sightlines = trident.generate_radial_sightlines(
    center=galaxy_center,
    radius=150,
    nrays=1000,
    plane="xy",
    length=300,
    seed=398784,
)
```

Threaded batch tracing is optional:

```python
tracer.trace_rays(starts, end_positions=ends, parallel="threads", n_jobs=4)
```

Use `parallel="none"` for the most conservative behavior.  Threaded mode is
tested to match serial batch tracing on deterministic synthetic cases.

## Many-Ray Benchmarks

Benchmark the old per-ray tree rebuild path against batch serial/threaded
tracing and catalog writing:

```bash
/Users/wavefunction/github_repos/m61-tng/.venv/bin/python \
    dev/meshless_tests/benchmark_many_sightlines.py \
    --mode synthetic \
    --npoints 10000 \
    --nrays 1 10 100 \
    --output-dir /private/tmp/trident_meshless_benchmarks
```

The script writes CSV/JSON summaries and runtime plots under the selected
output directory.  Those outputs are ignored and should not be committed.
