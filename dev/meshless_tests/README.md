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
