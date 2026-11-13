# Stage 2: static PoreSpy attenuation validation

Stage 2 tests the unchanged static NeCT reconstruction on a calibrated porous medium. PoreSpy supplies only the Boolean geometry: pore voxels are explicitly assigned `0.0 mm^-1` and rock voxels `0.020 mm^-1`. No categorical `AIR/WATER/ROCK` intensities and no PoreSpy/NeCT phantom normalization are used.

## Experiment

- Shape and order: `64 × 64 × 64`, `[z,y,x]`.
- Voxel spacing: `1 mm` isotropic.
- Pore generator: `porespy.generators.blobs` with fixed seed 42, target porosity 0.35 and blobiness `[1]`.
- Acquisition: the validated Stage 1 cone geometry (`DSO=1000 mm`, `DSD=1500 mm`, `65 × 65` detector, `1.5 mm` detector pitch).
- Projections: 180 noiseless TIGRE views over `[0,2π)`.
- Reconstruction: original file-based normalization and normalized-length NeCT forward model, static HashGrid, L2 loss, no extended model and no Poisson noise.
- Default training: 100 epochs, 128 samples per ray, one detector batch per projection and a 30-minute checkpoint interval.
- Primary corrected evaluation: direct INR evaluation at voxel centers, converted using the actual legacy projection range and `max_distance_traveled`, then expressed in the TIGRE physical frame through the documented `Rz(pi)` model-to-world mapping. The projection minimum is not added to voxel attenuation.

## Coordinate convention finding

TIGRE and NeCT use the same `[z,y,x]` volume order and the generated projection file is correctly loaded as `N,H,W`. They differ at angle zero: TIGRE places the source at `+x`, detector at `-x`, with detector `u` increasing toward `+y`; NeCT places the source at `-x`, detector at `+x`, with detector columns increasing toward `-y`. A single `Rz(pi)` maps the complete ray geometry, including the detector basis. This is a fixed 180-degree angular-origin offset—not an axis swap, angle-direction reversal, or detector-only reversal.

The completed legacy Stage 2 run passed TIGRE angles unchanged to NeCT, so its model coordinates represent the TIGRE world rotated by 180 degrees around z. The public `nect.reconstruct()` API already applies this same rotation to returned static volumes; the direct checkpoint evaluator had bypassed it. The corrected reevaluation is therefore scientifically valid without retraining.

The generated metadata records the requested and actual porosity, masks, material coefficients, seed, generation parameters, git commit, worktree status and package versions.

## Submit on IDUN

The log directory is tracked in the repository, so submission requires only:

```bash
cd /cluster/work/jakobkm/NeCTv3
sbatch experiments/attenuation_validation/run_porous_idun.slurm
```

The supplied script uses:

- Repository and virtual environment: `/cluster/work/jakobkm/NeCTv3` and `.venv`.
- `GPUQ`, one A100, 8 CPUs and 64 GB RAM.
- `GCCcore/14.2.0` and `CUDA/12.8.0`.
- Exclusion of `idun-07-06`.
- Strict shell error handling and CUDA/import smoke tests.

Adjust only the requested Slurm time if the cluster rejects the 24-hour limit.

## Direct execution and resuming

From an allocated GPU node with the environment active:

```bash
python experiments/attenuation_validation/run_porous_experiment.py
```

The epoch count is configurable without starting a parameter sweep:

```bash
python experiments/attenuation_validation/run_porous_experiment.py --epochs 150
```

Use a new output directory when changing the scientific configuration. Individual stages are available for diagnosis:

```bash
python experiments/attenuation_validation/run_porous_experiment.py --stage phantom
python experiments/attenuation_validation/run_porous_experiment.py --stage projections
python experiments/attenuation_validation/run_porous_experiment.py --stage reconstruct
python experiments/attenuation_validation/run_porous_experiment.py --stage evaluate
python experiments/attenuation_validation/run_porous_experiment.py --stage reevaluate-coordinates
```

The workflow reuses stages only when their required artifacts exist. A reconstruction is considered complete only when `reconstruct()` returns successfully and `reconstruction/completed_run.json` is written. The marker is checked against the configured epoch count, projection path and final checkpoint. If no completion marker exists but NeCT saved a periodic `last.ckpt`, the newest incomplete run is resumed. Evaluation has its own completion marker and never launches reconstruction.

`reevaluate-coordinates` writes to `evaluation_coordinate_corrected/`, leaving the original `evaluation/` artifacts unchanged. To verify the ray correspondence with a deliberately asymmetric three-feature phantom and then reevaluate the existing checkpoint, submit:

```bash
sbatch experiments/attenuation_validation/run_coordinate_diagnostic_idun.slurm
```

This runs four small TIGRE projections and NeCT ray sampling only; it performs no optimization or reconstruction.

## Projection checks

Before training, Stage 2 verifies:

- N,H,W projection shape, finite values and nonnegative line integrals.
- Exact zero response for a zero-attenuation volume.
- Central rays at the four cardinal angles against independent axis-aligned voxel line sums.
- Projection units (`mm^-1 × mm`, hence dimensionless), angle count and detector centre.

The even-sized volume places the physical centre between voxels. Validation therefore records candidate line sums on both sides of the central planes and the closest axis convention rather than silently assuming a particular tie-breaking rule.

## Evaluation outputs

The default output root is:

```text
experiments/attenuation_validation/outputs/stage2_porous/
```

Important artifacts are:

```text
experiment_config.json
provenance.json
ground_truth/phantom_mu_mm_inv.npy
ground_truth/rock_mask.npy
ground_truth/pore_mask.npy
ground_truth/metadata.json
projections/projections_physical.npy
projections/validation.json
reconstruction/completed_run.json
evaluation/metrics.json
evaluation/profiles_x.csv
evaluation/volumes/reconstruction_voxel_centers_mm_inv.npy
evaluation/figures/central_slice_comparison.png
evaluation/figures/attenuation_profiles.png
evaluation/figures/distributions_and_boundary_error.png
evaluation/visualizations/porous_3d_comparison.png
evaluation/visualizations/ground_truth_rock_surface.ply
evaluation/visualizations/reconstructed_rock_surface.ply
evaluation/visualizations/porous_volumes_mm_inv.npz
evaluation_coordinate_corrected/metrics.json
evaluation_coordinate_corrected/volumes/reconstruction_model_coordinates_mm_inv.npy
evaluation_coordinate_corrected/volumes/reconstruction_voxel_centers_mm_inv.npy
```

The PLY files are thresholded surface point clouds and can be opened in MeshLab, CloudCompare or ParaView on macOS. Reconstruction uses the documented `0.010 mm^-1` midpoint threshold. The PNG uses 2× display binning for lightweight rendering; quantitative metrics always use the full `64³` arrays.

Metrics include global, rock and pore MAE/RMSE/bias/error spread, pore false-positive attenuation, phase fractions, error percentiles, and masks/statistics for one-voxel boundaries, the second boundary shell, two-voxel boundaries, and rock/pore interiors beyond two voxels. No volume is independently normalized.

## Scientific scope and later dynamic extension

This remains a noiseless synthetic baseline. Errors can reflect INR capacity, convergence, NeCT ray sampling, limited resolution, boundaries, legacy projection normalization and coordinate alignment; they do not establish calibration accuracy for experimental CT data.

A later dynamic experiment can reuse the saved PoreSpy geometry and `PorousMediumPhantom.get_target()` invasion sequence, but should replace its categorical/scaled output with explicit air/fluid/rock coefficients at every time. For a new run, pass `(tigre_angle + pi) mod 2pi` to NeCT with `nect_angles_for_tigre_world`, while retaining acquisition order and timestamps. The learned model coordinates then coincide directly with TIGRE world coordinates and no output rotation is needed. Do not combine shifted training angles with the legacy output mapping. TIGRE projections would then be generated per acquisition time and reconstructed with MixedCubes. That extension also needs temporal ground-truth masks, phase-wise metrics over time and a controlled acquisition schedule; none of those dynamic reconstruction changes are included in Stage 2.
