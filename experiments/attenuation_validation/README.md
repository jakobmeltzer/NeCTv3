# Stage 1: calibrated cube with legacy NeCT

This experiment tests whether the unchanged static NeCT pipeline recovers a known attenuation coefficient from projections generated independently with TIGRE.

## Specification

- Volume: `64 × 64 × 64`, explicitly `[z,y,x]`.
- Voxel size: `1 mm` isotropic.
- Cube: centred, `20 mm` side, `0.010 mm^-1`.
- Background: `0.0 mm^-1`.
- Acquisition: cone beam, `DSO=1000 mm`, `DSD=1500 mm`.
- Detector: `65 × 65`, `1.5 mm` pitch. This corresponds to `1 mm` at isocentre and gives an exact central detector pixel.
- Views: 180 over `[0,2π)` by default.

TIGRE projections are saved without normalization. NeCT is intentionally given the path to the single `.npy` file so the original file-based min–max loader and original normalized-length forward model are exercised.

## Run on IDUN

Activate the existing CUDA NeCT environment, ensure PyTigre is installed, then run from the repository root:

```bash
python experiments/attenuation_validation/run_experiment.py
```

The command generates and validates the phantom/projections, reconstructs, evaluates the checkpoint, invokes the existing static exporter, and creates figures. It reuses completed artifacts on rerun. If NeCT saved `last.ckpt` during an interrupted run but the experiment did not reach its completion marker, the reconstruction stage automatically resumes the newest checkpoint beneath this experiment's reconstruction directory.

To run or repeat individual stages:

```bash
python experiments/attenuation_validation/run_experiment.py --stage phantom
python experiments/attenuation_validation/run_experiment.py --stage projections
python experiments/attenuation_validation/run_experiment.py --stage reconstruct
python experiments/attenuation_validation/run_experiment.py --stage evaluate
```

Useful options:

```bash
python experiments/attenuation_validation/run_experiment.py --epochs 20 --num-angles 180
python experiments/attenuation_validation/run_experiment.py --output-dir /path/on/idun/stage1_cube
```

`--force` regenerates lightweight phantom/projection inputs but deliberately does not delete checkpoints. Use a new output directory for a genuinely independent reconstruction.

The optional `run_idun.slurm` is an example only; its account, partition, modules and environment activation must be adapted to the local IDUN setup.

## Projection validation

Before reconstruction, the script checks:

- Output order and shape are `[projection,height,width]`.
- Every value is finite and nonnegative (within numerical tolerance).
- The angle-zero central ray is within 5% of `0.010 × 20 = 0.2`.
- Detector-corner rays miss the cube.
- Angles and all geometry/material units are recorded.

A validation failure stops before the expensive reconstruction.

## Outputs

The default root is:

```text
experiments/attenuation_validation/outputs/stage1_cube/
```

Important outputs include:

```text
experiment_config.json
ground_truth/phantom_mu_mm_inv.npy
ground_truth/cube_mask.npy
ground_truth/metadata.json
projections/projections_physical.npy
projections/angles_rad.npy
projections/validation.json
reconstruction/completed_run.json
evaluation/metrics.json
evaluation/profile_x.csv
evaluation/volumes/*.npy
figures/attenuation_profile_x.png
figures/central_slices_and_error.png
figures/training_loss.png
visualizations/cube_isosurfaces.png
visualizations/volumes_mm_inv.npz
```

`volumes_mm_inv.npz` contains both full 3D arrays for local inspection in Python, Fiji (after conversion), ParaView, napari, or another volume viewer. The PNG is a reproducible fixed-threshold isosurface overview; both volumes use the same `0.005 mm^-1` threshold.

## Values compared

Evaluation keeps four quantities separate:

1. Calibrated ground truth in `mm^-1`.
2. Raw network output in NeCT's internal legacy scale.
3. Direct INR output converted using the actual file normalization and `max_distance_traveled` convention.
4. Output produced by NeCT's existing standalone static volume exporter.

No independent image normalization is applied. The standalone exporter samples endpoints, whereas direct INR evaluation uses the same voxel-centre coordinates as `BaseTrainer.create_volume`; this distinction is recorded in `metrics.json`.

## Limitations of this baseline

- It intentionally preserves the legacy normalization/export behavior.
- TIGRE interpolation makes the central ray integral approximate rather than exactly analytic.
- The lightweight 20-epoch, one-ray-batch-per-projection configuration is a pilot, not a convergence study.
- Resume depends on NeCT having saved a full `last.ckpt`; very early interruption (before NeCT's checkpoint threshold) starts a new run while reusing the phantom and projections.

