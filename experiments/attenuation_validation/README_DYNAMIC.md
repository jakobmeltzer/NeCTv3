# Stage 3: dynamic PoreSpy attenuation validation

Stage 3 reuses the Stage-2 `64³` pore geometry, generates a monotonic IBIP invasion from the `z=0` face, projects the changing object once per acquisition event with TIGRE, and reconstructs it with NeCT MixedCubes.

## Scientific conventions

- Arrays and masks are `[z,y,x]`; projections are `N,H,W`.
- Attenuation is assigned explicitly: rock `0.020`, fluid `0.010`, air `0.0 mm⁻¹`.
- IBIP supplies discrete occupancy order only. Stored states use right-continuous lookup and never interpolate attenuation.
- Default acquisition: 720 projections over four revolutions and 120 nominal seconds. Timestamps are normalized to `[0,1]`; no physical invasion speed is claimed.
- Every projection uses the state at its own timestamp. Static scans are not concatenated.
- NeCT receives `(theta_TIGRE + pi) mod 2pi`, preserving projection and timestamp order. Evaluation therefore applies no volume rotation.
- Projections are ideal instantaneous line integrals. Noise, fly-scan integration, detector effects, regularizers, and the extended forward model are disabled.
- L2 is used for direct comparability with Stage 2 and to penalize quantitative attenuation deviations; loss/regularization comparisons are intentionally deferred.

## Submit on IDUN

```bash
cd /cluster/work/jakobkm/NeCTv3
sbatch experiments/attenuation_validation/run_dynamic_idun.slurm
```

The default pilot uses MixedCubes, L2 loss, 30 epochs, 128 samples per ray and one detector batch per projection. It performs 21,600 projection-training steps (roughly 10.8 billion sampled ray points at the configured batch size). Runtime is hardware and I/O dependent; reserve the supplied 24 hours initially and use the completion/resume behavior if necessary.

For a lightweight end-to-end IDUN test, from an allocated GPU node:

```bash
python -u experiments/attenuation_validation/run_dynamic_experiment.py --smoke
```

This uses 24 projections, one revolution, seven stored states and two epochs. It is a pipeline check, not a scientifically adequate reconstruction.

Stages can be run independently:

```bash
python -u experiments/attenuation_validation/run_dynamic_experiment.py --stage phantom
python -u experiments/attenuation_validation/run_dynamic_experiment.py --stage projections
python -u experiments/attenuation_validation/run_dynamic_experiment.py --stage reconstruct
python -u experiments/attenuation_validation/run_dynamic_experiment.py --stage evaluate
```

The runner validates explicit completion markers. An interrupted checkpoint is resumed, but a checkpoint alone is never treated as completion. Ground truth, projections, reconstruction and evaluation live below `outputs/stage3_dynamic/`; Stage 1 and Stage 2 outputs are not modified.

## Outputs

- `ground_truth/`: fixed rock/pore/inlet masks, IBIP sequence and normalized invasion time, stored fluid masks, saturation and metadata.
- `projections/`: physical projections, TIGRE and corrected NeCT angles, normalized/second timestamps, state indices and validation.
- `reconstruction/completed_run.json`: verified checkpoint provenance.
- `evaluation/metrics.json`: time-resolved material, segmentation, saturation, interface, interior and outer-boundary metrics.
- `evaluation/volumes/`: quantitative ground-truth/reconstruction frames and a compressed 4-D series.
- `evaluation/figures/`: slices, signed errors, saturation, temporal profiles, material distributions, segmentation and training loss.
- `evaluation/visualizations/`: ground-truth and reconstructed fluid surface point clouds at every evaluation time.

Known limitations include discrete state timing, ideal projections, only six quantitative evaluation times, threshold-based fluid segmentation, possible unreachable pores, and no physical capillary time model. The known-pore segmentation is an evaluation diagnostic only; the reconstruction never receives the pore mask.
