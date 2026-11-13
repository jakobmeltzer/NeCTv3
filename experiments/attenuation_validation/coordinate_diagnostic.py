#!/usr/bin/env python3
"""Small TIGRE-versus-NeCT ray-coordinate diagnostic (GPU, no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.attenuation_validation.common import write_json  # noqa: E402
from experiments.attenuation_validation.coordinates import (  # noqa: E402
    tigre_world_volume_to_nect_model,
)
from experiments.attenuation_validation.porous_config import porous_experiment_config  # noqa: E402
from experiments.attenuation_validation.projections import project_with_tigre  # noqa: E402


def _asymmetric_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Three non-overlapping cuboids identify all positive world axes."""
    volume = np.zeros(shape, dtype=np.float32)
    volume[18:22, 7:11, 21:26] = 0.010   # +z, -y, +x
    volume[5:10, 20:25, 13:16] = 0.017  # -z, +y, near x=0
    volume[13:16, 14:18, 4:9] = 0.025   # near z/y=0, -x
    return volume


def _nect_forward(volume_model_zyx: np.ndarray, config: dict, angles: np.ndarray) -> np.ndarray:
    import torch
    from nect.sampling import Geometry, Projector

    geom = config["geometry"]
    vol = config["volume"]
    geometry = Geometry(
        mode="cone",
        DSO=geom["DSO_mm"],
        DSD=geom["DSD_mm"],
        nDetector=geom["nDetector_hw"],
        dDetector=geom["dDetector_hw_mm"],
        nVoxel=vol["shape_zyx"],
        dVoxel=vol["voxel_size_zyx_mm"],
        offOrigin=geom["offOrigin_zyx_mm"],
        offDetector=geom["offDetector_hw_mm"],
        COR=geom["COR_mm"],
        angles=angles,
    )
    height, width = geometry.nDetector
    rays = height * width
    points_per_ray = 256
    projector = Projector(
        geometry,
        points_per_batch=rays * points_per_ray,
        points_per_ray=points_per_ray,
        device="cuda:0",
        random_offset_detector=0.0,
        uniform_ray_spacing=True,
    )
    volume = torch.as_tensor(volume_model_zyx, device="cuda:0")
    dummy = torch.zeros(rays, device="cuda:0")
    result = []
    for angle in angles:
        projector.update(float(angle))
        projector.random_indexes = torch.arange(rays, dtype=torch.int64, device="cuda:0")
        points, _ = projector(0, dummy)
        indices = torch.floor(points * torch.tensor(volume.shape, device="cuda:0")).long()
        for axis, size in enumerate(volume.shape):
            indices[..., axis].clamp_(0, size - 1)
        samples = volume[indices[..., 0], indices[..., 1], indices[..., 2]]
        line_integrals = samples.sum(dim=1) * projector.distances
        result.append(line_integrals.reshape(height, width).cpu().numpy())
    return np.asarray(result, dtype=np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "attenuation_validation" / "outputs" / "coordinate_diagnostic",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = porous_experiment_config(num_angles=4, epochs=1)
    config["volume"]["shape_zyx"] = [32, 32, 32]
    config["geometry"]["nDetector_hw"] = [33, 35]
    angles = np.asarray([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float32)
    world = _asymmetric_volume((32, 32, 32))
    tigre_projection, _ = project_with_tigre(world, config, angles)

    # Existing Stage 2 supplied TIGRE angles unchanged.  Its learned NeCT model
    # must therefore contain Rz(pi) * world for identical detector samples.
    model_rotated = tigre_world_volume_to_nect_model(world)
    nect_expected = _nect_forward(model_rotated, config, angles)
    nect_unmapped = _nect_forward(world, config, angles)
    candidates = {
        "Rz(pi) model, same angle and detector order": _corr(tigre_projection, nect_expected),
        "same-coordinate model (incorrect for legacy Stage 2)": _corr(tigre_projection, nect_unmapped),
        "Rz(pi) model plus detector-width reversal": _corr(tigre_projection, nect_expected[:, :, ::-1]),
        "Rz(pi) model plus detector-height reversal": _corr(tigre_projection, nect_expected[:, ::-1, :]),
        "same-coordinate model plus pi angle roll": _corr(tigre_projection, np.roll(nect_unmapped, 2, axis=0)),
    }
    expected_key = "Rz(pi) model, same angle and detector order"
    angle_offset_key = "same-coordinate model plus pi angle roll"
    report = {
        "phantom_array_order": "zyx",
        "projection_array_order": "nhw",
        "angles_rad": angles.tolist(),
        "feature_description": [
            "+z,-y,+x cuboid (mu=0.010)",
            "-z,+y,near-zero-x cuboid (mu=0.017)",
            "near-zero-z/y,-x cuboid (mu=0.025)",
        ],
        "tigre_angle_zero": {"source": "+x", "detector": "-x", "increasing_u": "+y"},
        "nect_angle_zero": {"source": "-x", "detector": "+x", "increasing_column": "-y"},
        "candidate_projection_correlations": candidates,
        "equivalent_geometry_descriptions": [expected_key, angle_offset_key],
        "decision_rule": (
            "The mapping is fixed analytically from source and detector-basis signs; correlations only verify it "
            "and reject detector-only reversals."
        ),
        "conclusion": "fixed pi angular-origin offset; Rz(pi) maps TIGRE world to legacy NeCT model coordinates",
    }
    np.save(args.output_dir / "asymmetric_phantom_world_zyx.npy", world)
    np.save(args.output_dir / "tigre_projections_nhw.npy", tigre_projection)
    np.save(args.output_dir / "nect_expected_projections_nhw.npy", nect_expected)
    write_json(args.output_dir / "report.json", report)
    print(report)
    detector_only = max(
        candidates["Rz(pi) model plus detector-width reversal"],
        candidates["Rz(pi) model plus detector-height reversal"],
    )
    if candidates[expected_key] <= detector_only:
        raise RuntimeError("Detector-only reversal matched at least as well as the derived Rz(pi) mapping")
    if abs(candidates[expected_key] - candidates[angle_offset_key]) > 0.05:
        raise RuntimeError("The equivalent pi-angle and Rz(pi)-volume descriptions disagree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
