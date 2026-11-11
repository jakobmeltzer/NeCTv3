from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json
from .projections import make_angles, project_with_tigre


def _central_line_candidates(volume: np.ndarray, voxel_size: list[float]) -> dict[str, float]:
    """Independent axis-aligned voxel sums for the even-sized central plane."""
    z_candidates = sorted({volume.shape[0] // 2 - 1, volume.shape[0] // 2})
    y_candidates = sorted({volume.shape[1] // 2 - 1, volume.shape[1] // 2})
    x_candidates = sorted({volume.shape[2] // 2 - 1, volume.shape[2] // 2})
    values: dict[str, float] = {}
    for z in z_candidates:
        for y in y_candidates:
            values[f"x_line_z{z}_y{y}"] = float(volume[z, y, :].sum() * voxel_size[2])
        for x in x_candidates:
            values[f"y_line_z{z}_x{x}"] = float(volume[z, :, x].sum() * voxel_size[1])
    return values


def validate_porous_projections(
    volume: np.ndarray,
    projections: np.ndarray,
    angles: np.ndarray,
    config: dict,
    output_dir: Path,
) -> dict:
    tolerance = 1e-6
    if not np.isfinite(projections).all():
        raise ValueError("TIGRE porous projections contain NaN or infinite values")
    if float(projections.min()) < -tolerance:
        raise ValueError(f"TIGRE porous projections contain negative values: {projections.min()}")

    zero_projection, _ = project_with_tigre(
        np.zeros_like(volume), config, angles=np.asarray([angles[0]], dtype=np.float32)
    )
    zero_max = float(np.max(np.abs(zero_projection)))
    if zero_max > 1e-7:
        raise ValueError(f"Zero attenuation volume produced nonzero projection: {zero_max}")

    height, width = projections.shape[1:]
    center_hw = (height // 2, width // 2)
    candidates = _central_line_candidates(volume, config["volume"]["voxel_size_zyx_mm"])
    candidate_values = np.asarray(list(candidates.values()), dtype=np.float64)
    cardinal_checks = []
    for requested in [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]:
        wrapped_distance = np.abs(np.angle(np.exp(1j * (angles.astype(np.float64) - requested))))
        index = int(np.argmin(wrapped_distance))
        measured = float(projections[index, center_hw[0], center_hw[1]])
        differences = np.abs(candidate_values - measured)
        best_index = int(np.argmin(differences))
        best_expected = float(candidate_values[best_index])
        relative = float(differences[best_index] / max(abs(best_expected), 1e-12))
        cardinal_checks.append(
            {
                "requested_angle_rad": requested,
                "actual_angle_rad": float(angles[index]),
                "projection_index": index,
                "measured_central_integral": measured,
                "best_independent_line": list(candidates)[best_index],
                "best_independent_integral": best_expected,
                "relative_error": relative,
            }
        )

    maximum_cardinal_error = max(check["relative_error"] for check in cardinal_checks)
    validation = {
        "array_order": "nhw",
        "shape_nhw": list(projections.shape),
        "angles_count": int(len(angles)),
        "detector_center_index_hw": list(center_hw),
        "projection_min": float(projections.min()),
        "projection_max": float(projections.max()),
        "all_finite": True,
        "all_nonnegative_with_tolerance": True,
        "zero_volume_projection_max_abs": zero_max,
        "independent_axis_line_integrals": candidates,
        "cardinal_central_ray_checks": cardinal_checks,
        "maximum_cardinal_relative_error": maximum_cardinal_error,
        "cardinal_check_note": (
            "The 64-voxel volume has no single central z/y/x voxel. Candidate sums on both "
            "sides of the central plane and both in-plane axes are reported; TIGRE's closest "
            "axis-aligned convention is selected for each cardinal view."
        ),
        "projection_units": "dimensionless integral of mm^-1 over mm",
    }
    write_json(output_dir / "projections" / "validation.json", validation)
    if maximum_cardinal_error > 0.15:
        raise ValueError(
            "TIGRE central cardinal rays disagree with independent voxel line sums by up to "
            f"{100 * maximum_cardinal_error:.2f}%. Check geometry, axes and detector ordering."
        )
    return validation


def generate_porous_projections(volume: np.ndarray, config: dict, output_dir: Path) -> np.ndarray:
    projections, angles = project_with_tigre(volume, config, angles=make_angles(config))
    root = output_dir / "projections"
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "projections_physical.npy", projections)
    np.save(root / "angles_rad.npy", angles)
    validate_porous_projections(volume, projections, angles, config, output_dir)
    return projections
