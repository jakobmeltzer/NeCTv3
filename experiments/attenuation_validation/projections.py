from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json


def make_angles(config: dict) -> np.ndarray:
    return np.linspace(
        0.0,
        2.0 * np.pi,
        int(config["geometry"]["num_angles"]),
        endpoint=False,
        dtype=np.float32,
    )


def _make_tigre_geometry(config: dict):
    try:
        import tigre
    except ImportError as exc:
        raise RuntimeError(
            "TIGRE is required for the projection stage. Run this stage in the IDUN "
            "environment where PyTigre is installed."
        ) from exc

    volume = config["volume"]
    geom = config["geometry"]
    geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = np.asarray(volume["shape_zyx"], dtype=np.int32)
    geo.dVoxel = np.asarray(volume["voxel_size_zyx_mm"], dtype=np.float32)
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.nDetector = np.asarray(geom["nDetector_hw"], dtype=np.int32)
    geo.dDetector = np.asarray(geom["dDetector_hw_mm"], dtype=np.float32)
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.DSO = float(geom["DSO_mm"])
    geo.DSD = float(geom["DSD_mm"])
    geo.offOrigin = np.asarray(geom["offOrigin_zyx_mm"], dtype=np.float32)
    geo.offDetector = np.asarray(geom["offDetector_hw_mm"], dtype=np.float32)
    geo.rotDetector = np.zeros(3, dtype=np.float32)
    geo.COR = float(geom["COR_mm"])
    geo.accuracy = 0.5
    return geo


def generate_projections(volume: np.ndarray, config: dict, output_dir: Path) -> np.ndarray:
    projections, angles = project_with_tigre(volume, config)

    projection_dir = output_dir / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    np.save(projection_dir / "projections_physical.npy", projections)
    np.save(projection_dir / "angles_rad.npy", angles)
    validate_projections(projections, config, output_dir)
    return projections


def project_with_tigre(
    volume: np.ndarray, config: dict, angles: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Run the shared independent TIGRE forward projector in N,H,W order."""
    try:
        import tigre
    except ImportError as exc:
        raise RuntimeError("PyTigre is not installed; projection generation must run on IDUN.") from exc

    geo = _make_tigre_geometry(config)
    if angles is None:
        angles = make_angles(config)
    else:
        angles = np.asarray(angles, dtype=np.float32)
    projections = tigre.Ax(volume.astype(np.float32, copy=False), geo, angles)
    projections = np.asarray(projections, dtype=np.float32)

    expected_shape = (len(angles), *config["geometry"]["nDetector_hw"])
    if projections.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected TIGRE projection shape {projections.shape}; expected N,H,W={expected_shape}. "
            "Check TIGRE detector ordering before reconstruction."
        )

    return projections, angles


def validate_projections(projections: np.ndarray, config: dict, output_dir: Path) -> dict:
    if not np.isfinite(projections).all():
        raise ValueError("TIGRE projections contain NaN or infinite values")
    tolerance = 1e-6
    if float(projections.min()) < -tolerance:
        raise ValueError(f"TIGRE projections contain negative values: min={projections.min()}")

    n, height, width = projections.shape
    center_hw = (height // 2, width // 2)
    central_by_angle = projections[:, center_hw[0], center_hw[1]]
    expected = (
        config["volume"]["cube_attenuation_mm_inv"]
        * config["volume"]["cube_side_mm"]
    )
    central_at_zero = float(central_by_angle[0])
    central_rel_error = abs(central_at_zero - expected) / expected

    # Corners are far outside the projected 20 mm cube for this geometry.
    corner_values = np.stack(
        [
            projections[:, 0, 0],
            projections[:, 0, -1],
            projections[:, -1, 0],
            projections[:, -1, -1],
        ]
    )
    max_miss_value = float(np.max(np.abs(corner_values)))

    validation = {
        "array_order": "nhw",
        "shape_nhw": list(projections.shape),
        "angles_count": int(n),
        "detector_center_index_hw": list(center_hw),
        "expected_orthogonal_central_integral": expected,
        "central_integral_angle_0": central_at_zero,
        "central_integral_angle_0_relative_error": central_rel_error,
        "central_integral_min_over_angles": float(central_by_angle.min()),
        "central_integral_max_over_angles": float(central_by_angle.max()),
        "projection_min": float(projections.min()),
        "projection_max": float(projections.max()),
        "corner_ray_max_abs": max_miss_value,
        "all_finite": True,
        "all_nonnegative_with_tolerance": True,
    }
    write_json(output_dir / "projections" / "validation.json", validation)

    # TIGRE's interpolation is allowed a small discretization error, but a major
    # geometry/order error should stop the expensive reconstruction immediately.
    if central_rel_error > 0.05:
        raise ValueError(
            f"Central ray integral {central_at_zero:.6g} differs from expected {expected:.6g} "
            f"by {100 * central_rel_error:.2f}%. Check geometry and axis ordering."
        )
    if max_miss_value > 1e-4:
        raise ValueError(
            f"Detector corner rays should miss the cube, but max |p|={max_miss_value:.6g}."
        )
    return validation

