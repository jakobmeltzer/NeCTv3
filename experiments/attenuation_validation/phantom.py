from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json


def generate_phantom(config: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Generate the calibrated cube explicitly in [z, y, x] array order."""
    spec = config["volume"]
    shape = tuple(spec["shape_zyx"])
    side = int(spec["cube_side_voxels"])
    if side <= 0 or any(side > n for n in shape):
        raise ValueError(f"Invalid cube side {side} for volume shape {shape}")

    starts = tuple((n - side) // 2 for n in shape)
    stops = tuple(start + side for start in starts)
    cube_mask = np.zeros(shape, dtype=bool)
    cube_mask[
        starts[0] : stops[0],
        starts[1] : stops[1],
        starts[2] : stops[2],
    ] = True

    volume = np.full(shape, spec["background_attenuation_mm_inv"], dtype=np.float32)
    volume[cube_mask] = np.float32(spec["cube_attenuation_mm_inv"])

    phantom_dir = output_dir / "ground_truth"
    phantom_dir.mkdir(parents=True, exist_ok=True)
    np.save(phantom_dir / "phantom_mu_mm_inv.npy", volume)
    np.save(phantom_dir / "cube_mask.npy", cube_mask)
    write_json(
        phantom_dir / "metadata.json",
        {
            "array_order": "zyx",
            "shape_zyx": list(shape),
            "voxel_size_zyx_mm": spec["voxel_size_zyx_mm"],
            "cube_bounds_zyx_half_open": [list(starts), list(stops)],
            "cube_voxel_count": int(cube_mask.sum()),
            "background_attenuation_mm_inv": spec["background_attenuation_mm_inv"],
            "cube_attenuation_mm_inv": spec["cube_attenuation_mm_inv"],
        },
    )
    return volume, cube_mask


def load_phantom(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    root = output_dir / "ground_truth"
    return np.load(root / "phantom_mu_mm_inv.npy"), np.load(root / "cube_mask.npy")

