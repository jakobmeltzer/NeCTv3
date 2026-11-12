from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from .common import write_json


def generate_porous_phantom(config: dict, output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a PoreSpy geometry, then calibrate it independently in mm^-1."""
    try:
        import porespy as ps
    except ImportError as exc:
        raise RuntimeError("PoreSpy is required for Stage 2: python -m pip install porespy") from exc

    spec = config["volume"]
    shape = tuple(int(value) for value in spec["shape_zyx"])
    seed = int(config["seed"])
    generator = ps.generators.blobs
    parameters = inspect.signature(generator).parameters
    blobiness = spec["blobiness"]
    # Older NeCT/PoreSpy code commonly used ``[1]``. Current PoreSpy forwards
    # sequences to scipy.ndimage.gaussian_filter, where a sequence must have one
    # value per spatial axis. A length-one value is semantically isotropic, so
    # pass it as a scalar for compatibility across PoreSpy versions.
    if isinstance(blobiness, (list, tuple, np.ndarray)) and len(blobiness) == 1:
        blobiness = blobiness[0]
    kwargs = {
        "shape": shape,
        "porosity": float(spec["target_porosity"]),
        "blobiness": blobiness,
    }
    if "divs" in parameters:
        kwargs["divs"] = int(spec["divs"])
    seeded_through_api = "seed" in parameters
    if seeded_through_api:
        kwargs["seed"] = seed
    else:
        # This is the mechanism used by NeCT's existing PorousMedium wrapper.
        np.random.seed(seed)
    pore_mask = np.asarray(generator(**kwargs), dtype=bool)
    if pore_mask.shape != shape:
        raise RuntimeError(f"PoreSpy returned shape {pore_mask.shape}, expected {shape}")
    rock_mask = ~pore_mask

    volume = np.full(shape, spec["pore_attenuation_mm_inv"], dtype=np.float32)
    volume[rock_mask] = np.float32(spec["rock_attenuation_mm_inv"])
    actual_porosity = float(pore_mask.mean())

    root = output_dir / "ground_truth"
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "phantom_mu_mm_inv.npy", volume)
    np.save(root / "pore_mask.npy", pore_mask)
    np.save(root / "rock_mask.npy", rock_mask)
    write_json(
        root / "metadata.json",
        {
            "array_order": "zyx",
            "shape_zyx": list(shape),
            "voxel_size_zyx_mm": spec["voxel_size_zyx_mm"],
            "generator": "porespy.generators.blobs",
            "generator_parameters": kwargs,
            "configured_blobiness": spec["blobiness"],
            "effective_blobiness": blobiness,
            "seed": seed,
            "seed_passed_through_porespy_api": seeded_through_api,
            "target_porosity": spec["target_porosity"],
            "actual_porosity": actual_porosity,
            "pore_voxel_count": int(pore_mask.sum()),
            "rock_voxel_count": int(rock_mask.sum()),
            "pore_attenuation_mm_inv": spec["pore_attenuation_mm_inv"],
            "rock_attenuation_mm_inv": spec["rock_attenuation_mm_inv"],
            "scaled_or_normalized": False,
        },
    )
    return volume, rock_mask, pore_mask


def load_porous_phantom(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = output_dir / "ground_truth"
    return (
        np.load(root / "phantom_mu_mm_inv.npy"),
        np.load(root / "rock_mask.npy").astype(bool),
        np.load(root / "pore_mask.npy").astype(bool),
    )
