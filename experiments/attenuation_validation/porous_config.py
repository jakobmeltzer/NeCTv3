from __future__ import annotations

from typing import Any

from .common import SEED, experiment_config


def porous_experiment_config(num_angles: int = 180, epochs: int = 100) -> dict[str, Any]:
    """Stage-2 configuration using Stage-1 acquisition conventions."""
    config = experiment_config(num_angles=num_angles, epochs=epochs)
    config["experiment"] = "stage2_static_porespy_legacy_nect"
    config["seed"] = SEED
    config["volume"] = {
        "shape_zyx": [64, 64, 64],
        "voxel_size_zyx_mm": [1.0, 1.0, 1.0],
        "target_porosity": 0.35,
        "rock_attenuation_mm_inv": 0.020,
        "pore_attenuation_mm_inv": 0.0,
        "porespy_generator": "porespy.generators.blobs",
        "blobiness": [1],
        "divs": 1,
    }
    config["legacy_reconstruction"].update(
        {
            "epochs": int(epochs),
            "loss": "L2",
            "batch_per_proj": 1,
            "points_per_batch": 1_000_000,
            "points_per_ray": 128,
            "checkpoint_interval_seconds": 1800,
            "add_poisson": False,
            "use_extended_forward_model": False,
            "model": "hash_grid",
        }
    )
    config["evaluation"] = {
        "boundary_distances_voxels": [1, 2],
        "primary_coordinate_convention": "voxel_centers",
        "visualization_threshold_mm_inv": 0.010,
    }
    return config
