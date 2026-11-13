from __future__ import annotations

from typing import Any

from .common import SEED, experiment_config


def dynamic_experiment_config(
    num_projections: int = 720,
    num_revolutions: float = 4.0,
    epochs: int = 30,
    stored_states: int = 41,
) -> dict[str, Any]:
    """Single source of truth for the Stage-3 pilot experiment."""
    cfg = experiment_config(num_angles=num_projections, epochs=epochs)
    cfg["experiment"] = "stage3_dynamic_porespy_mixedcubes"
    cfg["seed"] = SEED
    cfg["volume"] = {
        "shape_zyx": [64, 64, 64],
        "voxel_size_zyx_mm": [1.0, 1.0, 1.0],
        "target_porosity": 0.35,
        "blobiness": [1],
        "divs": 1,
        "rock_attenuation_mm_inv": 0.020,
        "air_attenuation_mm_inv": 0.0,
        "fluid_attenuation_mm_inv": 0.010,
    }
    cfg["invasion"] = {
        "algorithm": "porespy.filters.ibip",
        "inlet": "z_min_xy_plane",
        "maxiter": 15000,
        "stored_states": int(stored_states),
        "state_transition": "right-continuous discrete occupancy; no attenuation interpolation",
    }
    cfg["acquisition"] = {
        "num_projections": int(num_projections),
        "num_revolutions": float(num_revolutions),
        "duration_seconds": 120.0,
        "timestamps_normalized": "linspace(0,1,N), acquisition order",
        "instantaneous_projections": True,
    }
    cfg["geometry"]["num_angles"] = int(num_projections)
    cfg["geometry"]["angle_range"] = f"[0, {num_revolutions}*2*pi)"
    cfg["dynamic_reconstruction"] = {
        "architecture": "mixedcubes",
        "epochs": int(epochs),
        "loss": "L2",
        "batch_per_proj": 1,
        "points_per_batch": 1_000_000,
        "points_per_ray": 128,
        "checkpoint_interval_seconds": 1800,
        "add_poisson": False,
        "use_extended_forward_model": False,
        "training_angle_mapping": "theta_nect=(theta_tigre+pi) mod 2pi",
    }
    cfg["evaluation"] = {
        "times_normalized": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "fluid_threshold_mm_inv": 0.005,
        "rock_threshold_mm_inv": 0.015,
        "outer_boundary_distances_voxels": [1, 2, 4, 8],
        "coordinate_frame": "TIGRE physical world; no output rotation",
    }
    return cfg
