from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


SEED = 42


def set_cpu_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_config(num_angles: int = 180, epochs: int = 20) -> dict[str, Any]:
    """Return the single source of truth for Stage-1 geometry and materials."""
    return {
        "experiment": "stage1_geometric_legacy_nect",
        "seed": SEED,
        "legacy_reconstruction": {"epochs": int(epochs)},
        "units": {
            "length": "mm",
            "attenuation": "mm^-1",
            "projection": "dimensionless line integral",
        },
        "volume": {
            "shape_zyx": [64, 64, 64],
            "voxel_size_zyx_mm": [1.0, 1.0, 1.0],
            "background_attenuation_mm_inv": 0.0,
            "cube_side_voxels": 20,
            "cube_side_mm": 20.0,
            "cube_attenuation_mm_inv": 0.010,
        },
        "geometry": {
            "mode": "cone",
            "DSO_mm": 1000.0,
            "DSD_mm": 1500.0,
            "nDetector_hw": [65, 65],
            "dDetector_hw_mm": [1.5, 1.5],
            "offOrigin_zyx_mm": [0.0, 0.0, 0.0],
            "offDetector_hw_mm": [0.0, 0.0],
            "COR_mm": 0.0,
            "num_angles": int(num_angles),
            "angle_range": "[0, 2*pi)",
            "angle_units": "radians",
        },
    }

