"""Coordinate mappings shared by the TIGRE/NeCT validation experiments.

Arrays use ``[z, y, x]`` throughout.  TIGRE's zero-angle cone geometry has
its source at +x and detector at -x, with detector u increasing toward +y.
NeCT's CUDA sampler uses the opposite source/detector pair and detector u
basis.  The two frames are therefore related by Rz(pi), not by an array-axis
swap or a detector-only flip.
"""

from __future__ import annotations

import numpy as np


TIGRE_WORLD_TO_NECT_MODEL = "rotate_z_180"


def tigre_world_to_nect_model_points(points_zyx: np.ndarray) -> np.ndarray:
    """Map normalized TIGRE-world points to NeCT model coordinates.

    The normalized volume centre is (0.5, 0.5, 0.5), so Rz(pi) maps
    ``(z, y, x) -> (z, 1-y, 1-x)``.
    """
    points = np.asarray(points_zyx)
    if points.shape[-1] != 3:
        raise ValueError("Expected points with final dimension [z,y,x]")
    mapped = points.copy()
    mapped[..., 1:] = 1.0 - mapped[..., 1:]
    return mapped


def nect_model_volume_to_tigre_world(volume_zyx: np.ndarray) -> np.ndarray:
    """Express a NeCT model-grid volume in the TIGRE physical world frame."""
    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError("Expected a 3-D [z,y,x] volume")
    # Rz(pi) is self-inverse.  Copy avoids negative-stride arrays downstream.
    return np.flip(volume, axis=(1, 2)).copy()


def tigre_world_volume_to_nect_model(volume_zyx: np.ndarray) -> np.ndarray:
    """Express a TIGRE-world grid in NeCT model coordinates (Rz(pi) is self-inverse)."""
    return nect_model_volume_to_tigre_world(volume_zyx)


def nect_angles_for_tigre_world(angles_rad: np.ndarray) -> np.ndarray:
    """Angles that make NeCT model coordinates coincide with TIGRE world.

    This is for new training runs.  Existing Stage-2 checkpoints used the
    unshifted angles and must instead be converted at evaluation/export time.
    """
    return np.mod(np.asarray(angles_rad) + np.pi, 2.0 * np.pi)
