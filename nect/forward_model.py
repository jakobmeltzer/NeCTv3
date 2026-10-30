from __future__ import annotations

import torch


def log_projection(
    attenuation: torch.Tensor,
    integration_step_lengths: torch.Tensor,
    max_distance_traveled: float,
    *,
    beta: torch.Tensor | None = None,
    ray_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate attenuation and optionally apply the extended Beer--Lambert term.

    Both lengths are normalized by ``max_distance_traveled``, preserving NeCT's
    original linear integration convention. Consequently, beta is learned with
    respect to squared *normalized* total ray length, not physical length squared.
    """
    normalized_step = integration_step_lengths / max_distance_traveled
    projection = torch.sum(attenuation, dim=1) * normalized_step

    if beta is not None:
        if ray_lengths is None:
            raise ValueError("ray_lengths is required when beta is provided")
        normalized_ray_length = ray_lengths / max_distance_traveled
        projection = projection - beta * normalized_ray_length.square()

    return projection
