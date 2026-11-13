from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json
from .coordinates import nect_angles_for_tigre_world
from .dynamic_phantom import state_index_for_time
from .projections import project_with_tigre


def acquisition_arrays(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acq = config["acquisition"]
    count = int(acq["num_projections"])
    timestamps = np.linspace(0.0, 1.0, count, dtype=np.float32)
    unwrapped = np.linspace(
        0.0,
        float(acq["num_revolutions"]) * 2.0 * np.pi,
        count,
        endpoint=False,
        dtype=np.float64,
    )
    tigre_angles = np.mod(unwrapped, 2.0 * np.pi).astype(np.float32)
    nect_angles = nect_angles_for_tigre_world(tigre_angles).astype(np.float32)
    return timestamps, tigre_angles, nect_angles


def generate_dynamic_projections(config: dict, output_dir: Path) -> dict:
    timestamps, tigre_angles, nect_angles = acquisition_arrays(config)
    gt_root = output_dir / "ground_truth"
    state_times = np.load(gt_root / "state_times_normalized.npy")
    fluid_states = np.load(gt_root / "fluid_masks.npy", mmap_mode="r")
    rock = np.load(gt_root / "rock_mask.npy").astype(bool)
    projections = []
    state_indices = []
    for index, (time_value, angle) in enumerate(zip(timestamps, tigre_angles)):
        state_index = state_index_for_time(state_times, float(time_value))
        fluid = fluid_states[state_index]
        volume = np.full(rock.shape, config["volume"]["air_attenuation_mm_inv"], dtype=np.float32)
        volume[fluid] = np.float32(config["volume"]["fluid_attenuation_mm_inv"])
        volume[rock] = np.float32(config["volume"]["rock_attenuation_mm_inv"])
        projection, _ = project_with_tigre(volume, config, np.asarray([angle], dtype=np.float32))
        projections.append(projection[0])
        state_indices.append(state_index)
        if (index + 1) % 50 == 0 or index + 1 == len(timestamps):
            print(f"[projections] {index + 1}/{len(timestamps)}", flush=True)
    projections_array = np.asarray(projections, dtype=np.float32)
    state_indices_array = np.asarray(state_indices, dtype=np.int32)

    expected = (len(timestamps), *config["geometry"]["nDetector_hw"])
    if projections_array.shape != expected or not np.isfinite(projections_array).all():
        raise RuntimeError(f"Invalid dynamic projection array {projections_array.shape}; expected {expected}")
    if float(projections_array.min()) < -1e-6:
        raise RuntimeError("Dynamic TIGRE projections contain negative line integrals")
    if not (len(timestamps) == len(tigre_angles) == len(nect_angles) == len(state_indices_array)):
        raise RuntimeError("Projection/angle/time arrays are misaligned")
    angular_error = np.angle(np.exp(1j * (nect_angles - tigre_angles - np.pi)))
    if float(np.max(np.abs(angular_error))) > 2e-6:
        raise RuntimeError("TIGRE-to-NeCT pi angle mapping failed")
    # Independent line-integral check for the first (angle-zero, empty-pore)
    # state.  The even voxel grid has two candidates on each central plane.
    first_volume = np.full(rock.shape, config["volume"]["air_attenuation_mm_inv"], dtype=np.float32)
    first_volume[fluid_states[state_indices_array[0]]] = np.float32(config["volume"]["fluid_attenuation_mm_inv"])
    first_volume[rock] = np.float32(config["volume"]["rock_attenuation_mm_inv"])
    z_candidates = [rock.shape[0] // 2 - 1, rock.shape[0] // 2]
    y_candidates = [rock.shape[1] // 2 - 1, rock.shape[1] // 2]
    central_candidates = [float(first_volume[z, y, :].sum() * config["volume"]["voxel_size_zyx_mm"][2]) for z in z_candidates for y in y_candidates]
    measured_center = float(projections_array[0, projections_array.shape[1] // 2, projections_array.shape[2] // 2])
    central_relative_error = min(abs(measured_center - value) / max(abs(value), 1e-12) for value in central_candidates)
    if central_relative_error > 0.15:
        raise RuntimeError("First TIGRE central ray disagrees with independent angle-zero voxel line sums")

    root = output_dir / "projections"
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "projections_physical.npy", projections_array)
    np.save(root / "timestamps_normalized.npy", timestamps)
    np.save(root / "timestamps_seconds.npy", timestamps * float(config["acquisition"]["duration_seconds"]))
    np.save(root / "angles_tigre_rad.npy", tigre_angles)
    np.save(root / "angles_nect_rad.npy", nect_angles)
    np.save(root / "ground_truth_state_indices.npy", state_indices_array)
    validation = {
        "shape_nhw": list(projections_array.shape),
        "array_order": "nhw",
        "projection_units": "dimensionless integral of mm^-1 over mm",
        "projection_min": float(projections_array.min()),
        "projection_max": float(projections_array.max()),
        "all_finite": True,
        "num_revolutions": float(config["acquisition"]["num_revolutions"]),
        "duration_seconds": float(config["acquisition"]["duration_seconds"]),
        "timestamp_min_max": [float(timestamps.min()), float(timestamps.max())],
        "unique_ground_truth_states_observed": int(np.unique(state_indices_array).size),
        "angle_mapping": "theta_nect=(theta_tigre+pi) mod 2pi",
        "maximum_angle_mapping_error_rad": float(np.max(np.abs(angular_error))),
        "first_projection_central_integral": measured_center,
        "independent_central_x_line_candidates": central_candidates,
        "central_line_minimum_relative_error": float(central_relative_error),
        "one_dynamic_state_per_acquisition_event": True,
        "instantaneous_projection_model": True,
    }
    write_json(root / "validation.json", validation)
    write_json(root / "completed.json", {"validation": str(root / "validation.json")})
    return validation
