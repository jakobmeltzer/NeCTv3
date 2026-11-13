from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .common import read_json, write_json
from .coordinates import (
    TIGRE_WORLD_TO_NECT_MODEL,
    nect_model_volume_to_tigre_world,
)
from .evaluation import _evaluate_raw_volume, _load_model, _plot_loss
from .export_diagnostic import _projection_scaling


def _region_metrics(reconstruction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> dict:
    if not np.any(mask):
        raise ValueError("Evaluation mask is empty")
    reconstructed = reconstruction[mask].astype(np.float64)
    reference = ground_truth[mask].astype(np.float64)
    error = reconstructed - reference
    return {
        "voxel_count": int(mask.sum()),
        "volume_fraction": float(mask.mean()),
        "ground_truth_mean_mm_inv": float(reference.mean()),
        "reconstructed_mean_mm_inv": float(reconstructed.mean()),
        "bias_mm_inv": float(error.mean()),
        "error_standard_deviation_mm_inv": float(error.std()),
        "mae_mm_inv": float(np.abs(error).mean()),
        "rmse_mm_inv": float(np.sqrt(np.mean(error**2))),
        "error_percentiles_mm_inv": {
            str(q): float(np.percentile(error, q)) for q in [1, 5, 25, 50, 75, 95, 99]
        },
    }


def _make_region_masks(rock: np.ndarray, pore: np.ndarray) -> dict[str, np.ndarray]:
    structure = ndimage.generate_binary_structure(3, 1)
    rock_boundary_1 = rock & ndimage.binary_dilation(pore, structure=structure, iterations=1)
    pore_boundary_1 = pore & ndimage.binary_dilation(rock, structure=structure, iterations=1)
    rock_boundary_2 = rock & ndimage.binary_dilation(pore, structure=structure, iterations=2)
    pore_boundary_2 = pore & ndimage.binary_dilation(rock, structure=structure, iterations=2)
    boundary_1 = rock_boundary_1 | pore_boundary_1
    boundary_2 = rock_boundary_2 | pore_boundary_2
    return {
        "boundary_within_1_voxel": boundary_1,
        "boundary_within_2_voxels": boundary_2,
        "boundary_second_voxel_shell": boundary_2 & ~boundary_1,
        "rock_boundary_within_2_voxels": rock_boundary_2,
        "pore_boundary_within_2_voxels": pore_boundary_2,
        "rock_interior_beyond_2_voxels": rock & ~rock_boundary_2,
        "pore_interior_beyond_2_voxels": pore & ~pore_boundary_2,
    }


def _select_profiles(rock_mask: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.count_nonzero(np.diff(rock_mask.astype(np.int8), axis=2), axis=2)
    flat = transitions.ravel()
    targets = [
        (rock_mask.shape[0] // 2, rock_mask.shape[1] // 2),
    ]
    for quantile in [25, 60, 100]:
        target_value = np.percentile(flat, quantile)
        index = int(np.argmin(np.abs(flat - target_value)))
        targets.append(np.unravel_index(index, transitions.shape))
    unique: list[tuple[int, int]] = []
    for z, y in targets:
        pair = (int(z), int(y))
        if pair not in unique:
            unique.append(pair)
    return unique


def _write_profiles(
    evaluation_dir: Path,
    gt: np.ndarray,
    reconstruction: np.ndarray,
    rock_mask: np.ndarray,
    voxel_size_x: float,
) -> list[dict]:
    profiles = _select_profiles(rock_mask)
    x_mm = (np.arange(gt.shape[2]) + 0.5 - gt.shape[2] / 2.0) * voxel_size_x
    rows = []
    with (evaluation_dir / "profiles_x.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile", "z_index", "y_index", "x_mm", "ground_truth_mm_inv", "reconstruction_mm_inv"])
        for profile_index, (z, y) in enumerate(profiles):
            transitions = int(np.count_nonzero(np.diff(rock_mask[z, y].astype(np.int8))))
            rows.append({"profile": profile_index, "z_index": z, "y_index": y, "phase_transitions": transitions})
            for x in range(gt.shape[2]):
                writer.writerow([profile_index, z, y, x_mm[x], gt[z, y, x], reconstruction[z, y, x]])

    fig, axes = plt.subplots(len(profiles), 1, figsize=(10, 2.8 * len(profiles)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for index, ((z, y), ax) in enumerate(zip(profiles, axes)):
        ax.step(x_mm, gt[z, y], where="mid", label="Ground truth", lw=2)
        ax.plot(x_mm, reconstruction[z, y], label="NeCT", lw=1.5)
        ax.set_title(f"Profile {index}: z={z}, y={y}, transitions={rows[index]['phase_transitions']}")
        ax.set_ylabel(r"$\mu$ (mm$^{-1}$)")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Physical x (mm)")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "figures" / "attenuation_profiles.png", dpi=220)
    plt.close(fig)
    return rows


def _plot_slices(evaluation_dir: Path, gt: np.ndarray, reconstruction: np.ndarray) -> None:
    z = gt.shape[0] // 2
    error = reconstruction - gt
    error_limit = max(0.002, float(np.percentile(np.abs(error), 99)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    entries = [
        (gt[z], "Ground truth", "gray", 0.0, 0.020),
        (reconstruction[z], "NeCT reconstruction", "gray", 0.0, 0.020),
        (error[z], "Signed error", "coolwarm", -error_limit, error_limit),
    ]
    for ax, (image, title, cmap, vmin, vmax) in zip(axes, entries):
        artist = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="x voxel", ylabel="y voxel")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="mm$^{-1}$")
    fig.tight_layout()
    fig.savefig(evaluation_dir / "figures" / "central_slice_comparison.png", dpi=220)
    plt.close(fig)


def _plot_distributions(
    evaluation_dir: Path,
    gt: np.ndarray,
    reconstruction: np.ndarray,
    rock: np.ndarray,
    pore: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    bins = np.linspace(0.0, 0.025, 120)
    axes[0].hist(reconstruction[pore], bins=bins, density=True, alpha=0.65, label="Pore voxels")
    axes[0].hist(reconstruction[rock], bins=bins, density=True, alpha=0.65, label="Rock voxels")
    axes[0].axvline(0.0, color="C0", ls="--", label="Pore truth")
    axes[0].axvline(0.020, color="C1", ls="--", label="Rock truth")
    axes[0].set(xlabel="Reconstructed attenuation (mm$^{-1}$)", ylabel="Density", title="Phase-wise attenuation distributions")
    axes[0].legend()

    error = reconstruction - gt
    error_limit = max(0.002, float(np.percentile(np.abs(error), 99.5)))
    error_bins = np.linspace(-error_limit, error_limit, 140)
    axes[1].hist(error[pore], bins=error_bins, density=True, alpha=0.65, label="Pore error")
    axes[1].hist(error[rock], bins=error_bins, density=True, alpha=0.65, label="Rock error")
    axes[1].axvline(0.0, color="black", lw=1, ls="--")
    axes[1].set(xlabel="Signed error (mm$^{-1}$)", ylabel="Density", title="Phase-wise error distributions")
    axes[1].legend()

    names = ["Boundary ≤1", "Boundary shell 1–2", "Rock interior", "Pore interior"]
    masks = [
        region_masks["boundary_within_1_voxel"],
        region_masks["boundary_second_voxel_shell"],
        region_masks["rock_interior_beyond_2_voxels"],
        region_masks["pore_interior_beyond_2_voxels"],
    ]
    values = [np.abs(reconstruction[mask] - gt[mask]) for mask in masks]
    axes[2].boxplot(values, labels=names, showfliers=False)
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set(ylabel="Absolute error (mm$^{-1}$)", title="Boundary versus interior error")
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(evaluation_dir / "figures" / "distributions_and_boundary_error.png", dpi=220)
    plt.close(fig)


def _write_ply(path: Path, mask: np.ndarray, color: tuple[int, int, int], max_points: int = 50_000) -> int:
    structure = ndimage.generate_binary_structure(3, 1)
    surface = mask & ~ndimage.binary_erosion(mask, structure=structure)
    zyx = np.argwhere(surface)
    stride = max(1, math.ceil(len(zyx) / max_points))
    zyx = zyx[::stride]
    xyz = zyx[:, ::-1].astype(np.float32) + 0.5
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in xyz:
            handle.write(f"{x:.3f} {y:.3f} {z:.3f} {color[0]} {color[1]} {color[2]}\n")
    return int(len(xyz))


def _visualize_3d(evaluation_dir: Path, rock: np.ndarray, reconstruction: np.ndarray, threshold: float) -> dict:
    root = evaluation_dir / "visualizations"
    root.mkdir(parents=True, exist_ok=True)
    reconstructed_rock = reconstruction >= threshold
    gt_points = _write_ply(root / "ground_truth_rock_surface.ply", rock, (50, 110, 190))
    rec_points = _write_ply(root / "reconstructed_rock_surface.ply", reconstructed_rock, (220, 120, 40))
    np.savez_compressed(root / "porous_volumes_mm_inv.npz", reconstruction=reconstruction.astype(np.float32), rock_mask=rock, threshold_mm_inv=np.float32(threshold))

    def downsample(mask_or_volume: np.ndarray, is_mask: bool) -> np.ndarray:
        blocks = mask_or_volume.reshape(32, 2, 32, 2, 32, 2).mean(axis=(1, 3, 5))
        return blocks >= (0.5 if is_mask else threshold)

    masks = [downsample(rock, True), downsample(reconstruction, False)]
    fig = plt.figure(figsize=(12, 6))
    for index, (mask, title, color) in enumerate(zip(masks, ["Ground-truth rock", "Reconstructed rock"], ["tab:blue", "tab:orange"]), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.voxels(mask, facecolors=color, edgecolor="none", alpha=0.35)
        ax.set_title(f"{title}\nthreshold={threshold:.3f} mm$^{{-1}}$, displayed at 2× binning")
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(root / "porous_3d_comparison.png", dpi=180)
    plt.close(fig)
    return {"threshold_mm_inv": threshold, "ground_truth_ply_points": gt_points, "reconstruction_ply_points": rec_points, "display_binning": 2}


def evaluate_porous(
    output_dir: Path,
    model_dir: Path,
    *,
    correct_coordinate_frame: bool = False,
) -> dict:
    gt = np.load(output_dir / "ground_truth" / "phantom_mu_mm_inv.npy")
    rock = np.load(output_dir / "ground_truth" / "rock_mask.npy").astype(bool)
    pore = np.load(output_dir / "ground_truth" / "pore_mask.npy").astype(bool)
    metadata = read_json(output_dir / "ground_truth" / "metadata.json")
    if not np.array_equal(rock, ~pore) or not np.allclose(gt[rock], metadata["rock_attenuation_mm_inv"]) or not np.allclose(gt[pore], metadata["pore_attenuation_mm_inv"]):
        raise ValueError("Ground-truth masks and calibrated attenuation volume are inconsistent")

    model, device, cfg, geometry = _load_model(model_dir)
    raw = _evaluate_raw_volume(model, device, tuple(gt.shape), "voxel_centers")
    projection_min, projection_max, projection_range = _projection_scaling(output_dir)
    reconstruction_model = raw * (projection_range / float(geometry.max_distance_traveled))
    reconstruction = (
        nect_model_volume_to_tigre_world(reconstruction_model)
        if correct_coordinate_frame
        else reconstruction_model
    )

    evaluation_dir = output_dir / (
        "evaluation_coordinate_corrected" if correct_coordinate_frame else "evaluation"
    )
    (evaluation_dir / "figures").mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "volumes").mkdir(parents=True, exist_ok=True)
    np.save(evaluation_dir / "volumes" / "raw_network_output.npy", raw)
    np.save(
        evaluation_dir / "volumes" / "reconstruction_model_coordinates_mm_inv.npy",
        reconstruction_model.astype(np.float32),
    )
    np.save(evaluation_dir / "volumes" / "reconstruction_voxel_centers_mm_inv.npy", reconstruction.astype(np.float32))

    regions = _make_region_masks(rock, pore)
    for name, mask in regions.items():
        np.save(evaluation_dir / "volumes" / f"mask_{name}.npy", mask)
    region_metrics = {name: _region_metrics(reconstruction, gt, mask) for name, mask in regions.items() if np.any(mask)}
    global_mask = np.ones(gt.shape, dtype=bool)
    model_correlation = float(
        np.corrcoef(reconstruction_model.astype(np.float64).ravel(), gt.astype(np.float64).ravel())[0, 1]
    )
    reported_correlation = float(
        np.corrcoef(reconstruction.astype(np.float64).ravel(), gt.astype(np.float64).ravel())[0, 1]
    )
    metrics = {
        "model_directory": str(model_dir),
        "coordinate_convention": {
            "array_order": "zyx",
            "sampling": "voxel centers: (i + 0.5) / N",
            "reported_frame": "TIGRE physical world" if correct_coordinate_frame else "NeCT model",
            "model_to_world_mapping": TIGRE_WORLD_TO_NECT_MODEL if correct_coordinate_frame else "none",
            "reason": (
                "TIGRE and NeCT angle zero differ by pi about z; both detector-u and source signs map under Rz(pi)."
                if correct_coordinate_frame
                else "legacy direct checkpoint evaluation retained for comparison"
            ),
        },
        "attenuation_scaling": {
            "projection_min": projection_min,
            "projection_max": projection_max,
            "projection_range": projection_range,
            "max_distance_traveled_mm": float(geometry.max_distance_traveled),
            "additive_projection_minimum_applied": False,
        },
        "actual_porosity": float(pore.mean()),
        "rock_fraction": float(rock.mean()),
        "reconstruction_configuration": {
            "model": str(cfg.model),
            "encoder": str(cfg.encoder),
            "network": str(cfg.net),
            "loss": str(cfg.loss),
            "epochs": int(cfg.epochs),
            "points_per_ray_start": int(cfg.points_per_ray.start),
            "points_per_ray_end": int(cfg.points_per_ray.end),
            "batch_per_projection": str(cfg.batch_per_proj),
            "add_poisson": bool(cfg.add_poisson),
            "use_extended_forward_model": bool(cfg.use_extended_forward_model),
        },
        "boundary_definition": {
            "connectivity": "6-connected face neighbours",
            "within_1_voxel": "phase voxel reached by one binary dilation of the opposite phase",
            "within_2_voxels": "phase voxel reached by two binary dilations of the opposite phase",
            "interior": "phase voxels outside the two-voxel boundary mask",
        },
        "global": _region_metrics(reconstruction, gt, global_mask),
        "voxelwise_pearson_correlation": reported_correlation,
        "orientation_check": {
            "legacy_model_frame_correlation": model_correlation,
            "reported_world_frame_correlation": reported_correlation,
            "mapping_was_fixed_from_ray_geometry_before_metric_comparison": True,
        },
        "rock": _region_metrics(reconstruction, gt, rock),
        "pore": {
            **_region_metrics(reconstruction, gt, pore),
            "false_positive_mean_mm_inv": float(reconstruction[pore].mean()),
            "fraction_above_0.001_mm_inv": float(np.mean(reconstruction[pore] > 0.001)),
            "fraction_above_0.005_mm_inv": float(np.mean(reconstruction[pore] > 0.005)),
        },
        "regions": region_metrics,
    }
    profiles = _write_profiles(evaluation_dir, gt, reconstruction, rock, float(cfg.geometry.dVoxel[2]))
    _plot_slices(evaluation_dir, gt, reconstruction)
    _plot_distributions(evaluation_dir, gt, reconstruction, rock, pore, regions)
    convergence = _plot_loss(output_dir, model_dir, evaluation_dir / "figures")
    visualization = _visualize_3d(evaluation_dir, rock, reconstruction, threshold=0.010)
    metrics["profiles"] = profiles
    metrics["convergence"] = convergence
    metrics["visualization"] = visualization
    write_json(evaluation_dir / "metrics.json", metrics)
    return metrics
