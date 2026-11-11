from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, write_json


def _load_model(model_dir: Path):
    import torch
    from nect.config import get_cfg
    from nect.sampling.geometry import Geometry

    cfg = get_cfg(model_dir / "config.yaml")
    model = cfg.get_model()
    checkpoint_path = model_dir / "checkpoints" / "last.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    geometry = Geometry.from_cfg(
        cfg.geometry,
        reconstruction_mode=cfg.reconstruction_mode,
        sample_outside=cfg.sample_outside,
    )
    return model, device, cfg, geometry


def _normalized_axes(
    shape: tuple[int, int, int], convention: str, device
):
    import torch

    if convention == "voxel_centers":
        return [
            torch.linspace(0.5 / n, 1.0 - 0.5 / n, steps=n, device=device)
            for n in shape
        ]
    if convention == "endpoints":
        return [torch.linspace(0.0, 1.0, steps=n, device=device) for n in shape]
    raise ValueError(f"Unknown coordinate convention: {convention!r}")


def _evaluate_raw_volume(
    model,
    device,
    shape: tuple[int, int, int],
    convention: str = "voxel_centers",
) -> np.ndarray:
    import torch

    axes = _normalized_axes(shape, convention, device)
    output = np.empty(shape, dtype=np.float32)
    with torch.no_grad():
        for z_index, z_value in enumerate(axes[0]):
            zz, yy, xx = torch.meshgrid(z_value[None], axes[1], axes[2], indexing="ij")
            points = torch.stack((zz.ravel(), yy.ravel(), xx.ravel()), dim=1)
            output[z_index] = model(points).reshape(shape[1], shape[2]).float().cpu().numpy()
    return output


def _metrics(reconstruction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> dict:
    error = reconstruction.astype(np.float64) - ground_truth.astype(np.float64)
    gt_inside = float(ground_truth[mask].mean())
    rec_inside = float(reconstruction[mask].mean())
    abs_inside = abs(rec_inside - gt_inside)
    return {
        "cube_mean_mm_inv": rec_inside,
        "cube_ground_truth_mean_mm_inv": gt_inside,
        "background_mean_mm_inv": float(reconstruction[~mask].mean()),
        "cube_absolute_error_mm_inv": abs_inside,
        "cube_relative_error": abs_inside / gt_inside,
        "whole_volume_mae_mm_inv": float(np.mean(np.abs(error))),
        "whole_volume_rmse_mm_inv": float(np.sqrt(np.mean(error**2))),
        "whole_volume_bias_mm_inv": float(np.mean(error)),
    }


def _read_losses(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = model_dir / "epoch_losses.txt"
    epochs: list[int] = []
    losses: list[float] = []
    seconds: list[float] = []
    if path.exists():
        pattern = re.compile(r"epoch=(\d+), avg_loss=([^,]+), time=([^s]+)s")
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                seconds.append(float(match.group(3)))
    return np.asarray(epochs), np.asarray(losses), np.asarray(seconds)


def _plot_results(
    output_dir: Path,
    gt: np.ndarray,
    converted: np.ndarray,
    exported: np.ndarray,
    x_mm: np.ndarray,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    zc, yc = gt.shape[0] // 2, gt.shape[1] // 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_mm, gt[zc, yc], label="Ground truth", linewidth=2.5)
    ax.plot(x_mm, converted[zc, yc], label="Direct INR, legacy conversion", linewidth=1.8)
    ax.plot(x_mm, exported[zc, yc], label="Existing exporter", linestyle="--", linewidth=1.4)
    ax.set(xlabel="x (mm)", ylabel="Attenuation (mm$^{-1}$)", title="Central x-axis attenuation profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "attenuation_profile_x.png", dpi=200)
    plt.close(fig)

    error = converted - gt
    fixed_max = max(0.010, float(np.nanmax([gt.max(), converted.max(), exported.max()])))
    error_limit = max(0.001, float(np.nanpercentile(np.abs(error), 99)))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    images = [gt[zc], converted[zc], exported[zc], error[zc]]
    titles = ["Ground truth", "Direct INR converted", "Existing exporter", "Signed error"]
    for index, (ax, image, title) in enumerate(zip(axes, images, titles)):
        if index < 3:
            artist = ax.imshow(image, cmap="gray", vmin=0.0, vmax=fixed_max, origin="lower")
        else:
            artist = ax.imshow(image, cmap="coolwarm", vmin=-error_limit, vmax=error_limit, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("x voxel")
        ax.set_ylabel("y voxel")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="mm$^{-1}$")
    fig.tight_layout()
    fig.savefig(figures / "central_slices_and_error.png", dpi=200)
    plt.close(fig)


def _plot_loss(output_dir: Path, model_dir: Path, figures_dir: Path | None = None) -> dict:
    epochs, losses, seconds = _read_losses(model_dir)
    info = {"logged_epochs": int(len(epochs))}
    if len(epochs) == 0:
        return info
    info.update(
        {
            "initial_epoch_loss": float(losses[0]),
            "final_epoch_loss": float(losses[-1]),
            "elapsed_seconds_at_final_log": float(seconds[-1]),
        }
    )
    figures_dir = output_dir / "figures" if figures_dir is None else figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(epochs, losses, marker="o", markersize=3)
    ax.set(xlabel="Epoch", ylabel="Mean training loss", title="Legacy NeCT convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss.png", dpi=200)
    plt.close(fig)
    return info


def _plot_3d(output_dir: Path, gt: np.ndarray, converted: np.ndarray) -> None:
    visualization = output_dir / "visualizations"
    visualization.mkdir(parents=True, exist_ok=True)
    threshold = 0.005
    gt_mask = gt >= threshold
    rec_mask = converted >= threshold
    np.savez_compressed(
        visualization / "volumes_mm_inv.npz",
        ground_truth=gt.astype(np.float32),
        reconstruction=converted.astype(np.float32),
        threshold_mm_inv=np.float32(threshold),
    )

    fig = plt.figure(figsize=(12, 6))
    for index, (mask, title, color) in enumerate(
        [(gt_mask, "Ground truth", "tab:blue"), (rec_mask, "NeCT reconstruction", "tab:orange")],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.voxels(mask, facecolors=color, edgecolor="none", alpha=0.45)
        ax.set_title(f"{title}\nfixed isosurface threshold = {threshold:.3f} mm$^{{-1}}$")
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(visualization / "cube_isosurfaces.png", dpi=180)
    plt.close(fig)


def evaluate(output_dir: Path, model_dir: Path) -> dict:
    import tifffile
    from nect.static_export import export_volume

    gt = np.load(output_dir / "ground_truth" / "phantom_mu_mm_inv.npy")
    mask = np.load(output_dir / "ground_truth" / "cube_mask.npy").astype(bool)
    projections = np.load(output_dir / "projections" / "projections_physical.npy", mmap_mode="r")
    projection_min = float(projections.min())
    projection_max = float(projections.max())
    projection_range = projection_max - projection_min
    if projection_range <= 0:
        raise ValueError("Projection range must be positive")

    model, device, cfg, geometry = _load_model(model_dir)
    raw = _evaluate_raw_volume(model, device, tuple(gt.shape))
    # Correct conversion for the actual single-file training target:
    # p_train=(p-p_min)/range and NeCT uses ds/max_distance. Multiplicative
    # scaling is locally invertible; the additive p_min is not a voxel value.
    converted = raw / geometry.max_distance_traveled * projection_range
    legacy_formula = converted + projection_min

    arrays_dir = output_dir / "evaluation" / "volumes"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "raw_network_output.npy", raw)
    np.save(arrays_dir / "direct_inr_legacy_converted_mm_inv.npy", converted.astype(np.float32))
    np.save(arrays_dir / "legacy_formula_with_projection_min.npy", legacy_formula.astype(np.float32))

    exporter_path = export_volume(model_dir)
    exporter_file = Path(exporter_path) / "Static_Volume.tiff"
    exported = tifffile.imread(exporter_file).astype(np.float32)
    np.save(arrays_dir / "existing_exporter_volume_mm_inv.npy", exported)

    center = tuple(n // 2 for n in gt.shape)
    x_mm = (np.arange(gt.shape[2]) + 0.5 - gt.shape[2] / 2.0) * cfg.geometry.dVoxel[2]
    profile_path = output_dir / "evaluation" / "profile_x.csv"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["x_mm", "ground_truth_mm_inv", "raw_network_output", "direct_converted_mm_inv", "existing_exporter_mm_inv"]
        )
        for values in zip(x_mm, gt[center[0], center[1]], raw[center[0], center[1]], converted[center[0], center[1]], exported[center[0], center[1]]):
            writer.writerow([float(value) for value in values])

    _plot_results(output_dir, gt, converted, exported, x_mm)
    convergence = _plot_loss(output_dir, model_dir)
    _plot_3d(output_dir, gt, converted)

    normalized = (np.asarray(projections) - projection_min) / projection_range
    metrics = {
        "model_directory": str(model_dir),
        "max_distance_traveled_mm": float(geometry.max_distance_traveled),
        "projection_loading": {
            "input_kind": "single .npy file",
            "before_min": projection_min,
            "before_max": projection_max,
            "after_legacy_loader_min": float(normalized.min()),
            "after_legacy_loader_max": float(normalized.max()),
            "range": projection_range,
        },
        "direct_inr_legacy_converted": _metrics(converted, gt, mask),
        "existing_exporter": _metrics(exported, gt, mask),
        "raw_network": {
            "cube_mean": float(raw[mask].mean()),
            "background_mean": float(raw[~mask].mean()),
        },
        "convergence": convergence,
        "coordinate_conventions": {
            "array_order": "zyx",
            "network_coordinates": "zyx normalized to [0,1], evaluated at voxel centers",
            "profile_fixed_indices_zy": [center[0], center[1]],
            "exporter_note": "Standalone legacy exporter uses endpoint coordinates; direct INR result uses voxel centers.",
        },
    }
    write_json(output_dir / "evaluation" / "metrics.json", metrics)
    return metrics

