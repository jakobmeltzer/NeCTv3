from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, write_json
from .evaluation import _evaluate_raw_volume, _load_model, _metrics


def _numpy_normalized_axes(shape: tuple[int, int, int], convention: str) -> list[np.ndarray]:
    if convention == "voxel_centers":
        return [(np.arange(n, dtype=np.float64) + 0.5) / n for n in shape]
    if convention == "endpoints":
        return [np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape]
    raise ValueError(f"Unknown coordinate convention: {convention!r}")


def _sample_cube_ground_truth(
    shape: tuple[int, int, int],
    convention: str,
    metadata: dict,
    cfg,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict]:
    """Evaluate the piecewise-constant cube at the requested physical points.

    The original discrete cube occupies complete voxel cells. Its physical
    boundary is derived from the saved half-open voxel bounds rather than from
    the already sampled ground-truth array.
    """
    normalized = _numpy_normalized_axes(shape, convention)
    voxel_size = np.asarray(cfg.geometry.dVoxel, dtype=np.float64)
    size_mm = np.asarray(cfg.geometry.nVoxel, dtype=np.float64) * voxel_size
    offset = np.asarray(cfg.geometry.offOrigin, dtype=np.float64)
    volume_lower = offset - size_mm / 2.0
    physical_axes = [volume_lower[i] + normalized[i] * size_mm[i] for i in range(3)]

    starts = np.asarray(metadata["cube_bounds_zyx_half_open"][0], dtype=np.float64)
    stops = np.asarray(metadata["cube_bounds_zyx_half_open"][1], dtype=np.float64)
    cube_lower = volume_lower + starts * voxel_size
    cube_upper = volume_lower + stops * voxel_size

    inside_axes = [
        np.logical_and(axis >= cube_lower[i], axis < cube_upper[i])
        for i, axis in enumerate(physical_axes)
    ]
    mask = (
        inside_axes[0][:, None, None]
        & inside_axes[1][None, :, None]
        & inside_axes[2][None, None, :]
    )
    attenuation = float(metadata["cube_attenuation_mm_inv"])
    gt = np.zeros(shape, dtype=np.float32)
    gt[mask] = attenuation
    boundary = {
        "interval_convention": "lower-inclusive, upper-exclusive",
        "cube_lower_zyx_mm": cube_lower.tolist(),
        "cube_upper_zyx_mm": cube_upper.tolist(),
        "sample_points_exactly_on_lower_boundary": [
            int(np.count_nonzero(axis == cube_lower[i])) for i, axis in enumerate(physical_axes)
        ],
        "sample_points_exactly_on_upper_boundary": [
            int(np.count_nonzero(axis == cube_upper[i])) for i, axis in enumerate(physical_axes)
        ],
    }
    return gt, mask, physical_axes, boundary


def _projection_scaling(output_dir: Path) -> tuple[float, float, float]:
    projections = np.load(
        output_dir / "projections" / "projections_physical.npy", mmap_mode="r"
    )
    minimum = float(projections.min())
    maximum = float(projections.max())
    value_range = maximum - minimum
    if not np.isfinite(value_range) or value_range <= 0:
        raise ValueError(f"Invalid projection range: min={minimum}, max={maximum}")
    return minimum, maximum, value_range


def _load_or_run_exporter(model_dir: Path, refresh: bool) -> tuple[np.ndarray, Path, bool]:
    import tifffile

    output_file = model_dir / "volumefloat32" / "Static_Volume.tiff"
    ran_exporter = False
    if refresh or not output_file.exists():
        from nect.static_export import export_volume

        exported_dir = Path(export_volume(model_dir))
        output_file = exported_dir / "Static_Volume.tiff"
        ran_exporter = True
    if not output_file.exists():
        raise FileNotFoundError(f"Legacy exporter did not produce {output_file}")
    return tifffile.imread(output_file).astype(np.float32), output_file, ran_exporter


def _difference_stats(left: np.ndarray, right: np.ndarray) -> dict:
    difference = left.astype(np.float64) - right.astype(np.float64)
    return {
        "mean_signed_mm_inv": float(difference.mean()),
        "mean_absolute_mm_inv": float(np.abs(difference).mean()),
        "rmse_mm_inv": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_mm_inv": float(np.abs(difference).max()),
        "allclose_atol_1e-7_rtol_1e-5": bool(
            np.allclose(left, right, atol=1e-7, rtol=1e-5)
        ),
    }


def _plot_profiles(
    diagnostic_dir: Path,
    gt_center: np.ndarray,
    center_volume: np.ndarray,
    center_axes: list[np.ndarray],
    gt_endpoint: np.ndarray,
    endpoint_volume: np.ndarray,
    endpoint_axes: list[np.ndarray],
    exported: np.ndarray,
) -> None:
    figures = diagnostic_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    zc = gt_center.shape[0] // 2
    yc = gt_center.shape[1] // 2

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharey=True)
    axes[0].plot(center_axes[2], gt_center[zc, yc], label="GT at voxel centers", lw=2.5)
    axes[0].plot(center_axes[2], center_volume[zc, yc], label="INR at voxel centers", lw=1.7)
    axes[0].set_title("Voxel-center convention")
    axes[1].plot(endpoint_axes[2], gt_endpoint[zc, yc], label="GT at endpoints", lw=2.5)
    axes[1].plot(endpoint_axes[2], endpoint_volume[zc, yc], label="INR at endpoints", lw=1.7)
    axes[1].plot(endpoint_axes[2], exported[zc, yc], "--", label="Legacy exporter", lw=1.3)
    axes[1].set_title("Endpoint-inclusive convention")
    for ax in axes:
        ax.set(xlabel="Physical x (mm)", ylabel="Attenuation (mm$^{-1}$)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "profiles_by_coordinate_convention.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(center_axes[2], center_volume[zc, yc], label="Center-grid INR", lw=2)
    ax.plot(endpoint_axes[2], endpoint_volume[zc, yc], label="Endpoint-grid INR", lw=2)
    ax.set(
        xlabel="Physical x (mm)",
        ylabel="Attenuation (mm$^{-1}$)",
        title="Direct INR: center grid versus endpoint grid",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "direct_center_vs_endpoint_profile.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(endpoint_axes[2], endpoint_volume[zc, yc], label="Direct endpoint INR", lw=2)
    ax.plot(endpoint_axes[2], exported[zc, yc], "--", label="Legacy exporter", lw=1.5)
    ax.set(
        xlabel="Physical x (mm)",
        ylabel="Attenuation (mm$^{-1}$)",
        title="Endpoint-grid INR versus legacy export",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "endpoint_direct_vs_exporter_profile.png", dpi=200)
    plt.close(fig)


def _plot_difference_maps(
    diagnostic_dir: Path,
    center_volume: np.ndarray,
    endpoint_volume: np.ndarray,
    endpoint_export_formula: np.ndarray,
    exported: np.ndarray,
) -> None:
    zc = center_volume.shape[0] // 2
    coordinate_difference = endpoint_volume - center_volume
    exporter_residual = exported - endpoint_export_formula
    coordinate_limit = max(1e-8, float(np.percentile(np.abs(coordinate_difference), 99.5)))
    residual_limit = max(1e-10, float(np.max(np.abs(exporter_residual))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    images = [coordinate_difference[zc], exporter_residual[zc]]
    titles = [
        "Coordinate effect\nendpoint INR - center INR",
        "Residual exporter effect\nexport - endpoint legacy formula",
    ]
    limits = [coordinate_limit, residual_limit]
    for ax, image, title, limit in zip(axes, images, titles, limits):
        artist = ax.imshow(image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel="x index", ylabel="y index")
        fig.colorbar(artist, ax=ax, shrink=0.8, label="mm$^{-1}$")
    fig.tight_layout()
    fig.savefig(diagnostic_dir / "figures" / "difference_maps.png", dpi=200)
    plt.close(fig)


def diagnose_export(
    output_dir: Path,
    model_dir: Path,
    *,
    refresh_export: bool = False,
) -> dict:
    """Compare coordinate conventions without training or modifying NeCT."""
    metadata_path = output_dir / "ground_truth" / "metadata.json"
    projection_path = output_dir / "projections" / "projections_physical.npy"
    checkpoint_path = model_dir / "checkpoints" / "last.ckpt"
    config_path = model_dir / "config.yaml"
    required = [metadata_path, projection_path, checkpoint_path, config_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Export diagnostic requires the existing Stage 1 artifacts and will not reconstruct. "
            f"Missing: {', '.join(missing)}"
        )

    metadata = read_json(metadata_path)
    shape = tuple(int(value) for value in metadata["shape_zyx"])
    model, device, cfg, geometry = _load_model(model_dir)
    import torch

    training_dependent_modules = [
        module.__class__.__name__
        for module in model.modules()
        if isinstance(
            module,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.modules.dropout._DropoutNd,
            ),
        )
    ]
    raw_center = _evaluate_raw_volume(model, device, shape, "voxel_centers")
    raw_endpoint = _evaluate_raw_volume(model, device, shape, "endpoints")

    projection_min, projection_max, projection_range = _projection_scaling(output_dir)
    physical_factor = projection_range / float(geometry.max_distance_traveled)
    center_volume = raw_center * physical_factor
    endpoint_volume = raw_endpoint * physical_factor
    endpoint_export_formula = endpoint_volume + projection_min
    exported, exporter_file, ran_exporter = _load_or_run_exporter(
        model_dir, refresh_export
    )
    if exported.shape != shape:
        raise ValueError(
            f"Exporter shape {exported.shape} differs from expected z,y,x shape {shape}. "
            "The legacy exporter derives both y and x sizes from nVoxel[1]."
        )

    gt_center, mask_center, center_axes, center_boundary = _sample_cube_ground_truth(
        shape, "voxel_centers", metadata, cfg
    )
    gt_endpoint, mask_endpoint, endpoint_axes, endpoint_boundary = _sample_cube_ground_truth(
        shape, "endpoints", metadata, cfg
    )

    diagnostic_dir = output_dir / "export_diagnostic"
    volumes_dir = diagnostic_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    np.save(volumes_dir / "ground_truth_voxel_centers.npy", gt_center)
    np.save(volumes_dir / "ground_truth_endpoints.npy", gt_endpoint)
    np.save(volumes_dir / "direct_voxel_centers_mm_inv.npy", center_volume.astype(np.float32))
    np.save(volumes_dir / "direct_endpoints_mm_inv.npy", endpoint_volume.astype(np.float32))
    np.save(volumes_dir / "endpoint_legacy_formula_mm_inv.npy", endpoint_export_formula.astype(np.float32))
    np.save(volumes_dir / "original_exporter_mm_inv.npy", exported)
    np.save(volumes_dir / "coordinate_effect_indexwise.npy", (endpoint_volume - center_volume).astype(np.float32))
    np.save(volumes_dir / "exporter_residual.npy", (exported - endpoint_export_formula).astype(np.float32))

    _plot_profiles(
        diagnostic_dir,
        gt_center,
        center_volume,
        center_axes,
        gt_endpoint,
        endpoint_volume,
        endpoint_axes,
        exported,
    )
    _plot_difference_maps(
        diagnostic_dir,
        center_volume,
        endpoint_volume,
        endpoint_export_formula,
        exported,
    )

    center_index = tuple(n // 2 for n in shape)
    profile_path = diagnostic_dir / "profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index_x",
                "center_x_mm",
                "center_gt_mm_inv",
                "center_inr_mm_inv",
                "endpoint_x_mm",
                "endpoint_gt_mm_inv",
                "endpoint_inr_mm_inv",
                "exporter_mm_inv",
            ]
        )
        zc, yc = center_index[0], center_index[1]
        for x_index in range(shape[2]):
            writer.writerow(
                [
                    x_index,
                    center_axes[2][x_index],
                    gt_center[zc, yc, x_index],
                    center_volume[zc, yc, x_index],
                    endpoint_axes[2][x_index],
                    gt_endpoint[zc, yc, x_index],
                    endpoint_volume[zc, yc, x_index],
                    exported[zc, yc, x_index],
                ]
            )

    center_spacing = [float(axis[1] - axis[0]) for axis in center_axes]
    endpoint_spacing = [float(axis[1] - axis[0]) for axis in endpoint_axes]
    metrics_a = _metrics(center_volume, gt_center, mask_center)
    metrics_b = _metrics(endpoint_volume, gt_endpoint, mask_endpoint)
    metrics_c = _metrics(exported, gt_endpoint, mask_endpoint)
    comparison_ab = _difference_stats(center_volume, endpoint_volume)
    comparison_bc = _difference_stats(endpoint_volume, exported)
    comparison_legacy_c = _difference_stats(endpoint_export_formula, exported)
    metrics = {
        "model_directory": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "exporter_file": str(exporter_file),
        "ran_exporter_during_diagnostic": ran_exporter,
        "array_order": "zyx",
        "scaling": {
            "projection_min": projection_min,
            "projection_max": projection_max,
            "projection_range": projection_range,
            "max_distance_traveled_mm": float(geometry.max_distance_traveled),
            "direct_physical_multiplier": physical_factor,
            "legacy_exporter_additive_minimum": projection_min,
        },
        "coordinates": {
            "voxel_centers": {
                "normalized_formula": "(i + 0.5) / N",
                "physical_first_zyx_mm": [float(axis[0]) for axis in center_axes],
                "physical_last_zyx_mm": [float(axis[-1]) for axis in center_axes],
                "physical_spacing_zyx_mm": center_spacing,
                "ground_truth_boundary": center_boundary,
            },
            "endpoints": {
                "normalized_formula": "i / (N - 1)",
                "physical_first_zyx_mm": [float(axis[0]) for axis in endpoint_axes],
                "physical_last_zyx_mm": [float(axis[-1]) for axis in endpoint_axes],
                "physical_spacing_zyx_mm": endpoint_spacing,
                "ground_truth_boundary": endpoint_boundary,
            },
        },
        "A_direct_voxel_centers": metrics_a,
        "B_direct_endpoints": metrics_b,
        "C_original_exporter_against_endpoint_gt": metrics_c,
        "comparisons": {
            "A_vs_B_indexwise_coordinate_effect": comparison_ab,
            "B_physical_vs_C_exporter": comparison_bc,
            "B_legacy_formula_vs_C_exporter": comparison_legacy_c,
        },
        "cube_mean_attribution_mm_inv": {
            "observed_C_minus_A": metrics_c["cube_mean_mm_inv"]
            - metrics_a["cube_mean_mm_inv"],
            "coordinate_B_minus_A": metrics_b["cube_mean_mm_inv"]
            - metrics_a["cube_mean_mm_inv"],
            "legacy_projection_minimum_added": projection_min,
            "residual_C_minus_B_legacy_formula": float(
                (exported - endpoint_export_formula)[mask_endpoint].mean()
            ),
        },
        "verified_pipeline": {
            "same_checkpoint_for_A_and_B": True,
            "A_and_B_model_eval_mode": True,
            "exporter_calls_model_eval": False,
            "configured_model": str(cfg.model),
            "configured_output_activation": str(cfg.net.output_activation),
            "known_training_dependent_modules": training_dependent_modules,
            "network_has_known_training_dependent_layers": bool(
                training_dependent_modules
            ),
            "axis_order_A_B_C": "zyx for this cubic experiment",
            "exporter_uses_nVoxel_1_for_both_y_and_x": True,
        },
    }
    write_json(diagnostic_dir / "metrics.json", metrics)
    attribution = metrics["cube_mean_attribution_mm_inv"]
    summary_lines = [
        "NeCT export diagnostic",
        f"Checkpoint: {checkpoint_path}",
        "",
        f"A center-grid cube mean:   {metrics_a['cube_mean_mm_inv']:.9g} mm^-1",
        f"B endpoint-grid cube mean: {metrics_b['cube_mean_mm_inv']:.9g} mm^-1",
        f"C exporter cube mean:      {metrics_c['cube_mean_mm_inv']:.9g} mm^-1",
        "",
        f"Observed C-A cube-mean difference: {attribution['observed_C_minus_A']:.9g} mm^-1",
        f"Coordinate contribution B-A:       {attribution['coordinate_B_minus_A']:.9g} mm^-1",
        f"Legacy projection minimum added:   {projection_min:.9g} mm^-1",
        "Residual exporter contribution after endpoint coordinates and legacy scaling: "
        f"{attribution['residual_C_minus_B_legacy_formula']:.9g} mm^-1",
        "",
        "Endpoint legacy formula versus exporter:",
        f"  MAE: {comparison_legacy_c['mean_absolute_mm_inv']:.9g} mm^-1",
        f"  max absolute: {comparison_legacy_c['maximum_absolute_mm_inv']:.9g} mm^-1",
        f"  allclose: {comparison_legacy_c['allclose_atol_1e-7_rtol_1e-5']}",
    ]
    (diagnostic_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    return metrics
