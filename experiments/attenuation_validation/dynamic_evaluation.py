from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .common import read_json, write_json
from .dynamic_phantom import attenuation_at_time
from .evaluation import _load_model, _plot_loss
from .export_diagnostic import _projection_scaling
from .porous_evaluation import _write_ply


def _evaluate_frame(model, device, shape: tuple[int, int, int], time_value: float) -> np.ndarray:
    import torch

    axes = [torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=device) for n in shape]
    output = np.empty(shape, dtype=np.float32)
    with torch.no_grad():
        for zi, z in enumerate(axes[0]):
            zz, yy, xx = torch.meshgrid(z[None], axes[1], axes[2], indexing="ij")
            points = torch.stack((zz.ravel(), yy.ravel(), xx.ravel()), dim=1)
            output[zi] = model(points, float(time_value)).reshape(shape[1:]).float().cpu().numpy()
    return output


def _error_metrics(rec: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"voxel_count": 0}
    error = rec[mask].astype(np.float64) - gt[mask].astype(np.float64)
    return {
        "voxel_count": int(mask.sum()),
        "ground_truth_mean_mm_inv": float(gt[mask].mean()),
        "reconstructed_mean_mm_inv": float(rec[mask].mean()),
        "bias_mm_inv": float(error.mean()),
        "mae_mm_inv": float(np.abs(error).mean()),
        "rmse_mm_inv": float(np.sqrt(np.mean(error**2))),
        "error_standard_deviation_mm_inv": float(error.std()),
    }


def _segmentation(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth)); tn = int(np.sum(~pred & ~truth))
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "iou": tp / max(1, tp + fp + fn), "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn), "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
    }


def _boundary_masks(rock: np.ndarray, fluid: np.ndarray, air: np.ndarray) -> dict[str, np.ndarray]:
    structure = ndimage.generate_binary_structure(3, 1)
    rock_interface = rock & ndimage.binary_dilation(~rock, structure=structure)
    pore_interface = (~rock) & ndimage.binary_dilation(rock, structure=structure)
    fluid_air = (fluid & ndimage.binary_dilation(air, structure=structure)) | (air & ndimage.binary_dilation(fluid, structure=structure))
    any_material_interface = rock_interface | pore_interface | fluid_air
    return {
        "rock_pore_interface": rock_interface | pore_interface,
        "fluid_air_interface": fluid_air,
        "interior_rock": rock & ~ndimage.binary_dilation(~rock, structure=structure, iterations=2),
        "interior_fluid": fluid & ~ndimage.binary_dilation(~fluid, structure=structure, iterations=2),
        "interior_air": air & ~ndimage.binary_dilation(~air, structure=structure, iterations=2),
        "all_material_interfaces": any_material_interface,
    }


def evaluate_dynamic(output_dir: Path, model_dir: Path, config: dict) -> dict:
    model, device, cfg, geometry = _load_model(model_dir)
    rock = np.load(output_dir / "ground_truth" / "rock_mask.npy").astype(bool)
    pore = ~rock
    times = np.asarray(config["evaluation"]["times_normalized"], dtype=np.float32)
    pmin, pmax, prange = _projection_scaling(output_dir)
    scale = prange / float(geometry.max_distance_traveled)
    root = output_dir / "evaluation"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "volumes").mkdir(parents=True, exist_ok=True)
    (root / "visualizations").mkdir(parents=True, exist_ok=True)

    frame_metrics = []
    reconstructions = []
    truths = []
    true_saturation = []
    estimated_saturation = []
    fluid_threshold = float(config["evaluation"]["fluid_threshold_mm_inv"])
    for index, time_value in enumerate(times):
        gt, fluid, state_index = attenuation_at_time(output_dir, config, float(time_value))
        raw = _evaluate_frame(model, device, tuple(gt.shape), float(time_value))
        rec = raw * scale  # corrected training angles: model and TIGRE world frames coincide
        air = pore & ~fluid
        predicted_fluid_known_pore = pore & (rec >= fluid_threshold)
        predicted_fluid_practical = (rec >= fluid_threshold) & (rec < float(config["evaluation"]["rock_threshold_mm_inv"]))
        true_sat = float(fluid.sum() / max(1, pore.sum()))
        est_sat = float(predicted_fluid_known_pore.sum() / max(1, pore.sum()))
        boundaries = _boundary_masks(rock, fluid, air)
        structure = ndimage.generate_binary_structure(3, 1)
        true_front = fluid & ndimage.binary_dilation(air, structure=structure)
        predicted_air = pore & ~predicted_fluid_known_pore
        predicted_front = predicted_fluid_known_pore & ndimage.binary_dilation(predicted_air, structure=structure)
        components, component_count = ndimage.label(predicted_fluid_known_pore, structure=structure)
        component_sizes = np.bincount(components.ravel())[1:]
        outer_distance = ndimage.distance_transform_edt(np.pad(np.ones(gt.shape, dtype=bool), 1))[1:-1, 1:-1, 1:-1]
        outer_metrics = {}
        for distance in config["evaluation"]["outer_boundary_distances_voxels"]:
            mask = outer_distance <= float(distance)
            outer_metrics[f"within_{distance}_voxels"] = _error_metrics(rec, gt, mask)
        metrics = {
            "time_normalized": float(time_value), "ground_truth_state_index": state_index,
            "whole_volume": _error_metrics(rec, gt, np.ones(gt.shape, dtype=bool)),
            "rock": _error_metrics(rec, gt, rock), "air": _error_metrics(rec, gt, air),
            "fluid": _error_metrics(rec, gt, fluid),
            "true_fluid_saturation": true_sat, "estimated_fluid_saturation_known_pore": est_sat,
            "saturation_error": est_sat - true_sat,
            "fluid_segmentation_known_pore": _segmentation(predicted_fluid_known_pore[pore], fluid[pore]),
            "fluid_segmentation_practical_whole_volume": _segmentation(predicted_fluid_practical, fluid),
            "invasion_front_segmentation_known_pore": _segmentation(predicted_front[pore], true_front[pore]),
            "predicted_fluid_connectivity": {
                "component_count_6_connected": int(component_count),
                "largest_component_fraction": float(component_sizes.max() / max(1, predicted_fluid_known_pore.sum())) if component_sizes.size else 0.0,
            },
            "regions": {name: _error_metrics(rec, gt, mask) for name, mask in boundaries.items()},
            "outer_volume_boundary": outer_metrics,
        }
        frame_metrics.append(metrics); reconstructions.append(rec.astype(np.float32)); truths.append(gt)
        true_saturation.append(true_sat); estimated_saturation.append(est_sat)
        np.save(root / "volumes" / f"reconstruction_t{index:02d}.npy", rec.astype(np.float32))
        np.save(root / "volumes" / f"ground_truth_t{index:02d}.npy", gt.astype(np.float32))
        _write_ply(root / "visualizations" / f"ground_truth_fluid_t{index:02d}.ply", fluid, (30, 120, 240), 30_000)
        _write_ply(root / "visualizations" / f"reconstructed_fluid_t{index:02d}.ply", predicted_fluid_known_pore, (240, 130, 30), 30_000)

    rec4d = np.asarray(reconstructions); gt4d = np.asarray(truths)
    np.savez_compressed(root / "volumes" / "evaluation_series.npz", times=times, reconstruction=rec4d, ground_truth=gt4d)
    rock_temporal_std = np.std(rec4d[:, rock], axis=0)
    report = {
        "model_directory": str(model_dir), "architecture": str(cfg.model),
        "coordinate_frame": "TIGRE physical world; corrected angles used in training; no evaluation rotation",
        "attenuation_scaling": {"projection_min": pmin, "projection_max": pmax, "projection_range": prange, "max_distance_traveled_mm": float(geometry.max_distance_traveled)},
        "fluid_classification": {"known_pore_threshold_mm_inv": fluid_threshold, "practical_rock_upper_threshold_mm_inv": config["evaluation"]["rock_threshold_mm_inv"]},
        "frames": frame_metrics,
        "stationary_rock_temporal_std_mean_mm_inv": float(rock_temporal_std.mean()),
        "stationary_rock_temporal_std_p95_mm_inv": float(np.percentile(rock_temporal_std, 95)),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(times, true_saturation, "o-", label="Ground truth"); ax.plot(times, estimated_saturation, "s--", label="Reconstruction")
    ax.set(xlabel="Normalized acquisition time", ylabel="Fluid saturation of pore space", ylim=(0, 1)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(root / "figures" / "fluid_saturation.png", dpi=220); plt.close(fig)
    z = rock.shape[0] // 2
    fig, axes = plt.subplots(len(times), 3, figsize=(12, 3.2 * len(times)))
    for i, t in enumerate(times):
        error = rec4d[i] - gt4d[i]; lim = max(.002, float(np.percentile(np.abs(error), 99)))
        for ax, image, title, cmap, limits in [(axes[i,0], gt4d[i,z], f"GT t={t:.2f}", "gray", (0,.02)), (axes[i,1], rec4d[i,z], "MixedCubes", "gray", (0,.02)), (axes[i,2], error[z], "Signed error", "coolwarm", (-lim,lim))]:
            ax.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1]); ax.set_title(title); ax.set(xlabel="x", ylabel="y")
    fig.tight_layout(); fig.savefig(root / "figures" / "timepoint_slices.png", dpi=180); plt.close(fig)

    coords = np.argwhere(pore)
    chosen = coords[np.linspace(0, len(coords)-1, 5, dtype=int)]
    with (root / "temporal_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["point", "z", "y", "x", "time", "ground_truth_mm_inv", "reconstruction_mm_inv"])
        for pidx, (zz, yy, xx) in enumerate(chosen):
            for ti, t in enumerate(times): writer.writerow([pidx, zz, yy, xx, t, gt4d[ti,zz,yy,xx], rec4d[ti,zz,yy,xx]])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 2.4 * len(chosen)), sharex=True)
    for pidx, ((zz, yy, xx), ax) in enumerate(zip(chosen, np.atleast_1d(axes))):
        ax.step(times, gt4d[:,zz,yy,xx], where="post", label="Ground truth", lw=2)
        ax.plot(times, rec4d[:,zz,yy,xx], "o-", label="MixedCubes")
        ax.set(ylabel="mm$^{-1}$", title=f"Pore point {pidx}: z,y,x=({zz},{yy},{xx})"); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("Normalized acquisition time"); fig.tight_layout(); fig.savefig(root / "figures" / "temporal_attenuation_profiles.png", dpi=220); plt.close(fig)

    zc, yc = rock.shape[0] // 2, rock.shape[1] // 2
    x_mm = np.arange(rock.shape[2]) + 0.5 - rock.shape[2] / 2
    selected_frames = sorted({0, len(times) // 2, len(times) - 1})
    fig, axes = plt.subplots(len(selected_frames), 1, figsize=(9, 2.8 * len(selected_frames)), sharex=True)
    for frame, ax in zip(selected_frames, np.atleast_1d(axes)):
        ax.step(x_mm, gt4d[frame, zc, yc], where="mid", lw=2, label="Ground truth")
        ax.plot(x_mm, rec4d[frame, zc, yc], label="MixedCubes")
        ax.set(ylabel="mm$^{-1}$", title=f"Central x profile, t={times[frame]:.2f}"); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel("x (mm)"); fig.tight_layout(); fig.savefig(root / "figures" / "spatial_attenuation_profiles.png", dpi=220); plt.close(fig)

    final_gt, final_rec = gt4d[-1], rec4d[-1]
    final_fluid = np.isclose(final_gt, config["volume"]["fluid_attenuation_mm_inv"])
    final_air = np.isclose(final_gt, config["volume"]["air_attenuation_mm_inv"])
    fig, ax = plt.subplots(figsize=(8, 4.5)); bins = np.linspace(-.002, .024, 120)
    ax.hist(final_rec[rock], bins=bins, alpha=.55, density=True, label="Rock")
    ax.hist(final_rec[final_fluid], bins=bins, alpha=.55, density=True, label="Fluid")
    ax.hist(final_rec[final_air], bins=bins, alpha=.55, density=True, label="Air")
    for value in [0, .01, .02]: ax.axvline(value, color="black", ls="--", lw=.8)
    ax.set(xlabel="Reconstructed attenuation (mm$^{-1}$)", ylabel="Density", title="Final-state material distributions"); ax.legend(); fig.tight_layout(); fig.savefig(root / "figures" / "material_distributions_final.png", dpi=220); plt.close(fig)

    pred_final = pore & (final_rec >= fluid_threshold)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(final_fluid[z], origin="lower", cmap="Blues"); axes[0].set_title("True fluid")
    axes[1].imshow(pred_final[z], origin="lower", cmap="Oranges"); axes[1].set_title("Reconstructed fluid")
    axes[2].imshow((pred_final ^ final_fluid)[z], origin="lower", cmap="Reds"); axes[2].set_title("Occupancy error")
    for ax in axes: ax.set(xlabel="x", ylabel="y")
    fig.tight_layout(); fig.savefig(root / "figures" / "fluid_segmentation_final.png", dpi=220); plt.close(fig)
    _plot_loss(output_dir, model_dir, root / "figures")
    write_json(root / "metrics.json", report)
    write_json(root / "completed.json", {"metrics": str(root / "metrics.json")})
    return report
