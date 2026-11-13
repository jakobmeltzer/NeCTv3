from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json


def _checkpoint_epoch(checkpoint: Path) -> int:
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    return int(state.get("epoch", -1))


def write_dynamic_completion(config: dict, output_dir: Path, run_dir: Path) -> Path:
    model_dir = run_dir / "model"
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    expected = int(config["dynamic_reconstruction"]["epochs"])
    if not checkpoint.exists() or _checkpoint_epoch(checkpoint) < expected:
        raise RuntimeError("Dynamic checkpoint is absent or incomplete; completion marker not written")
    marker = output_dir / "reconstruction" / "completed_run.json"
    write_json(marker, {
        "run_directory": str(run_dir.resolve()),
        "model_directory": str(model_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": _checkpoint_epoch(checkpoint),
        "epochs": expected,
        "architecture": "mixedcubes",
        "projection_input": str((output_dir / "projections" / "projections_physical.npy").resolve()),
        "angles_input": str((output_dir / "projections" / "angles_nect_rad.npy").resolve()),
        "timestamps_input": str((output_dir / "projections" / "timestamps_normalized.npy").resolve()),
        "coordinate_frame": "TIGRE physical world via pi-shifted NeCT training angles",
    })
    return model_dir


def run_dynamic_reconstruction(config: dict, output_dir: Path) -> Path:
    import nect

    geom = config["geometry"]
    vol = config["volume"]
    angles = np.load(output_dir / "projections" / "angles_nect_rad.npy")
    times = np.load(output_dir / "projections" / "timestamps_normalized.npy")
    projection_path = (output_dir / "projections" / "projections_physical.npy").resolve()
    geometry = nect.Geometry(
        mode="cone", DSO=geom["DSO_mm"], DSD=geom["DSD_mm"],
        nDetector=geom["nDetector_hw"], dDetector=geom["dDetector_hw_mm"],
        nVoxel=vol["shape_zyx"], dVoxel=vol["voxel_size_zyx_mm"],
        offOrigin=geom["offOrigin_zyx_mm"], offDetector=geom["offDetector_hw_mm"],
        COR=geom["COR_mm"], angles=angles, timesteps=times, radians=True,
    )
    rcfg = config["dynamic_reconstruction"]
    override = {
        "epochs": int(rcfg["epochs"]), "loss": rcfg["loss"],
        "batch_per_proj": rcfg["batch_per_proj"], "points_per_batch": rcfg["points_per_batch"],
        "points_per_ray": {"start": rcfg["points_per_ray"], "end": rcfg["points_per_ray"], "update_interval": 1000, "linear": True},
        "downsampling_detector": {"start": 1, "end": 1, "update_interval": 1000},
        "checkpoint_interval": float(rcfg["checkpoint_interval_seconds"]),
        "checkpoint_epoch": 5, "image_interval": -1, "add_poisson": False,
        "use_extended_forward_model": False, "save_volume": False,
        "tv": 0.0, "tv_temporal": 0.0, "tv_spatial": 0.0,
    }
    root = (output_dir / "reconstruction").resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = nect.reconstruct(
        geometry=geometry, projections=projection_path, mode="dynamic",
        angles=angles, timesteps=times, radians=True, enc_arc="mixedcubes",
        niter=int(rcfg["epochs"]), verbose=True, log=True, exp_name=str(root),
        channel_order="NHW", config_override=override, save_ckpt=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected dynamic reconstruct result: {result!r}")
    # As in the static API, the second tuple member is the experiment run root;
    # the first dynamic member points at the model/checkpoint area.
    run_dir = Path(result[1]).resolve()
    write_dynamic_completion(config, output_dir, run_dir)
    return run_dir


def resume_dynamic_reconstruction(config: dict, output_dir: Path, model_dir: Path) -> Path:
    from nect.reconstruct import reconstruct_from_config_file

    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    cfg = model_dir / "config.yaml"
    reconstruct_from_config_file(cfg=cfg, checkpoint=str(checkpoint), save_ckpt=True, save_last=True, save_optimizer=True, prune=False, keep_two=True)
    run_dir = model_dir.parent
    write_dynamic_completion(config, output_dir, run_dir)
    return run_dir
