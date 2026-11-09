from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import write_json


def _write_completion(
    output_dir: Path,
    run_dir: Path,
    projections_path: Path,
    epochs: int,
    seed: int,
) -> None:
    model_dir = run_dir / "model"
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        raise RuntimeError(f"Final checkpoint is missing: {checkpoint}")
    write_json(
        output_dir / "reconstruction" / "completed_run.json",
        {
            "run_directory": str(run_dir),
            "model_directory": str(model_dir),
            "projection_input": str(projections_path.resolve()),
            "projection_input_kind": "single_npy_file",
            "legacy_file_normalization_expected": True,
            "epochs": int(epochs),
            "seed": seed,
        },
    )


def run_legacy_reconstruction(
    config: dict,
    output_dir: Path,
    *,
    epochs: int = 20,
) -> Path:
    """Run unchanged static NeCT using the saved projection file path.

    Passing a path is intentional: it exercises the original file-based min-max
    normalization that this baseline is designed to measure.
    """
    try:
        import nect
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "NeCT and its CUDA extensions are required for reconstruction. "
            "Run this stage on an IDUN GPU node."
        ) from exc

    geom = config["geometry"]
    vol = config["volume"]
    angles = np.load(output_dir / "projections" / "angles_rad.npy")
    projections_path = output_dir / "projections" / "projections_physical.npy"

    geometry = nect.Geometry(
        mode="cone",
        DSO=geom["DSO_mm"],
        DSD=geom["DSD_mm"],
        nDetector=geom["nDetector_hw"],
        dDetector=geom["dDetector_hw_mm"],
        nVoxel=vol["shape_zyx"],
        dVoxel=vol["voxel_size_zyx_mm"],
        offOrigin=geom["offOrigin_zyx_mm"],
        offDetector=geom["offDetector_hw_mm"],
        COR=geom["COR_mm"],
        angles=angles,
        radians=True,
    )

    config_override = {
        "epochs": int(epochs),
        "loss": "L2",
        "batch_per_proj": 1,
        "points_per_batch": 1_000_000,
        "downsampling_detector": {
            "start": 1,
            "end": 1,
            "update_interval": 1000,
        },
        "points_per_ray": {
            "start": 96,
            "end": 96,
            "update_interval": 1000,
            "linear": True,
        },
        "image_interval": -1,
        "checkpoint_interval": -1,
        "checkpoint_epoch": 5,
        "add_poisson": False,
        "s3im": False,
        "use_extended_forward_model": False,
        "save_volume": True,
    }

    reconstruction_root = (output_dir / "reconstruction").resolve()
    reconstruction_root.mkdir(parents=True, exist_ok=True)
    result = nect.reconstruct(
        geometry=geometry,
        projections=projections_path.resolve(),
        mode="static",
        angles=angles,
        radians=True,
        quality="medium",
        niter=epochs,
        verbose=True,
        log=True,
        exp_name=str(reconstruction_root),
        channel_order="NHW",
        config_override=config_override,
        save_ckpt=True,
    )

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected return value from nect.reconstruct: {type(result)!r}")
    api_volume, run_dir = result
    run_dir = Path(run_dir).resolve()
    np.save(output_dir / "reconstruction" / "legacy_reconstruct_api_volume.npy", api_volume)
    _write_completion(
        output_dir, run_dir, projections_path, epochs, config["seed"]
    )
    return run_dir


def resume_legacy_reconstruction(config: dict, output_dir: Path, model_dir: Path) -> Path:
    """Resume an interrupted legacy run from NeCT's saved full checkpoint."""
    try:
        from nect.reconstruct import reconstruct_from_config_file
    except (ImportError, OSError) as exc:
        raise RuntimeError("NeCT CUDA environment is required to resume reconstruction.") from exc

    model_dir = model_dir.resolve()
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    config_path = model_dir / "config.yaml"
    if not checkpoint.exists() or not config_path.exists():
        raise FileNotFoundError(f"Incomplete resume artifacts under {model_dir}")
    reconstruct_from_config_file(
        cfg=config_path,
        checkpoint=str(checkpoint),
        save_ckpt=True,
        save_last=True,
        save_optimizer=True,
        prune=False,
        keep_two=True,
    )
    run_dir = model_dir.parent
    projections_path = output_dir / "projections" / "projections_physical.npy"
    _write_completion(
        output_dir,
        run_dir,
        projections_path,
        int(config.get("legacy_reconstruction", {}).get("epochs", 20)),
        config["seed"],
    )
    return run_dir

