#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.attenuation_validation.common import collect_provenance, read_json, set_cpu_seeds, write_json
from experiments.attenuation_validation.dynamic_config import dynamic_experiment_config

STAGES = ("phantom", "projections", "reconstruct", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3: dynamic PoreSpy/MixedCubes validation")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage2-dir", type=Path, default=REPOSITORY_ROOT / "experiments/attenuation_validation/outputs/stage2_porous")
    parser.add_argument("--num-projections", type=int, default=720)
    parser.add_argument("--num-revolutions", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stored-states", type=int, default=41)
    parser.add_argument("--smoke", action="store_true", help="24 projections, 1 revolution, 2 epochs and 7 states")
    parser.add_argument("--force", action="store_true", help="Regenerate only the selected non-reconstruction stage")
    return parser.parse_args()


def _completed_reconstruction(output_dir: Path, config: dict) -> Path | None:
    marker_path = output_dir / "reconstruction" / "completed_run.json"
    if not marker_path.exists(): return None
    marker = read_json(marker_path)
    checkpoint = Path(marker["checkpoint"])
    if not checkpoint.exists() or int(marker["epochs"]) != int(config["dynamic_reconstruction"]["epochs"]):
        raise RuntimeError("Dynamic reconstruction completion marker is inconsistent")
    import torch
    try: state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError: state = torch.load(checkpoint, map_location="cpu")
    if int(state.get("epoch", -1)) < int(marker["epochs"]) or int(state.get("epoch", -1)) != int(marker["checkpoint_epoch"]):
        raise RuntimeError("Dynamic checkpoint epoch disagrees with its completion marker")
    return Path(marker["model_directory"])


def _require_artifacts(paths: list[Path], stage: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing: raise RuntimeError(f"{stage} completion marker exists but artifacts are missing: {missing}")


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.num_projections, args.num_revolutions, args.epochs, args.stored_states = 24, 1.0, 2, 7
    output_dir = (args.output_dir or REPOSITORY_ROOT / "experiments/attenuation_validation/outputs" / ("stage3_dynamic_smoke" if args.smoke else "stage3_dynamic")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dynamic_experiment_config(args.num_projections, args.num_revolutions, args.epochs, args.stored_states)
    config_path = output_dir / "experiment_config.json"
    if config_path.exists():
        existing = read_json(config_path)
        if existing != config: raise SystemExit(f"Existing configuration differs: {config_path}. Use matching arguments or another output directory.")
    else: write_json(config_path, config)
    if not (output_dir / "provenance.json").exists(): write_json(output_dir / "provenance.json", collect_provenance(REPOSITORY_ROOT))
    set_cpu_seeds(int(config["seed"]))
    stages = STAGES if args.stage == "all" else (args.stage,)
    print(f"Stage 3 output: {output_dir}\nSelected stages: {', '.join(stages)}", flush=True)
    if args.force and "phantom" in stages and (output_dir / "projections" / "completed.json").exists():
        raise SystemExit("Refusing to replace ground truth beneath completed projections; use a new --output-dir.")
    if args.force and "projections" in stages and (output_dir / "reconstruction" / "completed_run.json").exists():
        raise SystemExit("Refusing to replace projections beneath a completed reconstruction; use a new --output-dir.")

    if "phantom" in stages:
        marker = output_dir / "ground_truth" / "completed.json"
        if marker.exists() and not args.force:
            _require_artifacts([output_dir / "ground_truth" / name for name in ["rock_mask.npy", "pore_mask.npy", "fluid_masks.npy", "material_labels.npy", "state_times_normalized.npy", "saturation.npy", "metadata.json"]], "phantom")
            print("[phantom] Reusing completed invasion.", flush=True)
        else:
            from experiments.attenuation_validation.dynamic_phantom import generate_dynamic_phantom
            metadata = generate_dynamic_phantom(config, output_dir, args.stage2_dir.resolve())
            print(f"[phantom] final saturation={metadata['final_saturation']:.6g}", flush=True)

    if "projections" in stages:
        if not (output_dir / "ground_truth" / "completed.json").exists(): raise SystemExit("Ground truth is missing; run --stage phantom")
        marker = output_dir / "projections" / "completed.json"
        if marker.exists() and not args.force:
            _require_artifacts([output_dir / "projections" / name for name in ["projections_physical.npy", "timestamps_normalized.npy", "angles_tigre_rad.npy", "angles_nect_rad.npy", "validation.json"]], "projections")
            print("[projections] Reusing completed dynamic projections.", flush=True)
        else:
            from experiments.attenuation_validation.dynamic_projections import generate_dynamic_projections
            validation = generate_dynamic_projections(config, output_dir)
            print(f"[projections] shape={validation['shape_nhw']}, states observed={validation['unique_ground_truth_states_observed']}", flush=True)

    if "reconstruct" in stages:
        if not (output_dir / "projections" / "completed.json").exists(): raise SystemExit("Projections are missing; run --stage projections")
        model_dir = _completed_reconstruction(output_dir, config)
        if model_dir is not None: print(f"[reconstruct] Reusing completed MixedCubes model: {model_dir}", flush=True)
        else:
            from experiments.attenuation_validation.dynamic_reconstruction import resume_dynamic_reconstruction, run_dynamic_reconstruction
            checkpoints = sorted((output_dir / "reconstruction").glob("**/model/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                model_dir = checkpoints[0].parent.parent
                print(f"[reconstruct] Resuming {model_dir}", flush=True)
                resume_dynamic_reconstruction(config, output_dir, model_dir)
            else:
                print("[reconstruct] Starting MixedCubes training.", flush=True)
                run_dynamic_reconstruction(config, output_dir)
            model_dir = _completed_reconstruction(output_dir, config)
            if model_dir is None: raise RuntimeError("Training returned without a valid completion marker")

    if "evaluate" in stages:
        model_dir = _completed_reconstruction(output_dir, config)
        if model_dir is None: raise SystemExit("Evaluation will not reconstruct: completed checkpoint marker missing")
        marker = output_dir / "evaluation" / "completed.json"
        if marker.exists() and not args.force: print("[evaluate] Reusing completed evaluation.", flush=True)
        else:
            from experiments.attenuation_validation.dynamic_evaluation import evaluate_dynamic
            report = evaluate_dynamic(output_dir, model_dir, config)
            final = report["frames"][-1]
            print(f"[evaluate] final MAE={final['whole_volume']['mae_mm_inv']:.6g}, saturation error={final['saturation_error']:.6g}", flush=True)
    print("Stage 3 selected stages completed successfully.", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
