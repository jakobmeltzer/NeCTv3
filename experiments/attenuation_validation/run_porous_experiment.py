#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.attenuation_validation.common import (  # noqa: E402
    collect_provenance,
    read_json,
    set_cpu_seeds,
    write_json,
)
from experiments.attenuation_validation.porous_config import porous_experiment_config  # noqa: E402
from experiments.attenuation_validation.porous_phantom import (  # noqa: E402
    generate_porous_phantom,
    load_porous_phantom,
)
from experiments.attenuation_validation.porous_projections import generate_porous_projections  # noqa: E402


STAGES = ("phantom", "projections", "reconstruct", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2: static PoreSpy attenuation validation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "attenuation_validation" / "outputs" / "stage2_porous",
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-angles", type=int, default=180)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate deterministic lightweight inputs; never deletes checkpoints.",
    )
    return parser.parse_args()


def _validate_completed_reconstruction(output_dir: Path, config: dict) -> tuple[Path, Path] | None:
    marker_path = output_dir / "reconstruction" / "completed_run.json"
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    run_dir = Path(marker["run_directory"])
    model_dir = Path(marker.get("model_directory", run_dir / "model"))
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    problems = []
    if not checkpoint.exists():
        problems.append(f"checkpoint missing: {checkpoint}")
    else:
        import torch

        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        actual_checkpoint_epoch = int(state.get("epoch", -1))
        if actual_checkpoint_epoch < int(config["legacy_reconstruction"]["epochs"]):
            problems.append(
                f"checkpoint itself is incomplete at epoch {actual_checkpoint_epoch}"
            )
        if actual_checkpoint_epoch != int(marker.get("checkpoint_epoch", -1)):
            problems.append("checkpoint epoch differs from the explicit completion marker")
    if int(marker.get("epochs", -1)) != int(config["legacy_reconstruction"]["epochs"]):
        problems.append("completion-marker epoch count differs from experiment configuration")
    if int(marker.get("checkpoint_epoch", -1)) < int(config["legacy_reconstruction"]["epochs"]):
        problems.append("completion marker reports an incomplete checkpoint epoch")
    expected_projection = (output_dir / "projections" / "projections_physical.npy").resolve()
    if Path(marker.get("projection_input", "")).resolve() != expected_projection:
        problems.append("completion marker references a different projection file")
    if problems:
        raise RuntimeError("Inconsistent reconstruction completion marker: " + "; ".join(problems))
    return run_dir, model_dir


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.num_angles <= 0:
        raise SystemExit("--epochs and --num-angles must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = porous_experiment_config(args.num_angles, args.epochs)
    config_path = output_dir / "experiment_config.json"
    if config_path.exists() and not args.force:
        config = read_json(config_path)
        if config != requested:
            raise SystemExit(
                f"Existing configuration differs from requested settings: {config_path}. "
                "Use the original arguments, choose another --output-dir, or deliberately use --force."
            )
    else:
        config = requested
        write_json(config_path, config)
    provenance_path = output_dir / "provenance.json"
    if not provenance_path.exists() or args.force:
        write_json(provenance_path, collect_provenance(REPOSITORY_ROOT))
    set_cpu_seeds(int(config["seed"]))
    stages = STAGES if args.stage == "all" else (args.stage,)
    print(f"Stage 2 output: {output_dir}", flush=True)
    print(f"Selected stages: {', '.join(stages)}", flush=True)

    phantom_path = output_dir / "ground_truth" / "phantom_mu_mm_inv.npy"
    if "phantom" in stages:
        required = [phantom_path, output_dir / "ground_truth" / "rock_mask.npy", output_dir / "ground_truth" / "pore_mask.npy", output_dir / "ground_truth" / "metadata.json"]
        if all(path.exists() for path in required) and not args.force:
            print("[phantom] Reusing completed PoreSpy phantom.", flush=True)
        else:
            print("[phantom] Generating deterministic PoreSpy geometry and calibrated volume...", flush=True)
            generate_porous_phantom(config, output_dir)
            print(f"[phantom] Saved {phantom_path}", flush=True)

    projection_path = output_dir / "projections" / "projections_physical.npy"
    if "projections" in stages:
        if not phantom_path.exists():
            raise SystemExit("Porous phantom is missing; run --stage phantom first.")
        projection_required = [projection_path, output_dir / "projections" / "angles_rad.npy", output_dir / "projections" / "validation.json"]
        if all(path.exists() for path in projection_required) and not args.force:
            print("[projections] Reusing validated physical TIGRE projections.", flush=True)
        else:
            print("[projections] Generating and validating independent TIGRE projections...", flush=True)
            volume, _, _ = load_porous_phantom(output_dir)
            generate_porous_projections(volume, config, output_dir)
            print(f"[projections] Saved {projection_path}", flush=True)

    if "reconstruct" in stages:
        if not projection_path.exists():
            raise SystemExit("Physical projections are missing; run --stage projections first.")
        completed = _validate_completed_reconstruction(output_dir, config)
        if completed is not None:
            run_dir, model_dir = completed
            print(f"[reconstruct] Reusing explicitly completed run: {model_dir}", flush=True)
        else:
            from experiments.attenuation_validation.reconstruction import (
                resume_legacy_reconstruction,
                run_legacy_reconstruction,
            )

            checkpoints = sorted(
                (output_dir / "reconstruction").glob("**/model/checkpoints/last.ckpt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if checkpoints:
                model_dir = checkpoints[0].parent.parent
                print(f"[reconstruct] Resuming incomplete run from {model_dir}", flush=True)
                run_dir = resume_legacy_reconstruction(config, output_dir, model_dir)
            else:
                print(f"[reconstruct] Starting {args.epochs}-epoch legacy static HashGrid run...", flush=True)
                run_dir = run_legacy_reconstruction(config, output_dir, epochs=args.epochs)
                model_dir = run_dir / "model"
            _validate_completed_reconstruction(output_dir, config)
            print(f"[reconstruct] Explicit completion marker verified for {run_dir}", flush=True)

    if "evaluate" in stages:
        completed = _validate_completed_reconstruction(output_dir, config)
        if completed is None:
            raise SystemExit("Evaluation will not reconstruct: explicit reconstruction completion marker is missing.")
        _, model_dir = completed
        evaluation_marker = output_dir / "evaluation" / "completed.json"
        if evaluation_marker.exists() and not args.force:
            print("[evaluate] Reusing completed quantitative evaluation.", flush=True)
        else:
            from experiments.attenuation_validation.porous_evaluation import evaluate_porous

            print(f"[evaluate] Evaluating checkpoint {model_dir / 'checkpoints' / 'last.ckpt'}", flush=True)
            metrics = evaluate_porous(output_dir, model_dir)
            write_json(
                evaluation_marker,
                {
                    "model_directory": str(model_dir),
                    "metrics_file": str(output_dir / "evaluation" / "metrics.json"),
                    "rock_mean_mm_inv": metrics["rock"]["reconstructed_mean_mm_inv"],
                    "pore_mean_mm_inv": metrics["pore"]["reconstructed_mean_mm_inv"],
                },
            )
            print(
                f"[evaluate] rock mean={metrics['rock']['reconstructed_mean_mm_inv']:.6g}, "
                f"pore mean={metrics['pore']['reconstructed_mean_mm_inv']:.6g} mm^-1",
                flush=True,
            )
    print("Stage 2 selected stages completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
