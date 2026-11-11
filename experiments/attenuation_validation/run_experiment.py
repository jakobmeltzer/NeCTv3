#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python experiments/attenuation_validation/run_experiment.py` from a
# clean checkout without installing the experiments as a package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.attenuation_validation.common import (  # noqa: E402
    experiment_config,
    read_json,
    set_cpu_seeds,
    write_json,
)
from experiments.attenuation_validation.phantom import generate_phantom, load_phantom  # noqa: E402
from experiments.attenuation_validation.projections import generate_projections  # noqa: E402


STAGES = ("phantom", "projections", "reconstruct", "evaluate", "diagnose-export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 of the NeCT attenuation-validation experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "attenuation_validation" / "outputs" / "stage1_cube",
        help="Experiment output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
        help="Run all stages or one stage only.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Legacy NeCT epochs.")
    parser.add_argument("--num-angles", type=int, default=180, help="Number of cone-beam views.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate completed lightweight stages. Does not delete a reconstruction checkpoint.",
    )
    parser.add_argument(
        "--reuse-export",
        action="store_true",
        help="With --stage diagnose-export, reuse an existing TIFF instead of rerunning the original exporter.",
    )
    return parser.parse_args()


def selected_stages(stage: str) -> tuple[str, ...]:
    # Diagnostics are intentionally opt-in: `all` retains the original experiment
    # workflow and never adds another checkpoint evaluation unexpectedly.
    return STAGES[:-1] if stage == "all" else (stage,)


def run_export_diagnostic_only(args: argparse.Namespace, output_dir: Path) -> int:
    """Run a checkpoint-only diagnostic with no reconstruction fallback."""
    completed_path = output_dir / "reconstruction" / "completed_run.json"
    if not completed_path.exists():
        raise SystemExit(
            "Export diagnostic will not launch reconstruction. Missing completion marker: "
            f"{completed_path}"
        )
    completed = read_json(completed_path)
    run_dir = Path(completed["run_directory"])
    model_dir = Path(completed.get("model_directory", run_dir / "model"))
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.exists():
        raise SystemExit(
            "Export diagnostic will not launch reconstruction. Missing checkpoint: "
            f"{checkpoint}"
        )
    from experiments.attenuation_validation.export_diagnostic import diagnose_export

    print(f"Experiment output: {output_dir}", flush=True)
    print(f"[diagnose-export] Using existing checkpoint: {checkpoint}", flush=True)
    metrics = diagnose_export(
        output_dir, model_dir, refresh_export=not args.reuse_export
    )
    comparisons = metrics["comparisons"]
    residual = comparisons["B_legacy_formula_vs_C_exporter"]
    print(
        "[diagnose-export] endpoint legacy formula vs exporter: "
        f"MAE={residual['mean_absolute_mm_inv']:.6g}, "
        f"max={residual['maximum_absolute_mm_inv']:.6g} mm^-1",
        flush=True,
    )
    print(f"Results: {output_dir / 'export_diagnostic'}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.num_angles <= 0:
        raise SystemExit("--epochs and --num-angles must be positive")

    output_dir = args.output_dir.resolve()
    if args.stage == "diagnose-export":
        return run_export_diagnostic_only(args, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    requested_config = experiment_config(args.num_angles, args.epochs)
    if config_path.exists() and not args.force:
        config = read_json(config_path)
        if config != requested_config:
            raise SystemExit(
                f"Existing configuration at {config_path} differs from requested settings. "
                "Use a new --output-dir, or --force to replace lightweight inputs."
            )
    else:
        config = requested_config
        write_json(config_path, config)
    set_cpu_seeds(config["seed"])

    stages = selected_stages(args.stage)
    print(f"Experiment output: {output_dir}", flush=True)
    print(f"Selected stages: {', '.join(stages)}", flush=True)

    phantom_path = output_dir / "ground_truth" / "phantom_mu_mm_inv.npy"
    if "phantom" in stages:
        if phantom_path.exists() and not args.force:
            print("[phantom] Reusing existing calibrated phantom.", flush=True)
        else:
            print("[phantom] Generating 64^3 calibrated cube phantom...", flush=True)
            generate_phantom(config, output_dir)
            print(f"[phantom] Saved {phantom_path}", flush=True)

    projection_path = output_dir / "projections" / "projections_physical.npy"
    if "projections" in stages:
        if not phantom_path.exists():
            raise SystemExit("Ground-truth phantom is missing; run --stage phantom first.")
        if projection_path.exists() and not args.force:
            print("[projections] Reusing existing validated TIGRE projections.", flush=True)
        else:
            print("[projections] Running independent TIGRE cone-beam projection...", flush=True)
            volume, _ = load_phantom(output_dir)
            generate_projections(volume, config, output_dir)
            print(f"[projections] Saved and validated {projection_path}", flush=True)

    completed_run_path = output_dir / "reconstruction" / "completed_run.json"
    if "reconstruct" in stages:
        if not projection_path.exists():
            raise SystemExit("Physical projections are missing; run --stage projections first.")
        if completed_run_path.exists():
            completed = read_json(completed_run_path)
            run_dir = Path(completed["run_directory"])
            model_dir = Path(completed.get("model_directory", run_dir / "model"))
            checkpoint = model_dir / "checkpoints" / "last.ckpt"
            if checkpoint.exists():
                print(f"[reconstruct] Reusing completed checkpoint: {checkpoint}", flush=True)
            else:
                raise SystemExit(
                    f"Completion marker exists but checkpoint is missing: {checkpoint}. "
                    "Remove completed_run.json and rerun reconstruction."
                )
        else:
            from experiments.attenuation_validation.reconstruction import (
                resume_legacy_reconstruction,
                run_legacy_reconstruction,
            )

            candidates = sorted(
                (output_dir / "reconstruction").glob("**/model/checkpoints/last.ckpt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                model_dir = candidates[0].parent.parent
                print(f"[reconstruct] Resuming checkpoint under {model_dir}...", flush=True)
                run_dir = resume_legacy_reconstruction(config, output_dir, model_dir)
            else:
                print("[reconstruct] Starting legacy NeCT baseline on the GPU...", flush=True)
                run_dir = run_legacy_reconstruction(config, output_dir, epochs=args.epochs)
            print(f"[reconstruct] Completed run: {run_dir}", flush=True)

    if "evaluate" in stages:
        if not completed_run_path.exists():
            raise SystemExit("Completed NeCT run is missing; run --stage reconstruct first.")
        completed = read_json(completed_run_path)
        run_dir = Path(completed["run_directory"])
        model_dir = Path(completed.get("model_directory", run_dir / "model"))
        checkpoint = model_dir / "checkpoints" / "last.ckpt"
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint is missing: {checkpoint}")
        print("[evaluate] Evaluating raw INR and legacy exporter in physical units...", flush=True)
        from experiments.attenuation_validation.evaluation import evaluate

        metrics = evaluate(output_dir, model_dir)
        cube = metrics["direct_inr_legacy_converted"]
        print(
            "[evaluate] cube mean="
            f"{cube['cube_mean_mm_inv']:.6g} mm^-1; "
            f"MAE={cube['whole_volume_mae_mm_inv']:.6g} mm^-1",
            flush=True,
        )

    print("Experiment stage(s) completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

