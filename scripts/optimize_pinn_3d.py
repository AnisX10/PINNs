from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a staged optimization campaign for the 3D PINN.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_3d_case_matrix_case001.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stage1-output",
        default="outputs_3d_stage1_opt",
        help="Output directory for the first training stage.",
    )
    parser.add_argument(
        "--stage2-output",
        default="outputs_3d_stage2_refine",
        help="Output directory for the continuation stage.",
    )
    parser.add_argument(
        "--best-output",
        default="outputs_3d_best",
        help="Output directory where the best stage is copied.",
    )
    parser.add_argument("--stage1-adam-epochs", type=int, default=40, help="Adam epochs for stage 1.")
    parser.add_argument("--stage1-lbfgs-steps", type=int, default=8, help="L-BFGS steps for stage 1.")
    parser.add_argument("--stage2-adam-epochs", type=int, default=40, help="Adam epochs for stage 2.")
    parser.add_argument("--stage2-lbfgs-steps", type=int, default=4, help="L-BFGS steps for stage 2.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted notation, e.g. training_3d.seed=7",
    )
    return parser


def _run_train(command_args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "train_pinn_3d.py"), *command_args]
    subprocess.run(command, cwd=ROOT, check=True)


def _metrics(output_dir: Path) -> dict:
    with (output_dir / "training_metrics_3d.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = build_parser().parse_args()
    stage1_dir = ROOT / args.stage1_output
    stage2_dir = ROOT / args.stage2_output
    best_dir = ROOT / args.best_output
    overrides = [item for item in args.set]

    _run_train(
        [
            "--config",
            args.config,
            "--adam-epochs",
            str(args.stage1_adam_epochs),
            "--set",
            f"training_3d.lbfgs_steps={args.stage1_lbfgs_steps}",
            "--set",
            f"paths.output_dir={args.stage1_output}",
            *[entry for item in overrides for entry in ["--set", item]],
        ]
    )

    stage1_checkpoint = stage1_dir / "checkpoints" / "best_model_3d.pt"
    _run_train(
        [
            "--config",
            args.config,
            "--resume-checkpoint",
            str(stage1_checkpoint),
            "--adam-epochs",
            str(args.stage2_adam_epochs),
            "--set",
            f"training_3d.lbfgs_steps={args.stage2_lbfgs_steps}",
            "--set",
            f"paths.output_dir={args.stage2_output}",
            *[entry for item in overrides for entry in ["--set", item]],
        ]
    )

    stage1_metrics = _metrics(stage1_dir)
    stage2_metrics = _metrics(stage2_dir)
    leaderboard = {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
    }

    best_stage_dir = stage1_dir
    best_stage_name = "stage1"
    if float(stage2_metrics["best_validation_score"]) < float(stage1_metrics["best_validation_score"]):
        best_stage_dir = stage2_dir
        best_stage_name = "stage2"

    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best_stage_dir, best_dir)

    summary = {
        "best_stage": best_stage_name,
        "best_validation_score": leaderboard[best_stage_name]["best_validation_score"],
        "stage1_output": str(stage1_dir),
        "stage2_output": str(stage2_dir),
        "best_output": str(best_dir),
        "leaderboard": leaderboard,
    }
    with (best_dir / "optimization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Best stage: {best_stage_name}")
    print(f"Best output: {best_dir.resolve()}")
    print(f"Best validation score: {summary['best_validation_score']:.6f}")


if __name__ == "__main__":
    main()
