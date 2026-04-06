from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-seed optimization sweep for the 3D PINN.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_3d_case_matrix_case001.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 13, 21, 42, 84],
        help="List of seeds to evaluate.",
    )
    parser.add_argument("--stage1-adam-epochs", type=int, default=40)
    parser.add_argument("--stage1-lbfgs-steps", type=int, default=8)
    parser.add_argument("--stage2-adam-epochs", type=int, default=40)
    parser.add_argument("--stage2-lbfgs-steps", type=int, default=4)
    parser.add_argument(
        "--output-root",
        default="outputs_3d_seed_sweep",
        help="Directory where sweep summaries and per-seed results will be stored.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Additional config override using dotted notation, e.g. training_3d.data_weight=250.0",
    )
    return parser


def _run_optimize(command_args: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "optimize_pinn_3d.py"), *command_args]
    subprocess.run(command, cwd=ROOT, check=True)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = build_parser().parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    leaderboard: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        seed_prefix = f"seed_{seed}"
        stage1_output = output_root / f"{seed_prefix}_stage1"
        stage2_output = output_root / f"{seed_prefix}_stage2"
        best_output = output_root / f"{seed_prefix}_best"
        optimize_args = [
            "--config",
            args.config,
            "--stage1-adam-epochs",
            str(args.stage1_adam_epochs),
            "--stage1-lbfgs-steps",
            str(args.stage1_lbfgs_steps),
            "--stage2-adam-epochs",
            str(args.stage2_adam_epochs),
            "--stage2-lbfgs-steps",
            str(args.stage2_lbfgs_steps),
            "--stage1-output",
            str(stage1_output.relative_to(ROOT)),
            "--stage2-output",
            str(stage2_output.relative_to(ROOT)),
            "--best-output",
            str(best_output.relative_to(ROOT)),
            "--set",
            f"training_3d.seed={seed}",
        ]
        for override in args.set:
            optimize_args.extend(["--set", override])
        _run_optimize(optimize_args)

        metrics = _load_json(best_output / "training_metrics_3d.json")
        summary = _load_json(best_output / "optimization_summary.json")
        row = {
            "seed": seed,
            "best_stage": summary["best_stage"],
            "best_validation_score": float(metrics["best_validation_score"]),
            "hot_validation_rmse_K": float(metrics["validation_metrics"]["hot"]["rmse_K"]),
            "cold_validation_rmse_K": float(metrics["validation_metrics"]["cold"]["rmse_K"]),
            "final_hot_inlet_temperature_K": float(metrics["final_hot_inlet_temperature_K"]),
            "final_cold_inlet_temperature_K": float(metrics["final_cold_inlet_temperature_K"]),
            "final_k_wall_w_mk": float(metrics["final_k_wall_w_mk"]),
            "output_dir": str(best_output),
        }
        leaderboard.append(row)

    leaderboard.sort(key=lambda item: float(item["best_validation_score"]))
    summary_payload = {
        "config": args.config,
        "seeds": args.seeds,
        "stage1_adam_epochs": args.stage1_adam_epochs,
        "stage1_lbfgs_steps": args.stage1_lbfgs_steps,
        "stage2_adam_epochs": args.stage2_adam_epochs,
        "stage2_lbfgs_steps": args.stage2_lbfgs_steps,
        "overrides": list(args.set),
        "leaderboard": leaderboard,
    }
    with (output_root / "seed_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    with (output_root / "seed_sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leaderboard[0].keys()))
        writer.writeheader()
        writer.writerows(leaderboard)

    best = leaderboard[0]
    print(f"Best seed: {best['seed']}")
    print(f"Best validation score: {best['best_validation_score']:.6f}")
    print(f"Best output: {best['output_dir']}")


if __name__ == "__main__":
    main()
