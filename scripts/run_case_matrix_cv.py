from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "double_pipe_3d_case_matrix_conditioned_matrix.yaml"
ALL_CASE_IDS = [
    "case_001",
    "case_002",
    "case_003",
    "case_004",
    "case_005",
    "case_006",
    "case_007",
    "case_008",
    "case_009",
    "case_010",
    "case_011",
    "case_012",
    "case_013",
    "case_014",
    "case_015",
    "case_016",
]
FOLD_VALIDATION_CASE_IDS = {
    1: ["case_001", "case_010", "case_011", "case_013"],
    2: ["case_002", "case_004", "case_006", "case_015"],
    3: ["case_003", "case_007", "case_009", "case_016"],
    4: ["case_005", "case_008", "case_012", "case_014"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 4-fold unseen-case cross-validation for the case matrix 3D PINN.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=sorted(FOLD_VALIDATION_CASE_IDS),
        help="Fold ids to run. Defaults to all folds.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_3d_case_matrix_conditioned_case_cv",
        help="Directory where fold outputs and aggregate summaries will be stored.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse fold outputs when training_metrics_3d.json already exists.",
    )
    parser.add_argument(
        "--freeze-k-wall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze wall conductivity during CV to focus on surrogate generalization.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Additional config override using dotted notation, e.g. training_3d.adam_epochs=20",
    )
    return parser


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fold_train_case_ids(validation_case_ids: list[str]) -> list[str]:
    validation_set = set(validation_case_ids)
    return [case_id for case_id in ALL_CASE_IDS if case_id not in validation_set]


def _relative_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _run_fold(
    config_path: str,
    fold_id: int,
    output_dir: Path,
    reuse_existing: bool,
    freeze_k_wall: bool,
    extra_overrides: list[str],
) -> dict:
    metrics_path = output_dir / "training_metrics_3d.json"
    if reuse_existing and metrics_path.exists():
        print(f"fold={fold_id} reusing existing metrics at {metrics_path}")
        return _load_json(metrics_path)

    validation_case_ids = FOLD_VALIDATION_CASE_IDS[fold_id]
    train_case_ids = _fold_train_case_ids(validation_case_ids)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_pinn_3d.py"),
        "--config",
        config_path,
        "--set",
        f"case_matrix_3d.train_case_ids={json.dumps(train_case_ids)}",
        "--set",
        f"case_matrix_3d.validation_case_ids={json.dumps(validation_case_ids)}",
        "--set",
        f"paths.output_dir={_relative_output_path(output_dir)}",
    ]
    if freeze_k_wall:
        command.extend(["--set", "model_3d.learn_wall_conductivity=false"])
    for override in extra_overrides:
        command.extend(["--set", override])

    print(f"fold={fold_id} train_cases={train_case_ids}")
    print(f"fold={fold_id} validation_cases={validation_case_ids}")
    subprocess.run(command, cwd=ROOT, check=True)
    return _load_json(metrics_path)


def _aggregate_case_rows(fold_rows: list[dict]) -> tuple[list[dict], dict]:
    case_rows: list[dict] = []
    for fold_row in fold_rows:
        metrics = fold_row["metrics"]
        for case_id in metrics["validation_case_ids"]:
            hot_rmse = float(metrics["validation_metrics"]["hot"]["by_case"][case_id]["rmse_K"])
            cold_rmse = float(metrics["validation_metrics"]["cold"]["by_case"][case_id]["rmse_K"])
            case_rows.append(
                {
                    "fold": int(fold_row["fold"]),
                    "case_id": str(case_id),
                    "hot_rmse_K": hot_rmse,
                    "cold_rmse_K": cold_rmse,
                    "combined_rmse_K": hot_rmse + cold_rmse,
                    "output_dir": str(fold_row["output_dir"]),
                }
            )

    case_rows.sort(key=lambda row: row["case_id"])
    combined_scores = [float(row["combined_rmse_K"]) for row in case_rows]
    hot_scores = [float(row["hot_rmse_K"]) for row in case_rows]
    cold_scores = [float(row["cold_rmse_K"]) for row in case_rows]
    best_case = min(case_rows, key=lambda row: float(row["combined_rmse_K"]))
    worst_case = max(case_rows, key=lambda row: float(row["combined_rmse_K"]))
    aggregate = {
        "case_count": len(case_rows),
        "mean_hot_rmse_K": float(statistics.fmean(hot_scores)),
        "mean_cold_rmse_K": float(statistics.fmean(cold_scores)),
        "mean_combined_rmse_K": float(statistics.fmean(combined_scores)),
        "median_combined_rmse_K": float(statistics.median(combined_scores)),
        "best_case_id": str(best_case["case_id"]),
        "best_case_combined_rmse_K": float(best_case["combined_rmse_K"]),
        "worst_case_id": str(worst_case["case_id"]),
        "worst_case_combined_rmse_K": float(worst_case["combined_rmse_K"]),
    }
    return case_rows, aggregate


def main() -> None:
    args = build_parser().parse_args()
    invalid_folds = [fold_id for fold_id in args.folds if fold_id not in FOLD_VALIDATION_CASE_IDS]
    if invalid_folds:
        raise ValueError(f"Unknown fold ids: {invalid_folds}")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    for fold_id in args.folds:
        output_dir = output_root / f"fold_{fold_id}"
        metrics = _run_fold(
            config_path=args.config,
            fold_id=fold_id,
            output_dir=output_dir,
            reuse_existing=bool(args.reuse_existing),
            freeze_k_wall=bool(args.freeze_k_wall),
            extra_overrides=list(args.set),
        )
        fold_rows.append(
            {
                "fold": fold_id,
                "train_case_ids": metrics["train_case_ids"],
                "validation_case_ids": metrics["validation_case_ids"],
                "train_hot_rmse_K": float(metrics["train_metrics"]["hot"]["rmse_K"]),
                "train_cold_rmse_K": float(metrics["train_metrics"]["cold"]["rmse_K"]),
                "validation_hot_rmse_K": float(metrics["validation_metrics"]["hot"]["rmse_K"]),
                "validation_cold_rmse_K": float(metrics["validation_metrics"]["cold"]["rmse_K"]),
                "best_validation_score": float(metrics["best_validation_score"]),
                "final_k_wall_w_mk": float(metrics["final_k_wall_w_mk"]),
                "output_dir": _relative_output_path(output_dir),
                "metrics": metrics,
            }
        )

    case_rows, aggregate = _aggregate_case_rows(fold_rows)
    fold_summary_rows = []
    for row in fold_rows:
        fold_summary_rows.append(
            {
                "fold": int(row["fold"]),
                "train_case_ids": ",".join(str(case_id) for case_id in row["train_case_ids"]),
                "validation_case_ids": ",".join(str(case_id) for case_id in row["validation_case_ids"]),
                "train_hot_rmse_K": float(row["train_hot_rmse_K"]),
                "train_cold_rmse_K": float(row["train_cold_rmse_K"]),
                "validation_hot_rmse_K": float(row["validation_hot_rmse_K"]),
                "validation_cold_rmse_K": float(row["validation_cold_rmse_K"]),
                "best_validation_score": float(row["best_validation_score"]),
                "final_k_wall_w_mk": float(row["final_k_wall_w_mk"]),
                "output_dir": str(row["output_dir"]),
            }
        )

    summary_payload = {
        "config": args.config,
        "folds": {
            str(fold_id): {
                "validation_case_ids": FOLD_VALIDATION_CASE_IDS[fold_id],
                "train_case_ids": _fold_train_case_ids(FOLD_VALIDATION_CASE_IDS[fold_id]),
            }
            for fold_id in args.folds
        },
        "reuse_existing": bool(args.reuse_existing),
        "freeze_k_wall": bool(args.freeze_k_wall),
        "overrides": list(args.set),
        "aggregate": aggregate,
        "fold_results": fold_summary_rows,
        "case_results": case_rows,
    }
    with (output_root / "case_cv_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    with (output_root / "case_cv_fold_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_summary_rows)

    with (output_root / "case_cv_case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    print(f"Mean unseen-case combined RMSE: {aggregate['mean_combined_rmse_K']:.6f} K")
    print(f"Best case: {aggregate['best_case_id']} ({aggregate['best_case_combined_rmse_K']:.6f} K)")
    print(f"Worst case: {aggregate['worst_case_id']} ({aggregate['worst_case_combined_rmse_K']:.6f} K)")
    print(f"Summary saved to: {(output_root / 'case_cv_summary.json').resolve()}")


if __name__ == "__main__":
    main()
