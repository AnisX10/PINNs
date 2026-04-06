from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import apply_overrides, load_config
from pinn_hex.data.synthetic import OPERATING_COLUMNS, resolve_synthetic_3d_config, synthetic_case_ids_from_config
from pinn_hex.data.threed import build_3d_case_artifacts
from pinn_hex.postprocess.temperature_calibration import apply_temperature_calibration, load_temperature_calibration
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import geometry3d_from_config
from pinn_hex.training.trainer_3d import PINNTrainer3D


DEFAULT_CONFIG = ROOT / "configs" / "double_pipe_3d_synthetic_conditioned_final.yaml"
DEFAULT_VALIDATION_CASE_IDS = ["case_003", "case_007", "case_009", "case_016"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the locked final 3D PINN on a reserved synthetic holdout split."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--validation-case-ids",
        nargs="+",
        default=list(DEFAULT_VALIDATION_CASE_IDS),
        help="Held-out synthetic case ids used for final validation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_3d_synthetic_final_validation",
        help="Directory for training and validation artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. If set, validation runs without retraining.",
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse output-dir checkpoint when available.",
    )
    parser.add_argument(
        "--freeze-k-wall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze wall conductivity during final validation.",
    )
    parser.add_argument(
        "--adam-epochs",
        type=int,
        default=None,
        help="Optional Adam epoch override when training a new holdout model.",
    )
    parser.add_argument(
        "--temperature-calibration-json",
        default=None,
        help="Optional boundary-temperature calibration JSON applied after prediction.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Additional config override using dotted notation, e.g. training_3d.adam_epochs=20",
    )
    return parser


def _case_split(config: dict, validation_case_ids: list[str]) -> tuple[list[str], list[str]]:
    selected_case_ids = [str(case_id) for case_id in synthetic_case_ids_from_config(config)]
    if not selected_case_ids:
        raise ValueError("Final validation requires synthetic_3d.case_ids or related synthetic case configuration.")
    selected_set = set(selected_case_ids)
    validation = [str(case_id) for case_id in validation_case_ids]
    unknown = sorted(set(validation) - selected_set)
    if unknown:
        raise ValueError(f"Unknown validation case ids: {unknown}")
    validation_set = set(validation)
    train = [case_id for case_id in selected_case_ids if case_id not in validation_set]
    if not train:
        raise ValueError("Validation split left no training cases.")
    return train, validation


def _tensor_from_frame(frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.tensor(frame[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    operating = torch.tensor(
        frame[list(OPERATING_COLUMNS)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    return xyz, operating


def _predict_stream_frame(model: torch.nn.Module, frame: pd.DataFrame, stream: str, device: torch.device) -> pd.DataFrame:
    xyz, operating = _tensor_from_frame(frame, device)
    with torch.no_grad():
        state = model.hot(xyz, operating) if stream == "hot" else model.cold(xyz, operating)
    result = frame.copy()
    result["u_pred"] = state.u.detach().cpu().numpy().reshape(-1)
    result["v_pred"] = state.v.detach().cpu().numpy().reshape(-1)
    result["w_pred"] = state.w.detach().cpu().numpy().reshape(-1)
    result["p_pred"] = state.p.detach().cpu().numpy().reshape(-1)
    result["T_pred"] = state.T.detach().cpu().numpy().reshape(-1)
    result["stream"] = stream
    return result


def _attach_wall_predictions(model: torch.nn.Module, frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    result = frame.copy()
    result["T_wall_pred"] = np.nan
    interface_boundaries = {"hot": "hot_wall", "cold": "cold_inner_wall"}
    boundary_name = interface_boundaries[str(frame["stream"].iloc[0])]
    mask = result["boundary"] == boundary_name
    if not np.any(mask.to_numpy()):
        return result
    xyz, operating = _tensor_from_frame(result.loc[mask], device)
    with torch.no_grad():
        wall_state = model.wall(xyz, operating)
    result.loc[mask, "T_wall_pred"] = wall_state.T.detach().cpu().numpy().reshape(-1)
    return result


def _flux_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights_abs = np.abs(weights.astype(np.float64))
    values64 = values.astype(np.float64)
    total_weight = float(weights_abs.sum())
    if total_weight <= 1.0e-12:
        return float(values64.mean())
    return float(np.sum(weights_abs * values64) / total_weight)


def _boundary_mean(frame: pd.DataFrame, boundary: str, column: str, weight_column: str | None = None) -> float:
    subset = frame[frame["boundary"] == boundary]
    if subset.empty:
        raise ValueError(f"Boundary '{boundary}' is missing from validation data.")
    values = subset[column].to_numpy(dtype=np.float64)
    if weight_column is None:
        return float(values.mean())
    return _flux_weighted_mean(values, subset[weight_column].to_numpy(dtype=np.float64))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _safe_relative_error(value: float, reference: float) -> float:
    scale = max(abs(reference), 1.0e-8)
    return float(abs(value - reference) / scale * 100.0)


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if np.allclose(a.std(ddof=0), 0.0) or np.allclose(b.std(ddof=0), 0.0):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _interface_rmse(frame: pd.DataFrame) -> float | None:
    subset = frame.dropna(subset=["T_wall_pred"])
    if subset.empty:
        return None
    return _rmse(
        actual=subset["T_wall_pred"].to_numpy(dtype=np.float64),
        predicted=subset["T_pred"].to_numpy(dtype=np.float64),
    )


def _stream_metric_summary(frame: pd.DataFrame) -> dict[str, object]:
    metrics = {
        "rmse_K": _rmse(frame["T"].to_numpy(dtype=np.float64), frame["T_pred"].to_numpy(dtype=np.float64)),
        "mae_K": float(np.mean(np.abs(frame["T_pred"].to_numpy(dtype=np.float64) - frame["T"].to_numpy(dtype=np.float64)))),
    }
    by_case: dict[str, dict[str, float]] = {}
    for case_id, group in frame.groupby("case_id", sort=False):
        by_case[str(case_id)] = {
            "rmse_K": _rmse(group["T"].to_numpy(dtype=np.float64), group["T_pred"].to_numpy(dtype=np.float64)),
            "mae_K": float(
                np.mean(np.abs(group["T_pred"].to_numpy(dtype=np.float64) - group["T"].to_numpy(dtype=np.float64)))
            ),
            "mean_target_K": float(group["T"].mean()),
            "mean_pred_K": float(group["T_pred"].mean()),
        }
    metrics["by_case"] = by_case
    return metrics


def _case_summary_row(
    case_id: str,
    hot_case: pd.DataFrame,
    cold_case: pd.DataFrame,
    globals_row: dict[str, float | str],
    cp_j_kgk: float,
) -> dict[str, float | str]:
    hot_outlet_bulk_pred = _boundary_mean(hot_case, "hot_outlet", "T_pred", weight_column="w_pred")
    cold_outlet_bulk_pred = _boundary_mean(cold_case, "cold_outlet", "T_pred", weight_column="w_pred")
    hot_inlet_pressure_pred = _boundary_mean(hot_case, "hot_inlet", "p_pred")
    hot_outlet_pressure_pred = _boundary_mean(hot_case, "hot_outlet", "p_pred")
    cold_inlet_pressure_pred = _boundary_mean(cold_case, "cold_inlet", "p_pred")
    cold_outlet_pressure_pred = _boundary_mean(cold_case, "cold_outlet", "p_pred")
    hot_dp_pred = hot_inlet_pressure_pred - hot_outlet_pressure_pred
    cold_dp_pred = cold_inlet_pressure_pred - cold_outlet_pressure_pred

    hot_rmse = _rmse(hot_case["T"].to_numpy(dtype=np.float64), hot_case["T_pred"].to_numpy(dtype=np.float64))
    cold_rmse = _rmse(cold_case["T"].to_numpy(dtype=np.float64), cold_case["T_pred"].to_numpy(dtype=np.float64))

    reference_hot_in = float(globals_row.get("Th_in_bulk", globals_row["Th_in_K"]))
    reference_cold_in = float(globals_row.get("Tc_in_bulk", globals_row["Tc_in_K"]))
    reference_hot_out = float(globals_row["Th_out_bulk"])
    reference_cold_out = float(globals_row["Tc_out_bulk"])
    reference_q_total = float(globals_row["Q_total"])
    reference_hot_dp = float(globals_row["dp_hot"])
    reference_cold_dp = float(globals_row["dp_cold"])
    m_dot_hot = float(globals_row["m_dot_hot"])
    m_dot_cold = float(globals_row["m_dot_cold"])

    q_hot_pred = m_dot_hot * cp_j_kgk * (reference_hot_in - hot_outlet_bulk_pred)
    q_cold_pred = m_dot_cold * cp_j_kgk * (cold_outlet_bulk_pred - reference_cold_in)
    q_mean_pred = 0.5 * (q_hot_pred + q_cold_pred)
    energy_balance_gap_w = q_hot_pred - q_cold_pred
    energy_balance_scale = max(abs(reference_q_total), abs(q_hot_pred), abs(q_cold_pred), 1.0e-8)

    hot_outlet_rmse = _rmse(
        hot_case.loc[hot_case["boundary"] == "hot_outlet", "T"].to_numpy(dtype=np.float64),
        hot_case.loc[hot_case["boundary"] == "hot_outlet", "T_pred"].to_numpy(dtype=np.float64),
    )
    cold_outlet_rmse = _rmse(
        cold_case.loc[cold_case["boundary"] == "cold_outlet", "T"].to_numpy(dtype=np.float64),
        cold_case.loc[cold_case["boundary"] == "cold_outlet", "T_pred"].to_numpy(dtype=np.float64),
    )

    hot_interface_rmse = _interface_rmse(hot_case)
    cold_interface_rmse = _interface_rmse(cold_case)

    return {
        "case_id": case_id,
        "hot_rmse_K": hot_rmse,
        "cold_rmse_K": cold_rmse,
        "combined_rmse_K": hot_rmse + cold_rmse,
        "hot_outlet_rmse_K": hot_outlet_rmse,
        "cold_outlet_rmse_K": cold_outlet_rmse,
        "predicted_hot_outlet_bulk_K": hot_outlet_bulk_pred,
        "reference_hot_outlet_bulk_K": reference_hot_out,
        "hot_outlet_bulk_abs_error_K": float(abs(hot_outlet_bulk_pred - reference_hot_out)),
        "predicted_cold_outlet_bulk_K": cold_outlet_bulk_pred,
        "reference_cold_outlet_bulk_K": reference_cold_out,
        "cold_outlet_bulk_abs_error_K": float(abs(cold_outlet_bulk_pred - reference_cold_out)),
        "predicted_hot_dp_Pa": hot_dp_pred,
        "reference_hot_dp_Pa": reference_hot_dp,
        "hot_dp_abs_error_Pa": float(abs(hot_dp_pred - reference_hot_dp)),
        "hot_dp_rel_error_pct": _safe_relative_error(hot_dp_pred, reference_hot_dp),
        "predicted_cold_dp_Pa": cold_dp_pred,
        "reference_cold_dp_Pa": reference_cold_dp,
        "cold_dp_abs_error_Pa": float(abs(cold_dp_pred - reference_cold_dp)),
        "cold_dp_rel_error_pct": _safe_relative_error(cold_dp_pred, reference_cold_dp),
        "predicted_Q_hot_W": q_hot_pred,
        "predicted_Q_cold_W": q_cold_pred,
        "predicted_Q_mean_W": q_mean_pred,
        "reference_Q_total_W": reference_q_total,
        "Q_total_abs_error_W": float(abs(q_mean_pred - reference_q_total)),
        "Q_total_rel_error_pct": _safe_relative_error(q_mean_pred, reference_q_total),
        "energy_balance_gap_W": energy_balance_gap_w,
        "energy_balance_gap_pct": float(abs(energy_balance_gap_w) / energy_balance_scale * 100.0),
        "hot_interface_temp_rmse_K": hot_interface_rmse,
        "cold_interface_temp_rmse_K": cold_interface_rmse,
        "hot_cooling_positive": bool(hot_outlet_bulk_pred < reference_hot_in),
        "cold_heating_positive": bool(cold_outlet_bulk_pred > reference_cold_in),
        "reference_effectiveness": float(globals_row["effectiveness"]),
    }


def _aggregate_case_rows(case_rows: list[dict[str, float | str | bool | None]]) -> dict[str, float | str | None]:
    combined_scores = [float(row["combined_rmse_K"]) for row in case_rows]
    hot_scores = [float(row["hot_rmse_K"]) for row in case_rows]
    cold_scores = [float(row["cold_rmse_K"]) for row in case_rows]
    hot_outlet_abs = [float(row["hot_outlet_bulk_abs_error_K"]) for row in case_rows]
    cold_outlet_abs = [float(row["cold_outlet_bulk_abs_error_K"]) for row in case_rows]
    q_errors = [float(row["Q_total_rel_error_pct"]) for row in case_rows]
    energy_gaps = [float(row["energy_balance_gap_pct"]) for row in case_rows]
    hot_dp_errors = [float(row["hot_dp_rel_error_pct"]) for row in case_rows]
    cold_dp_errors = [float(row["cold_dp_rel_error_pct"]) for row in case_rows]
    hot_interface = [
        float(row["hot_interface_temp_rmse_K"])
        for row in case_rows
        if row["hot_interface_temp_rmse_K"] is not None
    ]
    cold_interface = [
        float(row["cold_interface_temp_rmse_K"])
        for row in case_rows
        if row["cold_interface_temp_rmse_K"] is not None
    ]
    best_case = min(case_rows, key=lambda row: float(row["combined_rmse_K"]))
    worst_case = max(case_rows, key=lambda row: float(row["combined_rmse_K"]))

    return {
        "case_count": len(case_rows),
        "mean_case_hot_rmse_K": float(np.mean(hot_scores)),
        "mean_case_cold_rmse_K": float(np.mean(cold_scores)),
        "mean_case_combined_rmse_K": float(np.mean(combined_scores)),
        "mean_hot_outlet_bulk_abs_error_K": float(np.mean(hot_outlet_abs)),
        "mean_cold_outlet_bulk_abs_error_K": float(np.mean(cold_outlet_abs)),
        "mean_Q_total_rel_error_pct": float(np.mean(q_errors)),
        "mean_energy_balance_gap_pct": float(np.mean(energy_gaps)),
        "mean_hot_dp_rel_error_pct": float(np.mean(hot_dp_errors)),
        "mean_cold_dp_rel_error_pct": float(np.mean(cold_dp_errors)),
        "mean_hot_interface_temp_rmse_K": float(np.mean(hot_interface)) if hot_interface else None,
        "mean_cold_interface_temp_rmse_K": float(np.mean(cold_interface)) if cold_interface else None,
        "best_case_id": str(best_case["case_id"]),
        "best_case_combined_rmse_K": float(best_case["combined_rmse_K"]),
        "worst_case_id": str(worst_case["case_id"]),
        "worst_case_combined_rmse_K": float(worst_case["combined_rmse_K"]),
        "all_hot_cooling_positive": bool(all(bool(row["hot_cooling_positive"]) for row in case_rows)),
        "all_cold_heating_positive": bool(all(bool(row["cold_heating_positive"]) for row in case_rows)),
        "Q_total_correlation": _pearson(
            [float(row["predicted_Q_mean_W"]) for row in case_rows],
            [float(row["reference_Q_total_W"]) for row in case_rows],
        ),
        "hot_dp_correlation": _pearson(
            [float(row["predicted_hot_dp_Pa"]) for row in case_rows],
            [float(row["reference_hot_dp_Pa"]) for row in case_rows],
        ),
        "cold_dp_correlation": _pearson(
            [float(row["predicted_cold_dp_Pa"]) for row in case_rows],
            [float(row["reference_cold_dp_Pa"]) for row in case_rows],
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = apply_overrides(config, list(args.set))
    train_case_ids, validation_case_ids = _case_split(config, list(args.validation_case_ids))
    config.setdefault("synthetic_3d", {})
    config["synthetic_3d"]["train_case_ids"] = list(train_case_ids)
    config["synthetic_3d"]["validation_case_ids"] = list(validation_case_ids)
    config.setdefault("paths", {})
    config["paths"]["output_dir"] = str(output_dir)
    if args.freeze_k_wall:
        config.setdefault("model_3d", {})
        config["model_3d"]["learn_wall_conductivity"] = False
    config = resolve_synthetic_3d_config(config)

    artifacts = build_3d_case_artifacts(config)
    geometry = geometry3d_from_config(config)
    operating = operating_point_from_config(config)
    trainer = PINNTrainer3D(
        config=config,
        geometry=geometry,
        operating=operating,
        hot_train=artifacts["hot_train"],
        cold_train=artifacts["cold_train"],
        hot_validation=artifacts["hot_validation"],
        cold_validation=artifacts["cold_validation"],
    )

    checkpoint_path: Path
    training_mode: str
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device))
        training_mode = "provided_checkpoint"
    else:
        checkpoint_path = output_dir / "checkpoints" / "best_model_3d.pt"
        if bool(args.reuse_existing) and checkpoint_path.exists():
            trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=trainer.device))
            training_mode = "reused_checkpoint"
        else:
            result = trainer.fit(output_dir=output_dir, adam_epochs_override=args.adam_epochs)
            checkpoint_path = result.checkpoint_path
            training_mode = "trained"
    trainer.model.eval()

    hot_validation_pred = _attach_wall_predictions(
        trainer.model,
        _predict_stream_frame(trainer.model, artifacts["hot_validation"], "hot", trainer.device),
        trainer.device,
    )
    cold_validation_pred = _attach_wall_predictions(
        trainer.model,
        _predict_stream_frame(trainer.model, artifacts["cold_validation"], "cold", trainer.device),
        trainer.device,
    )
    calibration_payload = None
    if args.temperature_calibration_json:
        calibration_payload = load_temperature_calibration(args.temperature_calibration_json)
        hot_validation_pred = apply_temperature_calibration(hot_validation_pred, calibration_payload, prediction_column="T_pred")
        cold_validation_pred = apply_temperature_calibration(cold_validation_pred, calibration_payload, prediction_column="T_pred")
    heldout_predictions = pd.concat([hot_validation_pred, cold_validation_pred], ignore_index=True)
    heldout_predictions_path = output_dir / "heldout_boundary_predictions.csv"
    heldout_predictions.to_csv(heldout_predictions_path, index=False)

    hot_metrics = _stream_metric_summary(hot_validation_pred)
    cold_metrics = _stream_metric_summary(cold_validation_pred)
    cp_j_kgk = float(config["reference_conditions"]["cp_J_per_kgK"])
    globals_by_case = {
        str(case_id): summary["globals"]
        for case_id, summary in artifacts["summary"]["per_case"].items()
        if case_id in validation_case_ids
    }
    case_rows = []
    for case_id in validation_case_ids:
        hot_case = hot_validation_pred[hot_validation_pred["case_id"] == case_id].reset_index(drop=True)
        cold_case = cold_validation_pred[cold_validation_pred["case_id"] == case_id].reset_index(drop=True)
        case_rows.append(_case_summary_row(case_id, hot_case, cold_case, globals_by_case[case_id], cp_j_kgk))

    aggregate = _aggregate_case_rows(case_rows)
    summary_payload = {
        "validation_protocol": {
            "type": "locked_internal_holdout",
            "description": (
                "Final synthetic holdout validation using the locked conditioned config on reserved cases. "
                "This is an internal holdout, not an external dataset."
            ),
            "config": str(Path(args.config).resolve()),
            "training_mode": training_mode,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "output_dir": str(output_dir),
            "train_case_ids": train_case_ids,
            "validation_case_ids": validation_case_ids,
            "freeze_k_wall": bool(args.freeze_k_wall),
            "adam_epochs_override": args.adam_epochs,
            "temperature_calibration_json": str(Path(args.temperature_calibration_json).resolve())
            if args.temperature_calibration_json
            else None,
            "config_overrides": list(args.set),
        },
        "surface_validation_metrics": {
            "hot_rmse_K": float(hot_metrics["rmse_K"]),
            "cold_rmse_K": float(cold_metrics["rmse_K"]),
            "combined_rmse_K": float(hot_metrics["rmse_K"] + cold_metrics["rmse_K"]),
            "hot_mae_K": float(hot_metrics["mae_K"]),
            "cold_mae_K": float(cold_metrics["mae_K"]),
            "by_case_hot": hot_metrics.get("by_case", {}),
            "by_case_cold": cold_metrics.get("by_case", {}),
        },
        "physics_checks": aggregate,
        "case_results": case_rows,
        "temperature_calibration": calibration_payload,
        "artifacts": {
            "heldout_boundary_predictions_csv": str(heldout_predictions_path.resolve()),
            "case_summary_csv": str((output_dir / "final_validation_case_summary.csv").resolve()),
            "summary_json": str((output_dir / "final_validation_summary.json").resolve()),
        },
    }

    case_summary_path = output_dir / "final_validation_case_summary.csv"
    pd.DataFrame(case_rows).to_csv(case_summary_path, index=False)
    summary_path = output_dir / "final_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(
        "Final holdout combined RMSE: "
        f"{summary_payload['surface_validation_metrics']['combined_rmse_K']:.6f} K"
    )
    print(
        "Mean Q error / energy gap: "
        f"{aggregate['mean_Q_total_rel_error_pct']:.3f}% / {aggregate['mean_energy_balance_gap_pct']:.3f}%"
    )
    print(f"Validation summary saved to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
