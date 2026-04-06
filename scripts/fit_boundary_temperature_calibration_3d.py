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
from pinn_hex.data.case_matrix import OPERATING_COLUMNS, resolve_case_matrix_3d_config
from pinn_hex.data.threed import build_3d_case_artifacts
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import geometry3d_from_config
from pinn_hex.postprocess.temperature_calibration import apply_temperature_calibration, compute_boundary_bias_calibration


DEFAULT_BOUNDARIES = ["hot_inlet", "hot_wall", "cold_inlet", "cold_inner_wall"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit boundary-temperature bias calibration on training cases.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--output-json", required=True, help="Output calibration JSON path.")
    parser.add_argument(
        "--boundaries",
        nargs="+",
        default=list(DEFAULT_BOUNDARIES),
        help="Boundary names to calibrate.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Additional config override using dotted notation.",
    )
    return parser


def _predict_stream_frame(model: torch.nn.Module, frame: pd.DataFrame, stream: str) -> pd.DataFrame:
    xyz = torch.tensor(frame[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32)
    operating = torch.tensor(frame[list(OPERATING_COLUMNS)].to_numpy(dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        state = model.hot(xyz, operating) if stream == "hot" else model.cold(xyz, operating)
    result = frame.copy()
    result["T_pred"] = state.T.detach().cpu().numpy().reshape(-1)
    result["stream"] = stream
    return result


def _combined_rmse(frame: pd.DataFrame) -> float:
    hot = frame[frame["stream"] == "hot"]
    cold = frame[frame["stream"] == "cold"]
    hot_rmse = float(np.sqrt(np.mean((hot["T_pred"].to_numpy(dtype=np.float64) - hot["T"].to_numpy(dtype=np.float64)) ** 2)))
    cold_rmse = float(
        np.sqrt(np.mean((cold["T_pred"].to_numpy(dtype=np.float64) - cold["T"].to_numpy(dtype=np.float64)) ** 2))
    )
    return hot_rmse + cold_rmse


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, list(args.set))
    config = resolve_case_matrix_3d_config(config)
    artifacts = build_3d_case_artifacts(config)
    geometry = geometry3d_from_config(config)
    operating = operating_point_from_config(config)
    model = build_double_pipe_pinn_3d(
        config=config,
        geometry=geometry,
        operating=operating,
        hot_inlet_temperature_K=float(config["reference_conditions"]["hot_inlet_temperature_K"]),
        cold_inlet_temperature_K=float(config["reference_conditions"]["cold_inlet_temperature_K"]),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    hot_train_pred = _predict_stream_frame(model, artifacts["hot_train"], "hot")
    cold_train_pred = _predict_stream_frame(model, artifacts["cold_train"], "cold")
    hot_val_pred = _predict_stream_frame(model, artifacts["hot_validation"], "hot")
    cold_val_pred = _predict_stream_frame(model, artifacts["cold_validation"], "cold")
    train_pred = pd.concat([hot_train_pred, cold_train_pred], ignore_index=True)
    validation_pred = pd.concat([hot_val_pred, cold_val_pred], ignore_index=True)

    corrections = compute_boundary_bias_calibration(
        train_pred,
        prediction_column="T_pred",
        target_column="T",
        boundaries=[str(boundary) for boundary in args.boundaries],
    )
    payload = {
        "type": "boundary_temperature_bias",
        "description": "Additive fluid-boundary temperature correction fitted on training-case residual means.",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "fit_boundaries": [str(boundary) for boundary in args.boundaries],
        "train_case_ids": sorted(str(case_id) for case_id in train_pred["case_id"].drop_duplicates().tolist()),
        "validation_case_ids": sorted(str(case_id) for case_id in validation_pred["case_id"].drop_duplicates().tolist()),
        "corrections_K": corrections,
    }

    calibrated_train = apply_temperature_calibration(train_pred, payload, prediction_column="T_pred")
    calibrated_validation = apply_temperature_calibration(validation_pred, payload, prediction_column="T_pred")
    payload["preview_metrics"] = {
        "train_combined_rmse_before_K": _combined_rmse(train_pred),
        "train_combined_rmse_after_K": _combined_rmse(calibrated_train),
        "validation_combined_rmse_before_K": _combined_rmse(validation_pred),
        "validation_combined_rmse_after_K": _combined_rmse(calibrated_validation),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
