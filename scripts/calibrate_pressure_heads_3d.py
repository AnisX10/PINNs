from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import load_config
from pinn_hex.data.synthetic import OPERATING_COLUMNS, resolve_synthetic_3d_config
from pinn_hex.data.threed import build_3d_case_artifacts
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import geometry3d_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate positive-pressure PINN gains against training-case pressure-drop targets."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for calibrated checkpoint and summary.")
    return parser


def _predict_case_pressure_drop(
    model: torch.nn.Module,
    frame,
    stream: str,
) -> list[tuple[str, float, float]]:
    boundary_inlet = "hot_inlet" if stream == "hot" else "cold_inlet"
    boundary_outlet = "hot_outlet" if stream == "hot" else "cold_outlet"
    ref_column = "dp_hot" if stream == "hot" else "dp_cold"
    results: list[tuple[str, float, float]] = []
    for case_id in frame["case_id"].drop_duplicates().tolist():
        case_frame = frame[frame["case_id"] == case_id]
        inlet = case_frame[case_frame["boundary"] == boundary_inlet]
        outlet = case_frame[case_frame["boundary"] == boundary_outlet]
        xyz_in = torch.tensor(inlet[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        ops_in = torch.tensor(inlet[list(OPERATING_COLUMNS)].to_numpy(dtype=np.float32), dtype=torch.float32)
        xyz_out = torch.tensor(outlet[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        ops_out = torch.tensor(outlet[list(OPERATING_COLUMNS)].to_numpy(dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            inlet_state = model.hot(xyz_in, ops_in) if stream == "hot" else model.cold(xyz_in, ops_in)
            outlet_state = model.hot(xyz_out, ops_out) if stream == "hot" else model.cold(xyz_out, ops_out)
        predicted = float(inlet_state.p.mean().cpu() - outlet_state.p.mean().cpu())
        reference = float(case_frame[ref_column].iloc[0])
        results.append((str(case_id), predicted, reference))
    return results


def _fit_scale(rows: list[tuple[str, float, float]]) -> dict[str, float]:
    predicted = np.asarray([row[1] for row in rows], dtype=np.float64)
    reference = np.asarray([row[2] for row in rows], dtype=np.float64)
    scale = float((predicted @ reference) / max(predicted @ predicted, 1.0e-12))
    before = float(np.mean(np.abs(predicted - reference) / np.maximum(np.abs(reference), 1.0e-8) * 100.0))
    after = float(
        np.mean(np.abs(scale * predicted - reference) / np.maximum(np.abs(reference), 1.0e-8) * 100.0)
    )
    return {
        "scale": scale,
        "mean_rel_error_before_pct": before,
        "mean_rel_error_after_pct": after,
    }


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_synthetic_3d_config(load_config(args.config))
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
    checkpoint_path = Path(args.checkpoint)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    hot_rows = _predict_case_pressure_drop(model, artifacts["hot_train"], "hot")
    cold_rows = _predict_case_pressure_drop(model, artifacts["cold_train"], "cold")
    hot_stats = _fit_scale(hot_rows)
    cold_stats = _fit_scale(cold_rows)

    state_dict["log_hot_pressure_gain"] = state_dict["log_hot_pressure_gain"] + torch.tensor(
        math.log(hot_stats["scale"]),
        dtype=state_dict["log_hot_pressure_gain"].dtype,
    )
    state_dict["log_cold_pressure_gain"] = state_dict["log_cold_pressure_gain"] + torch.tensor(
        math.log(cold_stats["scale"]),
        dtype=state_dict["log_cold_pressure_gain"].dtype,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    checkpoint_out = output_dir / "checkpoints"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, checkpoint_out / "best_model_3d.pt")

    summary = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "output_checkpoint": str((checkpoint_out / "best_model_3d.pt").resolve()),
        "hot": hot_stats,
        "cold": cold_stats,
        "train_cases": sorted({row[0] for row in hot_rows}),
    }
    (output_dir / "pressure_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
