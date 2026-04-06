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

from pinn_hex.config import load_config
from pinn_hex.data.case_matrix import OPERATING_COLUMNS, resolve_case_matrix_3d_config
from pinn_hex.data.threed import build_3d_case_artifacts
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import geometry3d_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune only the 3D wall branch on interface temperatures.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for tuned checkpoint and history.")
    parser.add_argument("--epochs", type=int, default=400, help="Wall-only Adam epochs.")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Wall-only Adam learning rate.")
    return parser


def _wall_batch(frame: pd.DataFrame, boundary: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subset = frame[frame["boundary"] == boundary].reset_index(drop=True)
    xyz = torch.tensor(subset[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32)
    operating = torch.tensor(subset[list(OPERATING_COLUMNS)].to_numpy(dtype=np.float32), dtype=torch.float32)
    target = torch.tensor(subset[["T"]].to_numpy(dtype=np.float32), dtype=torch.float32)
    return xyz, operating, target


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_case_matrix_3d_config(load_config(args.config))
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
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("wall_net.")
    model.train()

    train_hot = _wall_batch(artifacts["hot_train"], "hot_wall")
    train_cold = _wall_batch(artifacts["cold_train"], "cold_inner_wall")
    val_hot = _wall_batch(artifacts["hot_validation"], "hot_wall")
    val_cold = _wall_batch(artifacts["cold_validation"], "cold_inner_wall")
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_rmse = float("inf")
    history: list[dict[str, float]] = []
    output_dir = (ROOT / args.output_dir).resolve()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        train_loss = torch.zeros((), dtype=torch.float32)
        for xyz, ops, target in (train_hot, train_cold):
            train_loss = train_loss + torch.mean((model.wall(xyz, ops).T - target) ** 2)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_terms = []
            for xyz, ops, target in (val_hot, val_cold):
                val_terms.append(torch.mean((model.wall(xyz, ops).T - target) ** 2))
            train_rmse = float(torch.sqrt(train_loss / 2.0).cpu())
            val_rmse = float(torch.sqrt(sum(val_terms) / len(val_terms)).cpu())
        history.append(
            {
                "epoch": float(epoch),
                "train_wall_rmse_K": train_rmse,
                "validation_wall_rmse_K": val_rmse,
            }
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch={epoch:04d} train_wall_rmse={train_rmse:.4f} validation_wall_rmse={val_rmse:.4f}")

    if best_state is None:
        raise RuntimeError("Wall tuning did not produce a checkpoint.")

    torch.save(best_state, output_dir / "checkpoints" / "best_model_3d.pt")
    (output_dir / "wall_tune_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation wall RMSE: {best_val_rmse:.4f} K")
    print(f"Checkpoint saved to: {(output_dir / 'checkpoints' / 'best_model_3d.pt').resolve()}")


if __name__ == "__main__":
    main()
