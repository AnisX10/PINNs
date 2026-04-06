from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import load_config
from pinn_hex.data.comsol import build_case_artifacts
from pinn_hex.physics.double_pipe import geometry_from_config, operating_point_from_config
from pinn_hex.training.trainer import PINNTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the reduced-order inverse PINN.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_countercurrent.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--adam-epochs", type=int, default=None, help="Override Adam epochs for quick experiments.")
    parser.add_argument("--skip-lbfgs", action="store_true", help="Skip the second-stage L-BFGS refinement.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    artifacts = build_case_artifacts(config)
    trainer = PINNTrainer(
        config=config,
        geometry=geometry_from_config(config),
        operating=operating_point_from_config(config),
        hot_profile=artifacts["preprocessed"]["hot_processed"],
        cold_profile=artifacts["preprocessed"]["cold_processed"],
        mesh_classified=artifacts["mesh"],
        hot_train_profile=artifacts["preprocessed"]["hot_train"],
        cold_train_profile=artifacts["preprocessed"]["cold_train"],
        hot_val_profile=artifacts["preprocessed"]["hot_val"],
        cold_val_profile=artifacts["preprocessed"]["cold_val"],
    )
    result = trainer.fit(
        output_dir=config["paths"]["output_dir"],
        adam_epochs_override=args.adam_epochs,
        skip_lbfgs=args.skip_lbfgs,
    )
    print(f"Checkpoint saved to: {result.checkpoint_path.resolve()}")
    print(f"Predictions saved to: {result.predictions_path.resolve()}")
    print(f"Metrics saved to: {result.metrics_path.resolve()}")


if __name__ == "__main__":
    main()
