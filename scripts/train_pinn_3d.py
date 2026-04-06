from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import apply_overrides, load_config
from pinn_hex.data.synthetic import resolve_synthetic_3d_config
from pinn_hex.data.threed import build_3d_case_artifacts
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import geometry3d_from_config
from pinn_hex.training.trainer_3d import PINNTrainer3D


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the 3D multiphysics PINN.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_3d_synthetic_case001.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--adam-epochs", type=int, default=None, help="Override Adam epochs.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted notation, e.g. training_3d.data_weight=250.0",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path to load before training or refinement.",
    )
    parser.add_argument(
        "--resume-nonstrict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load resume checkpoint with strict=False to allow warm-starting changed architectures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, list(args.set))
    config = resolve_synthetic_3d_config(config)
    artifacts = build_3d_case_artifacts(config)
    trainer = PINNTrainer3D(
        config=config,
        geometry=geometry3d_from_config(config),
        operating=operating_point_from_config(config),
        hot_train=artifacts["hot_train"],
        cold_train=artifacts["cold_train"],
        hot_validation=artifacts["hot_validation"],
        cold_validation=artifacts["cold_validation"],
    )
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        trainer.model.load_state_dict(
            torch.load(checkpoint_path, map_location=trainer.device),
            strict=not bool(args.resume_nonstrict),
        )
    result = trainer.fit(output_dir=config["paths"]["output_dir"], adam_epochs_override=args.adam_epochs)
    print(f"Checkpoint saved to: {result.checkpoint_path.resolve()}")
    print(f"Predictions saved to: {result.predictions_path.resolve()}")
    print(f"Metrics saved to: {result.metrics_path.resolve()}")


if __name__ == "__main__":
    main()
