from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import load_config
from pinn_hex.data.case_matrix import resolve_case_matrix_3d_config
from pinn_hex.data.threed import build_3d_case_artifacts, save_3d_case_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare 3D multiphysics PINN surface datasets.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_3d_case_matrix_case001.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_case_matrix_3d_config(load_config(args.config))
    artifacts = build_3d_case_artifacts(config)
    save_3d_case_artifacts(artifacts, config["paths"]["processed_dir"])

    figures_dir = Path(config["paths"]["output_dir"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.5))
    hot = artifacts["hot_surface"]
    cold = artifacts["cold_surface"]
    plt.scatter(hot["z"], hot["r"], c=hot["T"], s=8, cmap="Reds", alpha=0.5, label="Hot surface")
    plt.scatter(cold["z"], cold["r"], c=cold["T"], s=8, cmap="Blues", alpha=0.5, label="Cold surface")
    plt.xlabel("Axial coordinate z [m]")
    plt.ylabel("Radius r [m]")
    plt.title("3D surface-temperature supervision points")
    plt.tight_layout()
    plt.savefig(figures_dir / "surface_points_3d.png", dpi=200)
    plt.close()

    print(f"Saved 3D processed data to: {Path(config['paths']['processed_dir']).resolve()}")
    print(f"Saved 3D surface plot to: {(figures_dir / 'surface_points_3d.png').resolve()}")


if __name__ == "__main__":
    main()
