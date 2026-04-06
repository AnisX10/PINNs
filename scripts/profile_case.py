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
from pinn_hex.data.comsol import build_case_artifacts, save_case_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the COMSOL reference case and export reduced-order data.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_countercurrent.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    artifacts = build_case_artifacts(config)
    save_case_artifacts(artifacts, config["paths"]["processed_dir"])

    output_dir = Path(config["paths"]["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    hot = artifacts["hot_profile"]
    cold = artifacts["cold_profile"]
    hot_processed = artifacts["preprocessed"]["hot_processed"]
    cold_processed = artifacts["preprocessed"]["cold_processed"]
    train_points = artifacts["preprocessed"]["train"]
    val_points = artifacts["preprocessed"]["validation"]
    plt.figure(figsize=(10, 5))
    plt.plot(hot["z_mean"], hot["T_mean"], label="Hot profile", color="tab:red")
    plt.fill_between(
        hot["z_mean"],
        hot["T_mean"] - hot["T_std"].fillna(0.0),
        hot["T_mean"] + hot["T_std"].fillna(0.0),
        color="tab:red",
        alpha=0.15,
    )
    plt.plot(cold["z_mean"], cold["T_mean"], label="Cold profile", color="tab:blue")
    plt.fill_between(
        cold["z_mean"],
        cold["T_mean"] - cold["T_std"].fillna(0.0),
        cold["T_mean"] + cold["T_std"].fillna(0.0),
        color="tab:blue",
        alpha=0.15,
    )
    plt.xlabel("Axial coordinate z [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Reduced-order axial profiles from the COMSOL temperature field")
    plt.legend()
    plt.tight_layout()
    figure_path = figures_dir / "reduced_profiles.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(hot["z_mean"], hot["T_mean"], color="tab:red", alpha=0.35, linewidth=2, label="Hot binned raw")
    plt.plot(
        hot_processed["z"],
        hot_processed["T_processed_K"],
        color="tab:red",
        linewidth=2.5,
        label="Hot processed",
    )
    plt.plot(cold["z_mean"], cold["T_mean"], color="tab:blue", alpha=0.35, linewidth=2, label="Cold binned raw")
    plt.plot(
        cold_processed["z"],
        cold_processed["T_processed_K"],
        color="tab:blue",
        linewidth=2.5,
        label="Cold processed",
    )
    train_hot = train_points[train_points["stream"] == "hot"]
    val_hot = val_points[val_points["stream"] == "hot"]
    train_cold = train_points[train_points["stream"] == "cold"]
    val_cold = val_points[val_points["stream"] == "cold"]
    plt.scatter(train_hot["z"], train_hot["T_processed_K"], color="tab:red", s=12, marker="o", label="Hot train")
    plt.scatter(val_hot["z"], val_hot["T_processed_K"], color="tab:red", s=16, marker="x", label="Hot val")
    plt.scatter(train_cold["z"], train_cold["T_processed_K"], color="tab:blue", s=12, marker="o", label="Cold train")
    plt.scatter(val_cold["z"], val_cold["T_processed_K"], color="tab:blue", s=16, marker="x", label="Cold val")
    plt.xlabel("Axial coordinate z [m]")
    plt.ylabel("Temperature [K]")
    plt.title("Processed supervision profiles with train/validation split")
    plt.legend(ncol=2)
    plt.tight_layout()
    processed_figure_path = figures_dir / "processed_supervision.png"
    plt.savefig(processed_figure_path, dpi=200)
    plt.close()

    print(f"Saved reduced data to: {Path(config['paths']['processed_dir']).resolve()}")
    print(f"Saved profile plot to: {figure_path.resolve()}")
    print(f"Saved processed supervision plot to: {processed_figure_path.resolve()}")


if __name__ == "__main__":
    main()
