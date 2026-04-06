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
from pinn_hex.data.case_matrix import resolve_case_matrix_3d_config
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.postprocess.temperature_calibration import apply_temperature_calibration, load_temperature_calibration
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import ThreeDGeometry, geometry3d_from_config
from pinn_hex.utils.repro import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict boundary states for a conditioned 3D PINN checkpoint ensemble.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "double_pipe_3d_case_matrix_conditioned_final.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional checkpoint path. Can be passed multiple times. Defaults to the 4 final CV fold checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(ROOT / "outputs_3d_case_matrix_conditioned_case_cv_final"),
        help="Checkpoint root used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Fold ids to load from --checkpoint-root when --checkpoint is omitted.",
    )
    parser.add_argument("--Th-in", type=float, required=True, help="Hot inlet temperature [K].")
    parser.add_argument("--Tc-in", type=float, required=True, help="Cold inlet temperature [K].")
    parser.add_argument("--uh-in", type=float, required=True, help="Hot inlet velocity [m/s].")
    parser.add_argument("--uc-in", type=float, required=True, help="Cold inlet velocity [m/s].")
    parser.add_argument(
        "--output",
        default="outputs_3d_case_matrix_boundary_inference/boundary_state_predictions.csv",
        help="CSV path for predicted boundary states.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSON metadata path. Defaults next to --output.",
    )
    parser.add_argument(
        "--temperature-calibration-json",
        default=None,
        help="Optional boundary-temperature calibration JSON applied to fluid temperatures.",
    )
    parser.add_argument("--n-theta", type=int, default=96, help="Azimuthal grid resolution.")
    parser.add_argument("--n-axial", type=int, default=96, help="Axial grid resolution for cylindrical surfaces.")
    parser.add_argument("--n-radial-hot", type=int, default=24, help="Radial grid resolution for hot inlet/outlet disks.")
    parser.add_argument("--n-radial-cold", type=int, default=24, help="Radial grid resolution for cold inlet/outlet annulus disks.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted notation, e.g. model_3d.hidden_dim=192",
    )
    return parser


def _theta_grid(n_theta: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float32)


def _cylinder_surface_frame(
    boundary: str,
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
    n_theta: int,
    n_axial: int,
) -> pd.DataFrame:
    theta = _theta_grid(n_theta)
    z = np.linspace(z_min_m, z_max_m, int(n_axial), dtype=np.float32)
    theta_grid, z_grid = np.meshgrid(theta, z, indexing="xy")
    x = radius_m * np.cos(theta_grid)
    y = radius_m * np.sin(theta_grid)
    frame = pd.DataFrame(
        {
            "boundary": boundary,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": z_grid.reshape(-1),
        }
    )
    frame["r"] = radius_m
    frame["phi_rad"] = np.arctan2(frame["y"], frame["x"])
    return frame


def _disk_frame(
    boundary: str,
    radius_m: float,
    z_value_m: float,
    n_theta: int,
    n_radial: int,
) -> pd.DataFrame:
    theta = _theta_grid(n_theta)
    radial = radius_m * np.sqrt(np.linspace(0.0, 1.0, int(n_radial), dtype=np.float32))
    theta_grid, radial_grid = np.meshgrid(theta, radial, indexing="xy")
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    z = np.full_like(x, z_value_m)
    frame = pd.DataFrame(
        {
            "boundary": boundary,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": z.reshape(-1),
        }
    )
    frame["r"] = np.sqrt(frame["x"] ** 2 + frame["y"] ** 2)
    frame["phi_rad"] = np.arctan2(frame["y"], frame["x"])
    return frame


def _annulus_disk_frame(
    boundary: str,
    inner_radius_m: float,
    outer_radius_m: float,
    z_value_m: float,
    n_theta: int,
    n_radial: int,
) -> pd.DataFrame:
    theta = _theta_grid(n_theta)
    radial = np.sqrt(
        np.linspace(inner_radius_m**2, outer_radius_m**2, int(n_radial), dtype=np.float32)
    )
    theta_grid, radial_grid = np.meshgrid(theta, radial, indexing="xy")
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    z = np.full_like(x, z_value_m)
    frame = pd.DataFrame(
        {
            "boundary": boundary,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": z.reshape(-1),
        }
    )
    frame["r"] = np.sqrt(frame["x"] ** 2 + frame["y"] ** 2)
    frame["phi_rad"] = np.arctan2(frame["y"], frame["x"])
    return frame


def _build_boundary_frames(
    geometry: ThreeDGeometry,
    n_theta: int,
    n_axial: int,
    n_radial_hot: int,
    n_radial_cold: int,
) -> list[tuple[str, str, pd.DataFrame]]:
    return [
        ("hot_fluid", "hot_inlet", _disk_frame("hot_inlet", geometry.hot_radius_m, geometry.hot_half_length_m, n_theta, n_radial_hot)),
        ("hot_fluid", "hot_outlet", _disk_frame("hot_outlet", geometry.hot_radius_m, -geometry.hot_half_length_m, n_theta, n_radial_hot)),
        (
            "hot_fluid",
            "hot_wall",
            _cylinder_surface_frame(
                "hot_wall",
                geometry.hot_radius_m,
                -geometry.hot_half_length_m,
                geometry.hot_half_length_m,
                n_theta,
                n_axial,
            ),
        ),
        (
            "cold_fluid",
            "cold_inlet",
            _annulus_disk_frame(
                "cold_inlet",
                geometry.cold_inner_radius_m,
                geometry.cold_outer_radius_m,
                -geometry.cold_half_length_m,
                n_theta,
                n_radial_cold,
            ),
        ),
        (
            "cold_fluid",
            "cold_outlet",
            _annulus_disk_frame(
                "cold_outlet",
                geometry.cold_inner_radius_m,
                geometry.cold_outer_radius_m,
                geometry.cold_half_length_m,
                n_theta,
                n_radial_cold,
            ),
        ),
        (
            "cold_fluid",
            "cold_inner_wall",
            _cylinder_surface_frame(
                "cold_inner_wall",
                geometry.cold_inner_radius_m,
                -geometry.cold_half_length_m,
                geometry.cold_half_length_m,
                n_theta,
                n_axial,
            ),
        ),
        (
            "wall_solid",
            "wall_inner_interface",
            _cylinder_surface_frame(
                "wall_inner_interface",
                geometry.hot_radius_m,
                -geometry.cold_half_length_m,
                geometry.cold_half_length_m,
                n_theta,
                n_axial,
            ),
        ),
        (
            "wall_solid",
            "wall_outer_interface",
            _cylinder_surface_frame(
                "wall_outer_interface",
                geometry.cold_inner_radius_m,
                -geometry.cold_half_length_m,
                geometry.cold_half_length_m,
                n_theta,
                n_axial,
            ),
        ),
    ]


def _resolve_checkpoint_paths(checkpoints: list[str], checkpoint_root: str, folds: list[int]) -> list[Path]:
    if checkpoints:
        paths = [Path(path) for path in checkpoints]
    else:
        root = Path(checkpoint_root)
        paths = [root / f"fold_{fold}" / "checkpoints" / "best_model_3d.pt" for fold in folds]
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing checkpoint paths: {missing_list}")
    return paths


def _build_operating_tensor(
    count: int,
    Th_in: float,
    Tc_in: float,
    uh_in: float,
    uc_in: float,
    device: torch.device,
) -> torch.Tensor:
    values = np.array([[Th_in, Tc_in, uh_in, uc_in]], dtype=np.float32)
    tiled = np.repeat(values, count, axis=0)
    return torch.tensor(tiled, dtype=torch.float32, device=device)


def _state_frame(
    domain: str,
    boundary: str,
    frame: pd.DataFrame,
    values_by_model: list[np.ndarray],
) -> pd.DataFrame:
    stacked = np.stack(values_by_model, axis=0)
    mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0, ddof=0)
    result = frame.copy()
    result["domain"] = domain
    result["u_pred_mean"] = mean_values[:, 0]
    result["v_pred_mean"] = mean_values[:, 1]
    result["w_pred_mean"] = mean_values[:, 2]
    result["p_pred_mean_Pa"] = mean_values[:, 3]
    result["T_pred_mean_K"] = mean_values[:, 4]
    result["u_pred_std"] = std_values[:, 0]
    result["v_pred_std"] = std_values[:, 1]
    result["w_pred_std"] = std_values[:, 2]
    result["p_pred_std_Pa"] = std_values[:, 3]
    result["T_pred_std_K"] = std_values[:, 4]
    result["ensemble_size"] = stacked.shape[0]
    return result


def _wall_frame(
    boundary: str,
    frame: pd.DataFrame,
    values_by_model: list[np.ndarray],
) -> pd.DataFrame:
    stacked = np.stack(values_by_model, axis=0)
    mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0, ddof=0)
    result = frame.copy()
    result["domain"] = "wall_solid"
    result["T_pred_mean_K"] = mean_values[:, 0]
    result["T_pred_std_K"] = std_values[:, 0]
    result["ensemble_size"] = stacked.shape[0]
    return result


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = apply_overrides(config, list(args.set))
    config = resolve_case_matrix_3d_config(config)
    geometry = geometry3d_from_config(config)
    operating = operating_point_from_config(config)
    device = resolve_device(str(config["training_3d"].get("device", "auto")))

    checkpoint_paths = _resolve_checkpoint_paths(list(args.checkpoint), args.checkpoint_root, list(args.folds))
    models = []
    for checkpoint_path in checkpoint_paths:
        model = build_double_pipe_pinn_3d(
            config=config,
            geometry=geometry,
            operating=operating,
            hot_inlet_temperature_K=float(config["reference_conditions"]["hot_inlet_temperature_K"]),
            cold_inlet_temperature_K=float(config["reference_conditions"]["cold_inlet_temperature_K"]),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models.append(model)

    boundary_frames = _build_boundary_frames(
        geometry=geometry,
        n_theta=int(args.n_theta),
        n_axial=int(args.n_axial),
        n_radial_hot=int(args.n_radial_hot),
        n_radial_cold=int(args.n_radial_cold),
    )
    predictions: list[pd.DataFrame] = []
    with torch.no_grad():
        for domain, boundary, frame in boundary_frames:
            xyz = torch.tensor(frame[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
            operating_point = _build_operating_tensor(
                count=len(frame),
                Th_in=float(args.Th_in),
                Tc_in=float(args.Tc_in),
                uh_in=float(args.uh_in),
                uc_in=float(args.uc_in),
                device=device,
            )
            if domain == "hot_fluid":
                values_by_model = []
                for model in models:
                    state = model.hot(xyz, operating_point)
                    values_by_model.append(
                        torch.cat([state.u, state.v, state.w, state.p, state.T], dim=1).detach().cpu().numpy()
                    )
                predictions.append(_state_frame(domain, boundary, frame, values_by_model))
            elif domain == "cold_fluid":
                values_by_model = []
                for model in models:
                    state = model.cold(xyz, operating_point)
                    values_by_model.append(
                        torch.cat([state.u, state.v, state.w, state.p, state.T], dim=1).detach().cpu().numpy()
                    )
                predictions.append(_state_frame(domain, boundary, frame, values_by_model))
            else:
                values_by_model = []
                for model in models:
                    state = model.wall(xyz, operating_point)
                    values_by_model.append(state.T.detach().cpu().numpy())
                predictions.append(_wall_frame(boundary, frame, values_by_model))

    output_frame = pd.concat(predictions, ignore_index=True)
    calibration_payload = None
    if args.temperature_calibration_json:
        calibration_payload = load_temperature_calibration(args.temperature_calibration_json)
        output_frame = apply_temperature_calibration(
            output_frame,
            calibration_payload,
            prediction_column="T_pred_mean_K",
            correction_column="T_calibration_K",
        )
    output_frame["Th_in_K"] = float(args.Th_in)
    output_frame["Tc_in_K"] = float(args.Tc_in)
    output_frame["uh_in_mps"] = float(args.uh_in)
    output_frame["uc_in_mps"] = float(args.uc_in)
    output_frame.to_csv(output_path, index=False)

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "ensemble_size": len(checkpoint_paths),
        "operating_point": {
            "Th_in_K": float(args.Th_in),
            "Tc_in_K": float(args.Tc_in),
            "uh_in_mps": float(args.uh_in),
            "uc_in_mps": float(args.uc_in),
        },
        "grid": {
            "n_theta": int(args.n_theta),
            "n_axial": int(args.n_axial),
            "n_radial_hot": int(args.n_radial_hot),
            "n_radial_cold": int(args.n_radial_cold),
        },
        "geometry": {
            "hot_half_length_m": float(geometry.hot_half_length_m),
            "cold_half_length_m": float(geometry.cold_half_length_m),
            "hot_radius_m": float(geometry.hot_radius_m),
            "cold_inner_radius_m": float(geometry.cold_inner_radius_m),
            "cold_outer_radius_m": float(geometry.cold_outer_radius_m),
        },
        "temperature_calibration": calibration_payload,
        "output_csv": str(output_path.resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Boundary predictions saved to: {output_path.resolve()}")
    print(f"Metadata saved to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
