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
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import ThreeDGeometry, geometry3d_from_config
from pinn_hex.postprocess.operating_points import resolve_operating_point
from pinn_hex.utils.repro import resolve_device


DEFAULT_CONFIG = ROOT / "configs" / "double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune" / "checkpoints" / "best_model_3d.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export dense pseudo-volume interior fields from the final 3D PINN.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint path.")
    parser.add_argument("--case-id", default=None, help="Case Matrix case id used to resolve the operating point.")
    parser.add_argument("--Th-in", type=float, default=None, help="Hot inlet temperature [K].")
    parser.add_argument("--Tc-in", type=float, default=None, help="Cold inlet temperature [K].")
    parser.add_argument("--uh-in", type=float, default=None, help="Hot inlet velocity [m/s].")
    parser.add_argument("--uc-in", type=float, default=None, help="Cold inlet velocity [m/s].")
    parser.add_argument(
        "--output",
        default="outputs_3d_case_matrix_interior_probe/interior_fields.csv",
        help="CSV path for interior field predictions.",
    )
    parser.add_argument("--metadata-output", default=None, help="Optional JSON metadata path.")
    parser.add_argument("--n-theta", type=int, default=48, help="Azimuthal grid resolution.")
    parser.add_argument("--n-axial", type=int, default=48, help="Axial grid resolution.")
    parser.add_argument("--n-radial-hot", type=int, default=16, help="Hot-fluid radial resolution.")
    parser.add_argument("--n-radial-cold", type=int, default=16, help="Cold-fluid radial resolution.")
    parser.add_argument("--n-radial-wall", type=int, default=10, help="Wall radial resolution.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted notation.",
    )
    return parser


def _theta_grid(n_theta: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False, dtype=np.float32)


def _cylinder_volume_frame(
    domain: str,
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
    n_theta: int,
    n_axial: int,
    n_radial: int,
) -> pd.DataFrame:
    theta = _theta_grid(n_theta)
    axial = np.linspace(z_min_m, z_max_m, int(n_axial), dtype=np.float32)
    radial = radius_m * np.sqrt(np.linspace(0.0, 1.0, int(n_radial), dtype=np.float32))
    theta_grid, axial_grid, radial_grid = np.meshgrid(theta, axial, radial, indexing="xy")
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    frame = pd.DataFrame(
        {
            "domain": domain,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": axial_grid.reshape(-1),
        }
    )
    frame["r"] = np.sqrt(frame["x"] ** 2 + frame["y"] ** 2)
    frame["phi_rad"] = np.arctan2(frame["y"], frame["x"])
    return frame


def _annulus_volume_frame(
    domain: str,
    inner_radius_m: float,
    outer_radius_m: float,
    z_min_m: float,
    z_max_m: float,
    n_theta: int,
    n_axial: int,
    n_radial: int,
) -> pd.DataFrame:
    theta = _theta_grid(n_theta)
    axial = np.linspace(z_min_m, z_max_m, int(n_axial), dtype=np.float32)
    radial = np.sqrt(
        np.linspace(inner_radius_m**2, outer_radius_m**2, int(n_radial), dtype=np.float32)
    )
    theta_grid, axial_grid, radial_grid = np.meshgrid(theta, axial, radial, indexing="xy")
    x = radial_grid * np.cos(theta_grid)
    y = radial_grid * np.sin(theta_grid)
    frame = pd.DataFrame(
        {
            "domain": domain,
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "z": axial_grid.reshape(-1),
        }
    )
    frame["r"] = np.sqrt(frame["x"] ** 2 + frame["y"] ** 2)
    frame["phi_rad"] = np.arctan2(frame["y"], frame["x"])
    return frame


def _operating_tensor(count: int, point: dict[str, float | str], device: torch.device) -> torch.Tensor:
    values = np.array(
        [[point["Th_in_K"], point["Tc_in_K"], point["uh_in_mps"], point["uc_in_mps"]]],
        dtype=np.float32,
    )
    tiled = np.repeat(values, count, axis=0)
    return torch.tensor(tiled, dtype=torch.float32, device=device)


def _predict_domain(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    point: dict[str, float | str],
    device: torch.device,
) -> pd.DataFrame:
    xyz = torch.tensor(frame[["x", "y", "z"]].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    operating = _operating_tensor(len(frame), point, device)
    result = frame.copy()
    result["u_pred"] = np.full(len(result), np.nan, dtype=np.float32)
    result["v_pred"] = np.full(len(result), np.nan, dtype=np.float32)
    result["w_pred"] = np.full(len(result), np.nan, dtype=np.float32)
    result["p_pred"] = np.full(len(result), np.nan, dtype=np.float32)
    with torch.no_grad():
        if str(frame["domain"].iloc[0]) == "hot_fluid":
            state = model.hot(xyz, operating)
            result["u_pred"] = state.u.detach().cpu().numpy().reshape(-1)
            result["v_pred"] = state.v.detach().cpu().numpy().reshape(-1)
            result["w_pred"] = state.w.detach().cpu().numpy().reshape(-1)
            result["p_pred"] = state.p.detach().cpu().numpy().reshape(-1)
            result["T_pred"] = state.T.detach().cpu().numpy().reshape(-1)
        elif str(frame["domain"].iloc[0]) == "cold_fluid":
            state = model.cold(xyz, operating)
            result["u_pred"] = state.u.detach().cpu().numpy().reshape(-1)
            result["v_pred"] = state.v.detach().cpu().numpy().reshape(-1)
            result["w_pred"] = state.w.detach().cpu().numpy().reshape(-1)
            result["p_pred"] = state.p.detach().cpu().numpy().reshape(-1)
            result["T_pred"] = state.T.detach().cpu().numpy().reshape(-1)
        else:
            state = model.wall(xyz, operating)
            result["T_pred"] = state.T.detach().cpu().numpy().reshape(-1)
    return result


def _build_frames(geometry: ThreeDGeometry, args: argparse.Namespace) -> list[pd.DataFrame]:
    return [
        _cylinder_volume_frame(
            "hot_fluid",
            geometry.hot_radius_m,
            -geometry.hot_half_length_m,
            geometry.hot_half_length_m,
            int(args.n_theta),
            int(args.n_axial),
            int(args.n_radial_hot),
        ),
        _annulus_volume_frame(
            "cold_fluid",
            geometry.cold_inner_radius_m,
            geometry.cold_outer_radius_m,
            -geometry.cold_half_length_m,
            geometry.cold_half_length_m,
            int(args.n_theta),
            int(args.n_axial),
            int(args.n_radial_cold),
        ),
        _annulus_volume_frame(
            "wall_solid",
            geometry.hot_radius_m,
            geometry.cold_inner_radius_m,
            -geometry.cold_half_length_m,
            geometry.cold_half_length_m,
            int(args.n_theta),
            int(args.n_axial),
            int(args.n_radial_wall),
        ),
    ]


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
    point = resolve_operating_point(config, args.case_id, args.Th_in, args.Tc_in, args.uh_in, args.uc_in)

    model = build_double_pipe_pinn_3d(
        config=config,
        geometry=geometry,
        operating=operating,
        hot_inlet_temperature_K=float(config["reference_conditions"]["hot_inlet_temperature_K"]),
        cold_inlet_temperature_K=float(config["reference_conditions"]["cold_inlet_temperature_K"]),
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    frames = _build_frames(geometry, args)
    predictions = [_predict_domain(model, frame, point, device) for frame in frames]
    output_frame = pd.concat(predictions, ignore_index=True)
    output_frame["case_id"] = str(point["case_id"])
    output_frame["Th_in_K"] = float(point["Th_in_K"])
    output_frame["Tc_in_K"] = float(point["Tc_in_K"])
    output_frame["uh_in_mps"] = float(point["uh_in_mps"])
    output_frame["uc_in_mps"] = float(point["uc_in_mps"])
    output_frame.to_csv(output_path, index=False)

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "operating_point": point,
        "grid": {
            "n_theta": int(args.n_theta),
            "n_axial": int(args.n_axial),
            "n_radial_hot": int(args.n_radial_hot),
            "n_radial_cold": int(args.n_radial_cold),
            "n_radial_wall": int(args.n_radial_wall),
        },
        "geometry": {
            "hot_half_length_m": float(geometry.hot_half_length_m),
            "cold_half_length_m": float(geometry.cold_half_length_m),
            "hot_radius_m": float(geometry.hot_radius_m),
            "cold_inner_radius_m": float(geometry.cold_inner_radius_m),
            "cold_outer_radius_m": float(geometry.cold_outer_radius_m),
        },
        "note": (
            "These are model-predicted pseudo-volume fields for inspection and visualization. "
            "They are not independent ground-truth validation data."
        ),
        "output_csv": str(output_path.resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Interior field export saved to: {output_path.resolve()}")
    print(f"Metadata saved to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
