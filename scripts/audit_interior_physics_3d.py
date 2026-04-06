from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.config import apply_overrides, load_config
from pinn_hex.data.synthetic import resolve_synthetic_3d_config
from pinn_hex.models.factory_3d import build_double_pipe_pinn_3d
from pinn_hex.physics.double_pipe import operating_point_from_config
from pinn_hex.physics.double_pipe_3d import (
    geometry3d_from_config,
    sample_annulus_volume,
    sample_cylinder_volume,
    sample_interface_pair,
)
from pinn_hex.physics.multiphysics_3d import conjugate_interface_residuals, fluid_residuals, wall_residuals
from pinn_hex.postprocess.operating_points import resolve_operating_point
from pinn_hex.utils.repro import resolve_device


DEFAULT_CONFIG = ROOT / "configs" / "double_pipe_3d_synthetic_conditioned_validation_optphys.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs_3d_synthetic_qagg_positivep_optphys2_10ep_dpcal_walltune" / "checkpoints" / "best_model_3d.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit interior PDE/interface residuals for the final 3D PINN.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint path.")
    parser.add_argument("--case-id", default=None, help="Synthetic case id used to resolve the operating point.")
    parser.add_argument("--Th-in", type=float, default=None, help="Hot inlet temperature [K].")
    parser.add_argument("--Tc-in", type=float, default=None, help="Cold inlet temperature [K].")
    parser.add_argument("--uh-in", type=float, default=None, help="Hot inlet velocity [m/s].")
    parser.add_argument("--uc-in", type=float, default=None, help="Cold inlet velocity [m/s].")
    parser.add_argument("--output-json", default="outputs_3d_synthetic_interior_probe/interior_physics_audit.json", help="Audit JSON path.")
    parser.add_argument("--hot-points", type=int, default=4000, help="Hot-fluid interior sample count.")
    parser.add_argument("--cold-points", type=int, default=4000, help="Cold-fluid interior sample count.")
    parser.add_argument("--wall-points", type=int, default=2500, help="Wall interior sample count.")
    parser.add_argument("--interface-points", type=int, default=4000, help="Interface sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value using dotted notation.",
    )
    return parser


def _operating_tensor(count: int, point: dict[str, float | str], device: torch.device) -> torch.Tensor:
    values = np.array(
        [[point["Th_in_K"], point["Tc_in_K"], point["uh_in_mps"], point["uc_in_mps"]]],
        dtype=np.float32,
    )
    tiled = np.repeat(values, count, axis=0)
    return torch.tensor(tiled, dtype=torch.float32, device=device)


def _summary(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    abs_values = np.abs(values)
    return {
        "mean_abs": float(abs_values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95_abs": float(np.percentile(abs_values, 95.0)),
        "max_abs": float(abs_values.max()),
    }


def _temperature_summary(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config = apply_overrides(config, list(args.set))
    config = resolve_synthetic_3d_config(config)
    geometry = geometry3d_from_config(config)
    operating = operating_point_from_config(config)
    device = resolve_device(str(config["training_3d"].get("device", "auto")))
    point = resolve_operating_point(config, args.case_id, args.Th_in, args.Tc_in, args.uh_in, args.uc_in)
    rng = np.random.default_rng(int(args.seed))

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

    hot_xyz_np = sample_cylinder_volume(
        int(args.hot_points),
        geometry.hot_radius_m,
        -geometry.hot_half_length_m,
        geometry.hot_half_length_m,
        rng,
    )
    cold_xyz_np = sample_annulus_volume(
        int(args.cold_points),
        geometry.cold_inner_radius_m,
        geometry.cold_outer_radius_m,
        -geometry.cold_half_length_m,
        geometry.cold_half_length_m,
        rng,
    )
    wall_xyz_np = sample_annulus_volume(
        int(args.wall_points),
        geometry.hot_radius_m,
        geometry.cold_inner_radius_m,
        -geometry.cold_half_length_m,
        geometry.cold_half_length_m,
        rng,
    )
    hot_if_np, cold_if_np = sample_interface_pair(
        int(args.interface_points),
        geometry.hot_radius_m,
        geometry.cold_inner_radius_m,
        -geometry.cold_half_length_m,
        geometry.cold_half_length_m,
        rng,
    )

    hot_xyz = torch.tensor(hot_xyz_np, dtype=torch.float32, device=device, requires_grad=True)
    cold_xyz = torch.tensor(cold_xyz_np, dtype=torch.float32, device=device, requires_grad=True)
    wall_xyz = torch.tensor(wall_xyz_np, dtype=torch.float32, device=device, requires_grad=True)
    hot_if = torch.tensor(hot_if_np, dtype=torch.float32, device=device, requires_grad=True)
    cold_if = torch.tensor(cold_if_np, dtype=torch.float32, device=device, requires_grad=True)
    hot_ops = _operating_tensor(len(hot_xyz_np), point, device)
    cold_ops = _operating_tensor(len(cold_xyz_np), point, device)
    wall_ops = _operating_tensor(len(wall_xyz_np), point, device)
    hot_if_ops = _operating_tensor(len(hot_if_np), point, device)
    cold_if_ops = _operating_tensor(len(cold_if_np), point, device)

    hot_state = model.hot(hot_xyz, hot_ops)
    cold_state = model.cold(cold_xyz, cold_ops)
    wall_state = model.wall(wall_xyz, wall_ops)
    hot_res = fluid_residuals(
        hot_state,
        hot_xyz,
        rho_kg_m3=float(operating.density_kg_per_m3),
        nu_m2_s=model.nu_hot,
        alpha_m2_s=model.alpha_hot,
    )
    cold_res = fluid_residuals(
        cold_state,
        cold_xyz,
        rho_kg_m3=float(operating.density_kg_per_m3),
        nu_m2_s=model.nu_cold,
        alpha_m2_s=model.alpha_cold,
    )
    wall_res = wall_residuals(wall_state, wall_xyz)
    interface_res = conjugate_interface_residuals(
        model,
        hot_if,
        hot_if.clone(),
        cold_if.clone(),
        cold_if,
        rho_kg_m3=float(operating.density_kg_per_m3),
        cp_J_kgK=float(operating.cp_J_per_kgK),
        hot_operating_point=hot_if_ops,
        wall_inner_operating_point=hot_if_ops,
        wall_outer_operating_point=cold_if_ops,
        cold_operating_point=cold_if_ops,
    )

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "operating_point": point,
        "sample_counts": {
            "hot_points": int(args.hot_points),
            "cold_points": int(args.cold_points),
            "wall_points": int(args.wall_points),
            "interface_points": int(args.interface_points),
        },
        "model_parameters": {
            "nu_hot_m2_s": float(model.nu_hot.detach().cpu()),
            "nu_cold_m2_s": float(model.nu_cold.detach().cpu()),
            "alpha_hot_m2_s": float(model.alpha_hot.detach().cpu()),
            "alpha_cold_m2_s": float(model.alpha_cold.detach().cpu()),
            "k_wall_w_mk": float(model.k_wall.detach().cpu()),
        },
        "temperature_ranges": {
            "hot_fluid_K": _temperature_summary(hot_state.T),
            "cold_fluid_K": _temperature_summary(cold_state.T),
            "wall_K": _temperature_summary(wall_state.T),
        },
        "hot_fluid_residuals": {name: _summary(values) for name, values in hot_res.items()},
        "cold_fluid_residuals": {name: _summary(values) for name, values in cold_res.items()},
        "wall_residuals": {name: _summary(values) for name, values in wall_res.items()},
        "interface_residuals": {name: _summary(values) for name, values in interface_res.items()},
        "note": (
            "This is a physics-consistency audit over model-predicted interior probe points. "
            "It is not independent volumetric validation against ground-truth CFD fields."
        ),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Interior physics audit saved to: {output_path.resolve()}")
    print(json.dumps(summary["interface_residuals"], indent=2))


if __name__ == "__main__":
    main()
