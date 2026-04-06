from __future__ import annotations

from typing import Any

import torch


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def _laplacian(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    first = _grad(outputs, inputs)
    second_x = _grad(first[:, 0:1], inputs)[:, 0:1]
    second_y = _grad(first[:, 1:2], inputs)[:, 1:2]
    second_z = _grad(first[:, 2:3], inputs)[:, 2:3]
    return second_x + second_y + second_z


def fluid_residuals(
    state: Any,
    xyz: torch.Tensor,
    rho_kg_m3: float,
    nu_m2_s: torch.Tensor,
    alpha_m2_s: torch.Tensor,
) -> dict[str, torch.Tensor]:
    grad_u = _grad(state.u, xyz)
    grad_v = _grad(state.v, xyz)
    grad_w = _grad(state.w, xyz)
    grad_p = _grad(state.p, xyz)
    grad_T = _grad(state.T, xyz)
    lap_u = _laplacian(state.u, xyz)
    lap_v = _laplacian(state.v, xyz)
    lap_w = _laplacian(state.w, xyz)
    lap_T = _laplacian(state.T, xyz)

    continuity = grad_u[:, 0:1] + grad_v[:, 1:2] + grad_w[:, 2:3]
    convection_u = state.u * grad_u[:, 0:1] + state.v * grad_u[:, 1:2] + state.w * grad_u[:, 2:3]
    convection_v = state.u * grad_v[:, 0:1] + state.v * grad_v[:, 1:2] + state.w * grad_v[:, 2:3]
    convection_w = state.u * grad_w[:, 0:1] + state.v * grad_w[:, 1:2] + state.w * grad_w[:, 2:3]
    convection_T = state.u * grad_T[:, 0:1] + state.v * grad_T[:, 1:2] + state.w * grad_T[:, 2:3]

    momentum_u = convection_u + grad_p[:, 0:1] / rho_kg_m3 - nu_m2_s * lap_u
    momentum_v = convection_v + grad_p[:, 1:2] / rho_kg_m3 - nu_m2_s * lap_v
    momentum_w = convection_w + grad_p[:, 2:3] / rho_kg_m3 - nu_m2_s * lap_w
    energy = convection_T - alpha_m2_s * lap_T
    return {
        "continuity": continuity,
        "momentum_u": momentum_u,
        "momentum_v": momentum_v,
        "momentum_w": momentum_w,
        "energy": energy,
    }


def wall_residuals(
    state: Any,
    xyz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {"conduction": _laplacian(state.T, xyz)}


def conjugate_interface_residuals(
    model: Any,
    hot_xyz: torch.Tensor,
    wall_inner_xyz: torch.Tensor,
    wall_outer_xyz: torch.Tensor,
    cold_xyz: torch.Tensor,
    rho_kg_m3: float,
    cp_J_kgK: float,
    hot_operating_point: torch.Tensor | None = None,
    wall_inner_operating_point: torch.Tensor | None = None,
    wall_outer_operating_point: torch.Tensor | None = None,
    cold_operating_point: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    hot_state = model.hot(hot_xyz, hot_operating_point)
    wall_inner_state = model.wall(wall_inner_xyz, wall_inner_operating_point)
    wall_outer_state = model.wall(wall_outer_xyz, wall_outer_operating_point)
    cold_state = model.cold(cold_xyz, cold_operating_point)
    grad_hot_T = _grad(hot_state.T, hot_xyz)
    grad_wall_inner_T = _grad(wall_inner_state.T, wall_inner_xyz)
    grad_wall_outer_T = _grad(wall_outer_state.T, wall_outer_xyz)
    grad_cold_T = _grad(cold_state.T, cold_xyz)

    hot_r = torch.sqrt(torch.clamp(hot_xyz[:, 0:1] ** 2 + hot_xyz[:, 1:2] ** 2, min=1.0e-12))
    cold_r = torch.sqrt(torch.clamp(cold_xyz[:, 0:1] ** 2 + cold_xyz[:, 1:2] ** 2, min=1.0e-12))
    hot_normal = torch.cat([hot_xyz[:, 0:1] / hot_r, hot_xyz[:, 1:2] / hot_r, torch.zeros_like(hot_r)], dim=1)
    cold_normal_inward = torch.cat(
        [-cold_xyz[:, 0:1] / cold_r, -cold_xyz[:, 1:2] / cold_r, torch.zeros_like(cold_r)],
        dim=1,
    )

    k_hot = rho_kg_m3 * cp_J_kgK * model.alpha_hot
    k_cold = rho_kg_m3 * cp_J_kgK * model.alpha_cold
    k_wall = model.k_wall
    q_hot = -k_hot * torch.sum(grad_hot_T * hot_normal, dim=1, keepdim=True)
    q_wall_inner = -k_wall * torch.sum(grad_wall_inner_T * hot_normal, dim=1, keepdim=True)
    q_wall_outer = -k_wall * torch.sum(grad_wall_outer_T * hot_normal, dim=1, keepdim=True)
    q_cold = -k_cold * torch.sum(grad_cold_T * cold_normal_inward, dim=1, keepdim=True)
    return {
        "temp_hot_wall": hot_state.T - wall_inner_state.T,
        "flux_hot_wall": q_hot - q_wall_inner,
        "temp_wall_cold": wall_outer_state.T - cold_state.T,
        "flux_wall_cold": q_wall_outer - q_cold,
        "ordering": torch.relu(cold_state.T - hot_state.T),
    }
