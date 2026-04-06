from __future__ import annotations

from typing import Any

import torch

from pinn_hex.physics.double_pipe import Geometry, OperatingPoint, exchange_mask


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def hot_residual(
    model: Any,
    geometry: Geometry,
    operating: OperatingPoint,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    outputs = model(z, t)
    th = outputs.hot
    tc = outputs.cold
    dth_dz = _grad(th, z)
    dth_dt = _grad(th, t)
    d2th_dz2 = _grad(dth_dz, z)
    coupling = model.U * geometry.interface_perimeter_m / (
        operating.density_kg_per_m3 * geometry.hot_area_m2 * operating.cp_J_per_kgK
    )
    non_overlap_mask = 1.0 - exchange_mask(z, geometry)
    return (
        dth_dt
        - operating.inlet_velocity_hot_m_per_s * dth_dz
        - model.alpha_hot * d2th_dz2
        + coupling * exchange_mask(z, geometry) * (th - tc)
        + model.hot_ambient_coupling * non_overlap_mask * (th - operating.initial_temperature_K)
    )


def cold_residual(
    model: Any,
    geometry: Geometry,
    operating: OperatingPoint,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    outputs = model(z, t)
    th = outputs.hot
    tc = outputs.cold
    dtc_dz = _grad(tc, z)
    dtc_dt = _grad(tc, t)
    d2tc_dz2 = _grad(dtc_dz, z)
    coupling = model.U * geometry.interface_perimeter_m / (
        operating.density_kg_per_m3 * geometry.cold_area_m2 * operating.cp_J_per_kgK
    )
    return (
        dtc_dt
        + operating.inlet_velocity_cold_m_per_s * dtc_dz
        - model.alpha_cold * d2tc_dz2
        - coupling * (th - tc)
    )


def compute_losses(
    model: Any,
    geometry: Geometry,
    operating: OperatingPoint,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    priors: dict[str, float],
    stationary: bool,
) -> dict[str, torch.Tensor]:
    temp_scale = operating.hot_inlet_temperature_K - operating.cold_inlet_temperature_K
    outputs_hot = model(batch["z_hot_data"], batch["t_hot_data"]).hot
    outputs_cold = model(batch["z_cold_data"], batch["t_cold_data"]).cold
    data_loss = torch.mean(batch["W_hot_data"] * ((outputs_hot - batch["T_hot_data"]) / temp_scale) ** 2) + torch.mean(
        batch["W_cold_data"] * ((outputs_cold - batch["T_cold_data"]) / temp_scale) ** 2
    )

    hot_pde = hot_residual(model, geometry, operating, batch["z_hot_collocation"], batch["t_hot_collocation"])
    cold_pde = cold_residual(model, geometry, operating, batch["z_cold_collocation"], batch["t_cold_collocation"])
    pde_loss = torch.mean(hot_pde**2) + torch.mean(cold_pde**2)

    hot_outlet = model(batch["z_hot_outlet"], batch["t_hot_outlet"]).hot
    cold_outlet = model(batch["z_cold_outlet"], batch["t_cold_outlet"]).cold
    d_hot_outlet_dz = _grad(hot_outlet, batch["z_hot_outlet"])
    d_cold_outlet_dz = _grad(cold_outlet, batch["z_cold_outlet"])
    cold_inlet_prior = ((model.cold_inlet_effective - batch["T_cold_inlet"].reshape(())) / temp_scale) ** 2
    bc_loss = torch.mean(d_hot_outlet_dz**2) + torch.mean(d_cold_outlet_dz**2)
    bc_loss = bc_loss + weights["cold_inlet_prior_weight"] * cold_inlet_prior

    if stationary:
        ic_loss = torch.zeros((), device=batch["z_hot_data"].device)
    else:
        hot_ic = model(batch["z_hot_ic"], batch["t_hot_ic"]).hot
        cold_ic = model(batch["z_cold_ic"], batch["t_cold_ic"]).cold
        ic_loss = torch.mean(((hot_ic - batch["T_hot_ic"]) / temp_scale) ** 2) + torch.mean(
            ((cold_ic - batch["T_cold_ic"]) / temp_scale) ** 2
        )

    hot_collocation_outputs = model(batch["z_hot_collocation"], batch["t_hot_collocation"]).hot
    cold_pair_outputs = model(batch["z_cold_collocation"], batch["t_cold_collocation"])
    cold_collocation_outputs = cold_pair_outputs.cold
    hot_on_cold_collocation = cold_pair_outputs.hot
    d_hot_collocation_dz = _grad(hot_collocation_outputs, batch["z_hot_collocation"])
    d_cold_collocation_dz = _grad(cold_collocation_outputs, batch["z_cold_collocation"])
    d2_hot_collocation_dz2 = _grad(d_hot_collocation_dz, batch["z_hot_collocation"])
    d2_cold_collocation_dz2 = _grad(d_cold_collocation_dz, batch["z_cold_collocation"])
    hot_overlap_mask = exchange_mask(batch["z_hot_collocation"], geometry)
    hot_monotonic = torch.sum(hot_overlap_mask * torch.relu(-d_hot_collocation_dz) ** 2) / torch.clamp(
        torch.sum(hot_overlap_mask), min=1.0
    )
    cold_monotonic = torch.mean(torch.relu(-d_cold_collocation_dz) ** 2)
    monotonic_loss = hot_monotonic + cold_monotonic
    ordering_loss = torch.mean(torch.relu(cold_collocation_outputs - hot_on_cold_collocation) ** 2)
    smoothness_loss = torch.mean(d2_hot_collocation_dz2**2) + torch.mean(d2_cold_collocation_dz2**2)

    reg_loss = ((model.U - priors["initial_u"]) / priors["initial_u"]) ** 2
    reg_loss = reg_loss + ((model.alpha_hot - priors["initial_alpha_hot"]) / priors["initial_alpha_hot"]) ** 2
    reg_loss = reg_loss + ((model.alpha_cold - priors["initial_alpha_cold"]) / priors["initial_alpha_cold"]) ** 2
    reg_loss = reg_loss + (
        (model.hot_ambient_coupling - priors["initial_hot_ambient_coupling"]) / priors["initial_hot_ambient_coupling"]
    ) ** 2

    total = (
        weights["data_weight"] * data_loss
        + weights["pde_weight"] * pde_loss
        + weights["bc_weight"] * bc_loss
        + weights["ic_weight"] * ic_loss
        + weights["reg_weight"] * reg_loss
        + weights["monotonic_weight"] * monotonic_loss
        + weights["ordering_weight"] * ordering_loss
        + weights["smoothness_weight"] * smoothness_loss
    )

    return {
        "total": total,
        "data": data_loss,
        "pde": pde_loss,
        "bc": bc_loss,
        "ic": ic_loss,
        "reg": reg_loss,
        "monotonic": monotonic_loss,
        "ordering": ordering_loss,
        "smoothness": smoothness_loss,
    }
