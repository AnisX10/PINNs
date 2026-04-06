from __future__ import annotations

from pinn_hex.models.pinn_3d import DoublePipePINN3D
from pinn_hex.physics.double_pipe import OperatingPoint
from pinn_hex.physics.double_pipe_3d import ThreeDGeometry


def build_double_pipe_pinn_3d(
    config: dict,
    geometry: ThreeDGeometry,
    operating: OperatingPoint,
    hot_inlet_temperature_K: float,
    cold_inlet_temperature_K: float,
) -> DoublePipePINN3D:
    model_cfg = config["model_3d"]
    return DoublePipePINN3D(
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        activation=str(model_cfg["activation"]),
        fourier_features=int(model_cfg["fourier_features"]),
        fourier_sigma=float(model_cfg["fourier_sigma"]),
        hot_radius_m=float(geometry.hot_radius_m),
        cold_inner_radius_m=float(geometry.cold_inner_radius_m),
        cold_outer_radius_m=float(geometry.cold_outer_radius_m),
        hot_half_length_m=float(geometry.hot_half_length_m),
        cold_half_length_m=float(geometry.cold_half_length_m),
        initial_nu_hot_m2_s=float(model_cfg["initial_nu_hot_m2_s"]),
        initial_nu_cold_m2_s=float(model_cfg["initial_nu_cold_m2_s"]),
        initial_alpha_hot_m2_s=float(model_cfg["initial_alpha_hot_m2_s"]),
        initial_alpha_cold_m2_s=float(model_cfg["initial_alpha_cold_m2_s"]),
        initial_k_wall_w_mk=float(model_cfg["initial_k_wall_w_mk"]),
        learn_wall_conductivity=bool(model_cfg.get("learn_wall_conductivity", True)),
        hot_inlet_velocity_m_s=float(operating.inlet_velocity_hot_m_per_s),
        cold_inlet_velocity_m_s=float(operating.inlet_velocity_cold_m_per_s),
        hot_inlet_temperature_K=float(hot_inlet_temperature_K),
        cold_inlet_temperature_K=float(cold_inlet_temperature_K),
        learn_hot_inlet_temperature=bool(model_cfg.get("learn_hot_inlet_temperature", False)),
        learn_cold_inlet_temperature=bool(model_cfg.get("learn_cold_inlet_temperature", True)),
        velocity_correction_scale_m_s=float(model_cfg["velocity_correction_scale_m_s"]),
        velocity_condition_scale_m_s=float(
            model_cfg.get(
                "velocity_condition_scale_m_s",
                max(
                    float(operating.inlet_velocity_hot_m_per_s),
                    float(operating.inlet_velocity_cold_m_per_s),
                    1.0,
                ),
            )
        ),
        pressure_scale_pa=float(model_cfg["pressure_scale_pa"]),
        temperature_reference_K=float(model_cfg["temperature_reference_K"]),
        temperature_scale_K=float(model_cfg["temperature_scale_K"]),
        condition_on_operating_point=bool(model_cfg.get("condition_on_operating_point", False)),
        conditioning_feature_mode=str(model_cfg.get("conditioning_feature_mode", "raw")),
        temperature_head_mode=str(model_cfg.get("temperature_head_mode", "signed_tanh")),
        pressure_head_mode=str(model_cfg.get("pressure_head_mode", "raw_linear")),
    )
