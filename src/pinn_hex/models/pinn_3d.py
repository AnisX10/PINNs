from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, n_features: int, sigma: float) -> None:
        super().__init__()
        self.register_buffer("B", sigma * torch.randn(in_features, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.pi * x @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        out_features: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        act = _activation(activation)
        dims = [in_features] + [hidden_dim] * num_hidden_layers
        for left, right in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(left, right)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, act])
        head = nn.Linear(hidden_dim, out_features)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FluidState3D:
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor
    T: torch.Tensor


@dataclass
class WallState3D:
    T: torch.Tensor


class DoublePipePINN3D(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        fourier_features: int,
        fourier_sigma: float,
        hot_radius_m: float,
        cold_inner_radius_m: float,
        cold_outer_radius_m: float,
        hot_half_length_m: float,
        cold_half_length_m: float,
        initial_nu_hot_m2_s: float,
        initial_nu_cold_m2_s: float,
        initial_alpha_hot_m2_s: float,
        initial_alpha_cold_m2_s: float,
        initial_k_wall_w_mk: float,
        learn_wall_conductivity: bool,
        hot_inlet_velocity_m_s: float,
        cold_inlet_velocity_m_s: float,
        hot_inlet_temperature_K: float,
        cold_inlet_temperature_K: float,
        learn_hot_inlet_temperature: bool,
        learn_cold_inlet_temperature: bool,
        velocity_correction_scale_m_s: float,
        velocity_condition_scale_m_s: float,
        pressure_scale_pa: float,
        temperature_reference_K: float,
        temperature_scale_K: float,
        condition_on_operating_point: bool = False,
        conditioning_feature_mode: str = "raw",
        temperature_head_mode: str = "signed_tanh",
        pressure_head_mode: str = "raw_linear",
    ) -> None:
        super().__init__()
        self.hot_radius_m = hot_radius_m
        self.cold_inner_radius_m = cold_inner_radius_m
        self.cold_outer_radius_m = cold_outer_radius_m
        self.hot_half_length_m = hot_half_length_m
        self.cold_half_length_m = cold_half_length_m
        self.hot_inlet_velocity_m_s = hot_inlet_velocity_m_s
        self.cold_inlet_velocity_m_s = cold_inlet_velocity_m_s
        self.velocity_correction_scale_m_s = velocity_correction_scale_m_s
        self.velocity_condition_scale_m_s = velocity_condition_scale_m_s
        self.pressure_scale_pa = pressure_scale_pa
        self.temperature_reference_K = temperature_reference_K
        self.temperature_scale_K = temperature_scale_K
        self.condition_on_operating_point = condition_on_operating_point
        self.conditioning_feature_mode = conditioning_feature_mode.lower().strip()
        self.temperature_head_mode = temperature_head_mode.lower().strip()
        self.pressure_head_mode = pressure_head_mode.lower().strip()
        if self.conditioning_feature_mode not in {"raw", "raw_plus_derived"}:
            raise ValueError(
                "Unsupported conditioning_feature_mode: "
                f"{conditioning_feature_mode}. Expected 'raw' or 'raw_plus_derived'."
            )
        if self.temperature_head_mode not in {"signed_tanh", "signed_span_square", "hot_span_square"}:
            raise ValueError(
                "Unsupported temperature_head_mode: "
                f"{temperature_head_mode}. Expected 'signed_tanh', 'signed_span_square', or 'hot_span_square'."
            )
        if self.pressure_head_mode not in {"raw_linear", "positive_velocity_square"}:
            raise ValueError(
                "Unsupported pressure_head_mode: "
                f"{pressure_head_mode}. Expected 'raw_linear' or 'positive_velocity_square'."
            )

        base_features = 3
        if self.condition_on_operating_point:
            condition_features = 4 if self.conditioning_feature_mode == "raw" else 8
        else:
            condition_features = 0
        if fourier_features > 0:
            self.hot_encoding = FourierFeatures(base_features, fourier_features, fourier_sigma)
            self.cold_encoding = FourierFeatures(base_features, fourier_features, fourier_sigma)
            self.wall_encoding = FourierFeatures(base_features, fourier_features, fourier_sigma)
            encoded_features = 2 * fourier_features + condition_features
        else:
            self.hot_encoding = None
            self.cold_encoding = None
            self.wall_encoding = None
            encoded_features = base_features + condition_features

        self.hot_net = MLP(encoded_features, hidden_dim, num_hidden_layers, activation, out_features=5)
        self.cold_net = MLP(encoded_features, hidden_dim, num_hidden_layers, activation, out_features=5)
        self.wall_net = MLP(encoded_features, hidden_dim, num_hidden_layers, activation, out_features=1)

        self.log_nu_hot = nn.Parameter(torch.log(torch.tensor(float(initial_nu_hot_m2_s), dtype=torch.float32)))
        self.log_nu_cold = nn.Parameter(torch.log(torch.tensor(float(initial_nu_cold_m2_s), dtype=torch.float32)))
        self.log_alpha_hot = nn.Parameter(
            torch.log(torch.tensor(float(initial_alpha_hot_m2_s), dtype=torch.float32))
        )
        self.log_alpha_cold = nn.Parameter(
            torch.log(torch.tensor(float(initial_alpha_cold_m2_s), dtype=torch.float32))
        )
        self.log_k_wall = nn.Parameter(
            torch.log(torch.tensor(float(initial_k_wall_w_mk), dtype=torch.float32)),
            requires_grad=learn_wall_conductivity,
        )
        self.hot_inlet_temperature_param = nn.Parameter(
            torch.tensor(0.0 if self.condition_on_operating_point else float(hot_inlet_temperature_K), dtype=torch.float32),
            requires_grad=learn_hot_inlet_temperature,
        )
        self.cold_inlet_temperature_param = nn.Parameter(
            torch.tensor(0.0 if self.condition_on_operating_point else float(cold_inlet_temperature_K), dtype=torch.float32),
            requires_grad=learn_cold_inlet_temperature,
        )
        self.log_hot_pressure_gain = nn.Parameter(torch.log(torch.tensor(3.0, dtype=torch.float32)))
        self.log_cold_pressure_gain = nn.Parameter(torch.log(torch.tensor(3.0, dtype=torch.float32)))

    @property
    def nu_hot(self) -> torch.Tensor:
        return torch.exp(self.log_nu_hot)

    @property
    def nu_cold(self) -> torch.Tensor:
        return torch.exp(self.log_nu_cold)

    @property
    def alpha_hot(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_hot)

    @property
    def alpha_cold(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_cold)

    @property
    def k_wall(self) -> torch.Tensor:
        return torch.exp(self.log_k_wall)

    @property
    def hot_inlet_temperature(self) -> torch.Tensor:
        return self.hot_inlet_temperature_param

    @property
    def cold_inlet_temperature(self) -> torch.Tensor:
        return self.cold_inlet_temperature_param

    @property
    def hot_inlet_temperature_bias(self) -> torch.Tensor:
        return self.hot_inlet_temperature_param

    @property
    def cold_inlet_temperature_bias(self) -> torch.Tensor:
        return self.cold_inlet_temperature_param

    def _normalize(self, xyz: torch.Tensor, stream: str) -> torch.Tensor:
        if stream == "hot":
            radial_scale = self.hot_radius_m
            axial_scale = self.hot_half_length_m
        elif stream == "cold":
            radial_scale = self.cold_outer_radius_m
            axial_scale = self.cold_half_length_m
        else:
            radial_scale = self.cold_inner_radius_m
            axial_scale = self.cold_half_length_m
        scaled = xyz.clone()
        scaled[:, 0:1] = scaled[:, 0:1] / radial_scale
        scaled[:, 1:2] = scaled[:, 1:2] / radial_scale
        scaled[:, 2:3] = scaled[:, 2:3] / axial_scale
        return scaled

    def _normalize_operating_point(self, operating_point: torch.Tensor | None) -> torch.Tensor | None:
        if operating_point is None:
            return None
        normalized = operating_point.clone()
        normalized[:, 0:1] = (normalized[:, 0:1] - self.temperature_reference_K) / self.temperature_scale_K
        normalized[:, 1:2] = (normalized[:, 1:2] - self.temperature_reference_K) / self.temperature_scale_K
        normalized[:, 2:3] = normalized[:, 2:3] / max(self.velocity_condition_scale_m_s, 1.0e-12)
        normalized[:, 3:4] = normalized[:, 3:4] / max(self.velocity_condition_scale_m_s, 1.0e-12)
        if self.conditioning_feature_mode == "raw_plus_derived":
            hot_temp = operating_point[:, 0:1]
            cold_temp = operating_point[:, 1:2]
            hot_velocity = operating_point[:, 2:3]
            cold_velocity = operating_point[:, 3:4]
            mean_temperature = 0.5 * (hot_temp + cold_temp)
            temperature_span = hot_temp - cold_temp
            mean_velocity = 0.5 * (hot_velocity + cold_velocity)
            velocity_log_ratio = torch.log((hot_velocity + 1.0e-6) / (cold_velocity + 1.0e-6))
            derived = torch.cat(
                [
                    (mean_temperature - self.temperature_reference_K) / self.temperature_scale_K,
                    temperature_span / self.temperature_scale_K,
                    mean_velocity / max(self.velocity_condition_scale_m_s, 1.0e-12),
                    velocity_log_ratio,
                ],
                dim=-1,
            )
            normalized = torch.cat([normalized, derived], dim=-1)
        return normalized

    def _encode(self, xyz: torch.Tensor, stream: str, operating_point: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self._normalize(xyz, stream)
        if stream == "hot":
            encoder = self.hot_encoding
        elif stream == "cold":
            encoder = self.cold_encoding
        else:
            encoder = self.wall_encoding
        features = normalized if encoder is None else encoder(normalized)
        if self.condition_on_operating_point:
            normalized_operating = self._normalize_operating_point(operating_point)
            if normalized_operating is None:
                raise ValueError("Operating-point conditioning is enabled but no operating-point tensor was provided.")
            features = torch.cat([features, normalized_operating], dim=-1)
        return features

    def _hot_wall_factor(self, xyz: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.clamp(xyz[:, 0:1] ** 2 + xyz[:, 1:2] ** 2, min=1.0e-12))
        return torch.clamp(1.0 - (r / self.hot_radius_m) ** 2, min=0.0, max=1.0)

    def _cold_wall_factor(self, xyz: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(torch.clamp(xyz[:, 0:1] ** 2 + xyz[:, 1:2] ** 2, min=1.0e-12))
        mid = 0.5 * (self.cold_inner_radius_m + self.cold_outer_radius_m)
        half_gap = 0.5 * (self.cold_outer_radius_m - self.cold_inner_radius_m)
        factor = 1.0 - ((r - mid) / max(half_gap, 1.0e-12)) ** 2
        return torch.clamp(factor, min=0.0, max=1.0)

    def hot_profile_factor(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._hot_wall_factor(xyz)

    def cold_profile_factor(self, xyz: torch.Tensor) -> torch.Tensor:
        return self._cold_wall_factor(xyz)

    def _stream_progress(self, xyz: torch.Tensor, stream: str) -> torch.Tensor:
        z = xyz[:, 2:3]
        if stream == "hot":
            progress = (self.hot_half_length_m - z) / max(2.0 * self.hot_half_length_m, 1.0e-12)
        else:
            progress = (z + self.cold_half_length_m) / max(2.0 * self.cold_half_length_m, 1.0e-12)
        return torch.clamp(progress, min=0.0, max=1.0)

    def _resolve_inlet_velocity(self, operating_point: torch.Tensor | None, stream: str) -> torch.Tensor:
        if self.condition_on_operating_point:
            if operating_point is None:
                raise ValueError("Operating-point conditioning is enabled but no operating-point tensor was provided.")
            column = 2 if stream == "hot" else 3
            return operating_point[:, column : column + 1]
        velocity = self.hot_inlet_velocity_m_s if stream == "hot" else self.cold_inlet_velocity_m_s
        return torch.full_like(operating_point[:, 0:1] if operating_point is not None else torch.zeros(1, 1), float(velocity))

    def _resolve_inlet_temperature(self, operating_point: torch.Tensor | None, stream: str) -> torch.Tensor:
        if self.condition_on_operating_point:
            if operating_point is None:
                raise ValueError("Operating-point conditioning is enabled but no operating-point tensor was provided.")
            column = 0 if stream == "hot" else 1
            base_temperature = operating_point[:, column : column + 1]
            bias = self.hot_inlet_temperature_bias if stream == "hot" else self.cold_inlet_temperature_bias
            return base_temperature + bias
        parameter = self.hot_inlet_temperature if stream == "hot" else self.cold_inlet_temperature
        return parameter

    def _resolve_temperature_span_scale(self, operating_point: torch.Tensor | None, template: torch.Tensor) -> torch.Tensor:
        if self.condition_on_operating_point:
            if operating_point is None:
                raise ValueError("Operating-point conditioning is enabled but no operating-point tensor was provided.")
            span = torch.clamp(operating_point[:, 0:1] - operating_point[:, 1:2], min=1.0e-6)
            return span
        hot_temperature = self.hot_inlet_temperature
        cold_temperature = self.cold_inlet_temperature
        span = torch.clamp(hot_temperature - cold_temperature, min=1.0e-6)
        return torch.full_like(template, float(span.detach().cpu()))

    def _resolve_pressure_drop_scale(self, operating_point: torch.Tensor | None, stream: str, template: torch.Tensor) -> torch.Tensor:
        if self.condition_on_operating_point:
            if operating_point is None:
                raise ValueError("Operating-point conditioning is enabled but no operating-point tensor was provided.")
            column = 2 if stream == "hot" else 3
            velocity = torch.clamp(operating_point[:, column : column + 1], min=1.0e-6)
        else:
            velocity_value = self.hot_inlet_velocity_m_s if stream == "hot" else self.cold_inlet_velocity_m_s
            velocity = torch.full_like(template, float(max(velocity_value, 1.0e-6)))
        gain = torch.exp(self.log_hot_pressure_gain if stream == "hot" else self.log_cold_pressure_gain)
        velocity_ratio = velocity / max(self.velocity_condition_scale_m_s, 1.0e-12)
        return self.pressure_scale_pa * gain * velocity_ratio**2

    @staticmethod
    def _bounded_positive(raw: torch.Tensor) -> torch.Tensor:
        squared = raw**2
        return squared / (1.0 + squared)

    def _decode(
        self,
        raw: torch.Tensor,
        xyz: torch.Tensor,
        stream: str,
        operating_point: torch.Tensor | None = None,
    ) -> FluidState3D:
        wall_factor = self._hot_wall_factor(xyz) if stream == "hot" else self._cold_wall_factor(xyz)
        axial_sign = -1.0 if stream == "hot" else 1.0
        inlet_velocity = self._resolve_inlet_velocity(operating_point, stream)
        stream_progress = self._stream_progress(xyz, stream)
        outlet_factor = 1.0 - stream_progress
        inlet_temperature = self._resolve_inlet_temperature(operating_point, stream)
        u = stream_progress * self.velocity_correction_scale_m_s * wall_factor * torch.tanh(raw[:, 0:1])
        v = stream_progress * self.velocity_correction_scale_m_s * wall_factor * torch.tanh(raw[:, 1:2])
        w = wall_factor * (
            axial_sign * inlet_velocity + stream_progress * self.velocity_correction_scale_m_s * torch.tanh(raw[:, 2:3])
        )
        if self.pressure_head_mode == "positive_velocity_square":
            pressure_fraction = self._bounded_positive(raw[:, 3:4])
            pressure_drop_scale = self._resolve_pressure_drop_scale(operating_point, stream, pressure_fraction)
            p = outlet_factor * pressure_drop_scale * pressure_fraction
        else:
            p = outlet_factor * self.pressure_scale_pa * raw[:, 3:4]
        use_span_square = self.temperature_head_mode == "signed_span_square" or (
            self.temperature_head_mode == "hot_span_square" and stream == "hot"
        )
        if use_span_square:
            temperature_scale = self._resolve_temperature_span_scale(operating_point, inlet_temperature)
            temperature_fraction = self._bounded_positive(raw[:, 4:5])
            temperature_delta = stream_progress * temperature_scale * temperature_fraction
            if stream == "hot":
                T = inlet_temperature - temperature_delta
            else:
                T = inlet_temperature + temperature_delta
        else:
            T = inlet_temperature + stream_progress * self.temperature_scale_K * torch.tanh(raw[:, 4:5])
        return FluidState3D(u=u, v=v, w=w, p=p, T=T)

    def hot(self, xyz: torch.Tensor, operating_point: torch.Tensor | None = None) -> FluidState3D:
        return self._decode(self.hot_net(self._encode(xyz, "hot", operating_point)), xyz, "hot", operating_point)

    def cold(self, xyz: torch.Tensor, operating_point: torch.Tensor | None = None) -> FluidState3D:
        return self._decode(self.cold_net(self._encode(xyz, "cold", operating_point)), xyz, "cold", operating_point)

    def wall(self, xyz: torch.Tensor, operating_point: torch.Tensor | None = None) -> WallState3D:
        raw = self.wall_net(self._encode(xyz, "wall", operating_point))
        T = self.temperature_reference_K + self.temperature_scale_K * torch.tanh(raw[:, 0:1])
        return WallState3D(T=T)
