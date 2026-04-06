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
        out_features: int = 1,
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
class PINNOutputs:
    hot: torch.Tensor
    cold: torch.Tensor


class DoublePipePINN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        fourier_features: int,
        fourier_sigma: float,
        initial_u: float,
        initial_alpha_hot: float,
        initial_alpha_cold: float,
        initial_hot_ambient_coupling: float,
        hot_half_length: float,
        cold_half_length: float,
        time_scale_s: float,
        hot_inlet_temperature_K: float,
        cold_inlet_temperature_K: float,
        temperature_scale_K: float,
        cold_inlet_learnable: bool,
        cold_inlet_max_offset_fraction: float,
    ) -> None:
        super().__init__()
        self.hot_half_length = hot_half_length
        self.cold_half_length = cold_half_length
        self.time_scale_s = time_scale_s
        self.hot_inlet_temperature_K = hot_inlet_temperature_K
        self.cold_inlet_temperature_K = cold_inlet_temperature_K
        self.temperature_scale_K = temperature_scale_K
        self.cold_inlet_max_offset_K = cold_inlet_max_offset_fraction * temperature_scale_K
        self.cold_inlet_learnable = cold_inlet_learnable
        base_features = 2
        if fourier_features > 0:
            self.hot_encoding = FourierFeatures(base_features, fourier_features, fourier_sigma)
            self.cold_encoding = FourierFeatures(base_features, fourier_features, fourier_sigma)
            encoded_features = 2 * fourier_features
        else:
            self.hot_encoding = None
            self.cold_encoding = None
            encoded_features = base_features
        self.hot_net = MLP(encoded_features, hidden_dim, num_hidden_layers, activation)
        self.cold_net = MLP(encoded_features, hidden_dim, num_hidden_layers, activation)
        self.log_u = nn.Parameter(torch.log(torch.tensor(float(initial_u), dtype=torch.float32)))
        self.log_alpha_hot = nn.Parameter(
            torch.log(torch.tensor(float(initial_alpha_hot), dtype=torch.float32))
        )
        self.log_alpha_cold = nn.Parameter(
            torch.log(torch.tensor(float(initial_alpha_cold), dtype=torch.float32))
        )
        self.log_hot_ambient_coupling = nn.Parameter(
            torch.log(torch.tensor(float(initial_hot_ambient_coupling), dtype=torch.float32))
        )
        self.cold_inlet_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def U(self) -> torch.Tensor:
        return torch.exp(self.log_u)

    @property
    def alpha_hot(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_hot)

    @property
    def alpha_cold(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_cold)

    @property
    def hot_ambient_coupling(self) -> torch.Tensor:
        return torch.exp(self.log_hot_ambient_coupling)

    @property
    def cold_inlet_effective(self) -> torch.Tensor:
        if not self.cold_inlet_learnable:
            return torch.tensor(self.cold_inlet_temperature_K, dtype=torch.float32, device=self.log_u.device)
        return self.cold_inlet_temperature_K + self.cold_inlet_max_offset_K * torch.tanh(self.cold_inlet_shift)

    def _streamwise_coordinate(self, z: torch.Tensor, stream: str) -> torch.Tensor:
        if stream == "hot":
            return (self.hot_half_length - z) / (2.0 * self.hot_half_length)
        if stream == "cold":
            return (z + self.cold_half_length) / (2.0 * self.cold_half_length)
        raise ValueError(f"Unsupported stream: {stream}")

    def _encode(self, s: torch.Tensor, t: torch.Tensor, stream: str) -> torch.Tensor:
        s_centered = 2.0 * s - 1.0
        t_scaled = t / self.time_scale_s
        x = torch.cat([s_centered, t_scaled], dim=-1)
        encoding = self.hot_encoding if stream == "hot" else self.cold_encoding
        if encoding is None:
            return x
        return encoding(x)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> PINNOutputs:
        s_hot = self._streamwise_coordinate(z, "hot")
        s_cold = self._streamwise_coordinate(z, "cold")
        x_hot = self._encode(s_hot, t, "hot")
        x_cold = self._encode(s_cold, t, "cold")
        raw_hot = self.hot_net(x_hot)
        raw_cold = self.cold_net(x_cold)
        cold_inlet_effective = self.cold_inlet_effective
        available_span = self.hot_inlet_temperature_K - cold_inlet_effective
        hot = self.hot_inlet_temperature_K + available_span * s_hot * torch.tanh(raw_hot)
        cold = cold_inlet_effective + available_span * s_cold * torch.sigmoid(raw_cold)
        return PINNOutputs(hot=hot, cold=cold)
