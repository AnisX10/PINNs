from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch


@dataclass(frozen=True)
class Geometry:
    hot_half_length_m: float
    cold_half_length_m: float
    hot_radius_m: float
    annulus_outer_radius_m: float
    radial_tolerance_m: float

    @property
    def hot_length_m(self) -> float:
        return 2.0 * self.hot_half_length_m

    @property
    def cold_length_m(self) -> float:
        return 2.0 * self.cold_half_length_m

    @property
    def hot_area_m2(self) -> float:
        return pi * self.hot_radius_m**2

    @property
    def cold_area_m2(self) -> float:
        return pi * (self.annulus_outer_radius_m**2 - self.hot_radius_m**2)

    @property
    def interface_perimeter_m(self) -> float:
        return 2.0 * pi * self.hot_radius_m

    @property
    def annulus_hydraulic_diameter_m(self) -> float:
        return 2.0 * (self.annulus_outer_radius_m - self.hot_radius_m)

    @property
    def hot_diameter_m(self) -> float:
        return 2.0 * self.hot_radius_m


@dataclass(frozen=True)
class OperatingPoint:
    hot_inlet_temperature_K: float
    cold_inlet_temperature_K: float
    initial_temperature_K: float
    inlet_velocity_hot_m_per_s: float
    inlet_velocity_cold_m_per_s: float
    density_kg_per_m3: float
    cp_J_per_kgK: float

    def hot_mass_flow_kg_s(self, geometry: Geometry) -> float:
        return self.density_kg_per_m3 * geometry.hot_area_m2 * self.inlet_velocity_hot_m_per_s

    def cold_mass_flow_kg_s(self, geometry: Geometry) -> float:
        return self.density_kg_per_m3 * geometry.cold_area_m2 * self.inlet_velocity_cold_m_per_s


def geometry_from_config(config: dict) -> Geometry:
    section = config["geometry"]
    return Geometry(
        hot_half_length_m=float(section["hot_half_length_m"]),
        cold_half_length_m=float(section["cold_half_length_m"]),
        hot_radius_m=float(section["hot_radius_m"]),
        annulus_outer_radius_m=float(section["annulus_outer_radius_m"]),
        radial_tolerance_m=float(section["radial_tolerance_m"]),
    )


def operating_point_from_config(config: dict) -> OperatingPoint:
    section = config["reference_conditions"]
    return OperatingPoint(
        hot_inlet_temperature_K=float(section["hot_inlet_temperature_K"]),
        cold_inlet_temperature_K=float(section["cold_inlet_temperature_K"]),
        initial_temperature_K=float(section["initial_temperature_K"]),
        inlet_velocity_hot_m_per_s=float(section["inlet_velocity_hot_m_per_s"]),
        inlet_velocity_cold_m_per_s=float(section["inlet_velocity_cold_m_per_s"]),
        density_kg_per_m3=float(section["density_kg_per_m3"]),
        cp_J_per_kgK=float(section["cp_J_per_kgK"]),
    )


def exchange_mask(z: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    return (torch.abs(z) <= geometry.cold_half_length_m).to(z.dtype)

