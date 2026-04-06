from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class ThreeDGeometry:
    hot_half_length_m: float
    cold_half_length_m: float
    hot_radius_m: float
    cold_inner_radius_m: float
    cold_outer_radius_m: float
    surface_tolerance_m: float

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
        return pi * (self.cold_outer_radius_m**2 - self.cold_inner_radius_m**2)

    @property
    def interface_area_m2(self) -> float:
        return 2.0 * pi * self.hot_radius_m * self.cold_length_m


def geometry3d_from_config(config: dict) -> ThreeDGeometry:
    section = config["geometry_3d"]
    return ThreeDGeometry(
        hot_half_length_m=float(section["hot_half_length_m"]),
        cold_half_length_m=float(section["cold_half_length_m"]),
        hot_radius_m=float(section["hot_radius_m"]),
        cold_inner_radius_m=float(section["cold_inner_radius_m"]),
        cold_outer_radius_m=float(section["cold_outer_radius_m"]),
        surface_tolerance_m=float(section["surface_tolerance_m"]),
    )


def _stack_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.column_stack([x, y, z])


def sample_cylinder_volume(
    n: int,
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radial = radius_m * np.sqrt(rng.random(n))
    angle = 2.0 * np.pi * rng.random(n)
    z = rng.uniform(z_min_m, z_max_m, size=n)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)
    return _stack_xyz(x, y, z)


def sample_annulus_volume(
    n: int,
    inner_radius_m: float,
    outer_radius_m: float,
    z_min_m: float,
    z_max_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radial_sq = rng.uniform(inner_radius_m**2, outer_radius_m**2, size=n)
    radial = np.sqrt(radial_sq)
    angle = 2.0 * np.pi * rng.random(n)
    z = rng.uniform(z_min_m, z_max_m, size=n)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)
    return _stack_xyz(x, y, z)


def sample_disk(
    n: int,
    radius_m: float,
    z_value_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radial = radius_m * np.sqrt(rng.random(n))
    angle = 2.0 * np.pi * rng.random(n)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)
    z = np.full(n, z_value_m)
    return _stack_xyz(x, y, z)


def sample_annulus_disk(
    n: int,
    inner_radius_m: float,
    outer_radius_m: float,
    z_value_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radial_sq = rng.uniform(inner_radius_m**2, outer_radius_m**2, size=n)
    radial = np.sqrt(radial_sq)
    angle = 2.0 * np.pi * rng.random(n)
    x = radial * np.cos(angle)
    y = radial * np.sin(angle)
    z = np.full(n, z_value_m)
    return _stack_xyz(x, y, z)


def sample_cylindrical_surface(
    n: int,
    radius_m: float,
    z_min_m: float,
    z_max_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    angle = 2.0 * np.pi * rng.random(n)
    z = rng.uniform(z_min_m, z_max_m, size=n)
    x = radius_m * np.cos(angle)
    y = radius_m * np.sin(angle)
    return _stack_xyz(x, y, z)


def sample_interface_pair(
    n: int,
    hot_radius_m: float,
    cold_inner_radius_m: float,
    z_min_m: float,
    z_max_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * rng.random(n)
    z = rng.uniform(z_min_m, z_max_m, size=n)
    x_hot = hot_radius_m * np.cos(angle)
    y_hot = hot_radius_m * np.sin(angle)
    x_cold = cold_inner_radius_m * np.cos(angle)
    y_cold = cold_inner_radius_m * np.sin(angle)
    hot_points = _stack_xyz(x_hot, y_hot, z)
    cold_points = _stack_xyz(x_cold, y_cold, z)
    return hot_points, cold_points
