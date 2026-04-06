from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from pinn_hex.data.comsol import read_temperature_csv, summarize_temperature_csv
from pinn_hex.data.synthetic import build_synthetic_3d_case_artifacts, uses_synthetic_3d_data
from pinn_hex.physics.double_pipe_3d import ThreeDGeometry, geometry3d_from_config


def _polarize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["phi_rad"] = np.arctan2(result["y"], result["x"])
    result["phi_cos"] = np.cos(result["phi_rad"])
    result["phi_sin"] = np.sin(result["phi_rad"])
    return result


def _label_hot_boundaries(frame: pd.DataFrame, geometry: ThreeDGeometry) -> pd.DataFrame:
    result = frame.copy()
    result["boundary"] = "hot_wall"
    tol = geometry.surface_tolerance_m
    result.loc[result["z"] >= geometry.hot_half_length_m - tol, "boundary"] = "hot_inlet"
    result.loc[result["z"] <= -geometry.hot_half_length_m + tol, "boundary"] = "hot_outlet"
    return result


def _label_cold_boundaries(frame: pd.DataFrame, geometry: ThreeDGeometry) -> pd.DataFrame:
    result = frame.copy()
    result["boundary"] = "cold_surface"
    tol = geometry.surface_tolerance_m
    result.loc[result["z"] <= -geometry.cold_half_length_m + tol, "boundary"] = "cold_inlet"
    result.loc[result["z"] >= geometry.cold_half_length_m - tol, "boundary"] = "cold_outlet"
    inner_mask = result["r"] <= geometry.cold_inner_radius_m + tol
    outer_mask = result["r"] >= geometry.cold_outer_radius_m - tol
    result.loc[(result["boundary"] == "cold_surface") & inner_mask, "boundary"] = "cold_inner_wall"
    result.loc[(result["boundary"] == "cold_surface") & outer_mask, "boundary"] = "cold_outer_wall"
    return result


def _deterministic_group_split(frame: pd.DataFrame, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["boundary", "z", "phi_rad", "r"]).reset_index(drop=True)
    ordered["split"] = "train"
    validation_indices: list[int] = []
    for _, group in ordered.groupby("boundary", sort=False):
        if len(group) <= 4:
            continue
        n_val = max(1, int(round(len(group) * validation_ratio)))
        positions = np.linspace(0, len(group) - 1, n_val, dtype=int)
        validation_indices.extend(group.iloc[positions].index.tolist())
    ordered.loc[validation_indices, "split"] = "validation"
    train = ordered[ordered["split"] == "train"].reset_index(drop=True)
    validation = ordered[ordered["split"] == "validation"].reset_index(drop=True)
    return train, validation


def _boundary_temperature_stats(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for boundary, group in frame.groupby("boundary", sort=False):
        summary[str(boundary)] = {
            "count": int(len(group)),
            "mean_K": float(group["T"].mean()),
            "std_K": float(group["T"].std(ddof=0)),
            "min_K": float(group["T"].min()),
            "max_K": float(group["T"].max()),
        }
    return summary


def build_3d_case_artifacts(config: dict) -> dict:
    if uses_synthetic_3d_data(config):
        return build_synthetic_3d_case_artifacts(config)

    geometry = geometry3d_from_config(config)
    solution_path = Path(config["paths"]["solution_csv"])
    solution = _polarize(read_temperature_csv(solution_path))

    hot_surface = _label_hot_boundaries(solution[solution["r"] <= geometry.hot_radius_m + geometry.surface_tolerance_m], geometry)
    cold_surface = _label_cold_boundaries(
        solution[
            (solution["r"] >= geometry.cold_inner_radius_m - geometry.surface_tolerance_m)
            & (solution["r"] <= geometry.cold_outer_radius_m + geometry.surface_tolerance_m)
            & (solution["z"].abs() <= geometry.cold_half_length_m + geometry.surface_tolerance_m)
        ],
        geometry,
    )

    split_ratio = float(config["preprocessing_3d"]["validation_ratio"])
    hot_train, hot_val = _deterministic_group_split(hot_surface, split_ratio)
    cold_train, cold_val = _deterministic_group_split(cold_surface, split_ratio)

    summary = {
        "solution_csv": asdict(summarize_temperature_csv(solution_path)),
        "geometry_3d": {
            "hot_half_length_m": geometry.hot_half_length_m,
            "cold_half_length_m": geometry.cold_half_length_m,
            "hot_radius_m": geometry.hot_radius_m,
            "cold_inner_radius_m": geometry.cold_inner_radius_m,
            "cold_outer_radius_m": geometry.cold_outer_radius_m,
        },
        "surface_counts": {
            "hot_total": int(len(hot_surface)),
            "cold_total": int(len(cold_surface)),
            "hot_train": int(len(hot_train)),
            "hot_validation": int(len(hot_val)),
            "cold_train": int(len(cold_train)),
            "cold_validation": int(len(cold_val)),
        },
        "hot_boundary_counts": hot_surface["boundary"].value_counts().to_dict(),
        "cold_boundary_counts": cold_surface["boundary"].value_counts().to_dict(),
        "hot_boundary_temperature_stats": _boundary_temperature_stats(hot_surface),
        "cold_boundary_temperature_stats": _boundary_temperature_stats(cold_surface),
        "observed_radii_m": {
            "hot_min": float(hot_surface["r"].min()),
            "hot_max": float(hot_surface["r"].max()),
            "cold_min": float(cold_surface["r"].min()),
            "cold_max": float(cold_surface["r"].max()),
        },
    }
    return {
        "hot_surface": hot_surface,
        "cold_surface": cold_surface,
        "hot_train": hot_train,
        "hot_validation": hot_val,
        "cold_train": cold_train,
        "cold_validation": cold_val,
        "summary": summary,
    }


def save_3d_case_artifacts(artifacts: dict, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts["hot_surface"].to_csv(output_path / "hot_surface_points.csv", index=False)
    artifacts["cold_surface"].to_csv(output_path / "cold_surface_points.csv", index=False)
    artifacts["hot_train"].to_csv(output_path / "hot_surface_train.csv", index=False)
    artifacts["hot_validation"].to_csv(output_path / "hot_surface_validation.csv", index=False)
    artifacts["cold_train"].to_csv(output_path / "cold_surface_train.csv", index=False)
    artifacts["cold_validation"].to_csv(output_path / "cold_surface_validation.csv", index=False)
    with (output_path / "analysis_3d_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts["summary"], handle, indent=2)
