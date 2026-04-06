from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def load_temperature_calibration(path: str | Path) -> dict[str, object]:
    calibration_path = Path(path)
    with calibration_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "corrections_K" not in payload or not isinstance(payload["corrections_K"], dict):
        raise ValueError(f"Temperature calibration file is missing a valid corrections_K mapping: {calibration_path}")
    return payload


def apply_temperature_calibration(
    frame: pd.DataFrame,
    calibration: dict[str, object],
    prediction_column: str,
    correction_column: str = "T_calibration_K",
) -> pd.DataFrame:
    corrections = {str(key): float(value) for key, value in dict(calibration["corrections_K"]).items()}
    result = frame.copy()
    result[correction_column] = result["boundary"].map(corrections).fillna(0.0).astype(float)
    result[prediction_column] = result[prediction_column].astype(float) + result[correction_column]
    return result


def compute_boundary_bias_calibration(
    frame: pd.DataFrame,
    prediction_column: str = "T_pred",
    target_column: str = "T",
    boundaries: list[str] | None = None,
) -> dict[str, float]:
    subset = frame
    if boundaries:
        subset = subset[subset["boundary"].isin(boundaries)]
    corrections: dict[str, float] = {}
    for boundary, group in subset.groupby("boundary", sort=False):
        corrections[str(boundary)] = float((group[target_column].astype(float) - group[prediction_column].astype(float)).mean())
    return corrections
