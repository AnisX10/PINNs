from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

from pinn_hex.physics.double_pipe import Geometry, OperatingPoint


def _stream_grid(stream: str, geometry: Geometry, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    s_flow = np.linspace(0.0, 1.0, n_points)
    if stream == "hot":
        z_grid = geometry.hot_half_length_m - s_flow * (2.0 * geometry.hot_half_length_m)
    elif stream == "cold":
        z_grid = -geometry.cold_half_length_m + s_flow * (2.0 * geometry.cold_half_length_m)
    else:
        raise ValueError(f"Unsupported stream {stream!r}.")
    return s_flow, z_grid


def _sign_changes(values: np.ndarray) -> int:
    delta = np.diff(values)
    sign = np.sign(delta)
    sign = sign[sign != 0.0]
    if sign.size <= 1:
        return 0
    return int(np.sum(sign[1:] * sign[:-1] < 0.0))


def _should_denoise(interpolated: np.ndarray, cfg: dict) -> bool:
    mode = str(cfg["denoise_mode"]).lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    threshold = int(cfg["denoise_sign_change_threshold"])
    return _sign_changes(interpolated) > threshold


def _denoise(values: np.ndarray, cfg: dict) -> tuple[np.ndarray, bool]:
    if not _should_denoise(values, cfg):
        return values.copy(), False
    if str(cfg["denoise_method"]).lower() != "savgol":
        raise ValueError(f"Unsupported denoise method: {cfg['denoise_method']}")
    window = int(cfg["denoise_window"])
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) - (1 - len(values) % 2))
    if window < 5:
        return values.copy(), False
    polyorder = min(int(cfg["denoise_polyorder"]), window - 2)
    smoothed = savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed, True


def make_processed_stream_profile(
    profile: pd.DataFrame,
    stream: str,
    geometry: Geometry,
    operating: OperatingPoint,
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    n_points = int(cfg[f"{stream}_uniform_points"])
    s_target, z_target = _stream_grid(stream, geometry, n_points)
    base = profile.sort_values("s_flow").drop_duplicates(subset=["s_flow"]).reset_index(drop=True)
    if float(base["s_flow"].iloc[0]) > 0.0:
        start = base.iloc[[0]].copy()
        start.loc[:, "s_flow"] = 0.0
        base = pd.concat([start, base], ignore_index=True)
    if float(base["s_flow"].iloc[-1]) < 1.0:
        end = base.iloc[[-1]].copy()
        end.loc[:, "s_flow"] = 1.0
        base = pd.concat([base, end], ignore_index=True)
    interpolator = PchipInterpolator(base["s_flow"], base["T_mean"], extrapolate=False)
    t_interp = interpolator(s_target)
    std_interp = np.interp(s_target, base["s_flow"], base["T_std"].fillna(0.0))
    n_interp = np.interp(s_target, base["s_flow"], base["n"])
    t_denoised, denoised = _denoise(t_interp, cfg)

    delta_t = operating.hot_inlet_temperature_K - operating.cold_inlet_temperature_K
    processed = pd.DataFrame(
        {
            "stream": stream,
            "index": np.arange(n_points, dtype=int),
            "s_flow": s_target,
            "s_centered": 2.0 * s_target - 1.0,
            "z": z_target,
            "z_norm_global": z_target / geometry.hot_half_length_m,
            "T_interp_K": t_interp,
            "T_processed_K": t_denoised,
            "T_std_interp_K": std_interp,
            "n_interp": n_interp,
            "theta": (t_denoised - operating.cold_inlet_temperature_K) / delta_t,
            "theta_centered": (t_denoised - operating.initial_temperature_K) / delta_t,
            "was_denoised": denoised,
        }
    )
    supervision = t_denoised.copy()
    supervision_mode = "processed"
    if stream == "hot" and bool(cfg.get("hot_piecewise_monotonic_supervision", False)):
        split_z = -geometry.cold_half_length_m
        extension_mask = z_target <= split_z
        overlap_mask = z_target > split_z
        if np.sum(extension_mask) > 1:
            extension_fit = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
                z_target[extension_mask], supervision[extension_mask]
            )
            supervision[extension_mask] = extension_fit
        if np.sum(overlap_mask) > 1:
            overlap_fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
                z_target[overlap_mask], supervision[overlap_mask]
            )
            supervision[overlap_mask] = overlap_fit
        supervision_mode = "piecewise_monotonic_hot"
    if stream == "cold" and bool(cfg.get("cold_monotonic_supervision", False)):
        sample_weight = np.ones_like(supervision)
        endpoint_weight = float(cfg.get("cold_monotonic_endpoint_weight", 1.0))
        sample_weight[0] = endpoint_weight
        sample_weight[-1] = endpoint_weight
        supervision = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
            z_target,
            supervision,
            sample_weight=sample_weight,
        )
        supervision_mode = "anchored_monotonic_cold"
    processed["T_supervision_K"] = supervision
    processed["theta_supervision"] = (supervision - operating.cold_inlet_temperature_K) / delta_t
    if stream == "hot":
        processed["z_norm_stream"] = z_target / geometry.hot_half_length_m
    else:
        processed["z_norm_stream"] = z_target / geometry.cold_half_length_m
    summary = {
        "stream": stream,
        "input_points": int(len(base)),
        "output_points": int(n_points),
        "interpolation": str(cfg["interpolation"]),
        "denoise_mode": str(cfg["denoise_mode"]),
        "denoise_applied": bool(denoised),
        "sign_changes_interpolated": _sign_changes(t_interp),
        "sign_changes_processed": _sign_changes(t_denoised),
        "sign_changes_supervision": _sign_changes(supervision),
        "supervision_mode": supervision_mode,
        "temperature_min_K": float(processed["T_processed_K"].min()),
        "temperature_max_K": float(processed["T_processed_K"].max()),
    }
    return processed, summary


def _even_validation_indices(n: int, validation_ratio: float, keep_boundaries_in_train: bool) -> np.ndarray:
    all_indices = np.arange(n, dtype=int)
    if keep_boundaries_in_train and n >= 2:
        candidates = all_indices[1:-1]
    else:
        candidates = all_indices
    n_val = max(1, int(round(len(candidates) * validation_ratio)))
    positions = np.linspace(0, len(candidates) - 1, n_val, dtype=int)
    return np.unique(candidates[positions])


def split_stream_profile(profile: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    val_indices = _even_validation_indices(
        n=len(profile),
        validation_ratio=float(cfg["validation_ratio"]),
        keep_boundaries_in_train=bool(cfg["keep_boundaries_in_train"]),
    )
    split = profile.copy()
    split["split"] = "train"
    split.loc[split["index"].isin(val_indices), "split"] = "validation"
    train = split[split["split"] == "train"].reset_index(drop=True)
    val = split[split["split"] == "validation"].reset_index(drop=True)
    summary = {
        "stream": str(profile["stream"].iloc[0]),
        "split_strategy": "deterministic_even_holdout",
        "train_points": int(len(train)),
        "validation_points": int(len(val)),
        "keep_boundaries_in_train": bool(cfg["keep_boundaries_in_train"]),
        "validation_ratio_target": float(cfg["validation_ratio"]),
    }
    return train, val, summary


def build_preprocessed_supervision(
    hot_profile: pd.DataFrame,
    cold_profile: pd.DataFrame,
    geometry: Geometry,
    operating: OperatingPoint,
    cfg: dict,
) -> dict:
    hot_processed, hot_summary = make_processed_stream_profile(hot_profile, "hot", geometry, operating, cfg)
    cold_processed, cold_summary = make_processed_stream_profile(cold_profile, "cold", geometry, operating, cfg)
    hot_train, hot_val, hot_split_summary = split_stream_profile(hot_processed, cfg)
    cold_train, cold_val, cold_split_summary = split_stream_profile(cold_processed, cfg)

    combined_train = pd.concat([hot_train, cold_train], ignore_index=True)
    combined_val = pd.concat([hot_val, cold_val], ignore_index=True)
    scalers = {
        "temperature": {
            "kind": str(cfg["normalization_temperature_reference"]),
            "reference_cold_inlet_K": float(operating.cold_inlet_temperature_K),
            "reference_initial_K": float(operating.initial_temperature_K),
            "scale_delta_T_K": float(operating.hot_inlet_temperature_K - operating.cold_inlet_temperature_K),
        },
        "coordinate": {
            "kind": str(cfg["normalization_coordinate_reference"]),
            "hot_half_length_m": float(geometry.hot_half_length_m),
            "cold_half_length_m": float(geometry.cold_half_length_m),
        },
    }
    return {
        "hot_processed": hot_processed,
        "cold_processed": cold_processed,
        "hot_train": hot_train,
        "hot_val": hot_val,
        "cold_train": cold_train,
        "cold_val": cold_val,
        "train": combined_train,
        "validation": combined_val,
        "summary": {
            "hot_processing": hot_summary,
            "cold_processing": cold_summary,
            "hot_split": hot_split_summary,
            "cold_split": cold_split_summary,
            "scalers": scalers,
        },
    }


def save_preprocessed_supervision(preprocessed: dict, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    preprocessed["hot_processed"].to_csv(output_path / "hot_profile_processed.csv", index=False)
    preprocessed["cold_processed"].to_csv(output_path / "cold_profile_processed.csv", index=False)
    preprocessed["train"].to_csv(output_path / "supervision_train.csv", index=False)
    preprocessed["validation"].to_csv(output_path / "supervision_validation.csv", index=False)

    np.savez(
        output_path / "supervision_tensors.npz",
        hot_train_z=preprocessed["hot_train"]["z"].to_numpy(),
        hot_train_theta=preprocessed["hot_train"]["theta"].to_numpy(),
        hot_val_z=preprocessed["hot_val"]["z"].to_numpy(),
        hot_val_theta=preprocessed["hot_val"]["theta"].to_numpy(),
        cold_train_z=preprocessed["cold_train"]["z"].to_numpy(),
        cold_train_theta=preprocessed["cold_train"]["theta"].to_numpy(),
        cold_val_z=preprocessed["cold_val"]["z"].to_numpy(),
        cold_val_theta=preprocessed["cold_val"]["theta"].to_numpy(),
    )
    with (output_path / "preprocessing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessed["summary"], handle, indent=2)
