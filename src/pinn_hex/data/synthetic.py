from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import requests

SYNTHETIC_DATASET_FOLDER_ID = "15BdOSbvnwOpUZYuDWLjArWPNAZ5eFDND"
SYNTHETIC_REQUIRED_CASE_FILES = (
    "globals.csv",
    "hot_inlet.csv",
    "hot_outlet.csv",
    "hot_wall_interface.csv",
    "cold_inlet.csv",
    "cold_outlet.csv",
    "wall_cold_interface.csv",
)
OPERATING_COLUMNS = ("Th_in_K", "Tc_in_K", "uh_in_mps", "uc_in_mps")


@dataclass(frozen=True)
class DriveEntry:
    entry_id: str
    name: str
    url: str
    is_folder: bool


def list_drive_folder(folder_id: str, timeout_s: float = 30.0) -> list[DriveEntry]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    pattern = re.compile(
        r'<div class="flip-entry" id="entry-([^"]+)".*?<a href="([^"]+)"[^>]*>.*?<div class="flip-entry-title">([^<]+)</div>',
        re.S,
    )
    entries: list[DriveEntry] = []
    for entry_id, entry_url, name in pattern.findall(response.text):
        entries.append(
            DriveEntry(
                entry_id=entry_id,
                name=name,
                url=entry_url,
                is_folder="/drive/folders/" in entry_url,
            )
        )
    return entries


def download_drive_file(file_id: str, destination: str | Path, timeout_s: float = 60.0) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=timeout_s,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def download_synthetic_dataset(
    output_dir: str | Path,
    folder_id: str = SYNTHETIC_DATASET_FOLDER_ID,
    case_ids: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    root_entries = list_drive_folder(folder_id)
    root_files = [entry for entry in root_entries if not entry.is_folder]
    case_folders = {entry.name: entry for entry in root_entries if entry.is_folder}
    selected_case_ids = sorted(case_ids) if case_ids else sorted(case_folders.keys())

    downloaded: list[str] = []
    for entry in root_files:
        destination = output_path / entry.name
        if overwrite or not destination.exists():
            download_drive_file(entry.entry_id, destination)
        downloaded.append(str(destination))

    missing_cases = [case_id for case_id in selected_case_ids if case_id not in case_folders]
    if missing_cases:
        raise FileNotFoundError(f"Unknown synthetic case ids: {missing_cases}")

    case_summary: dict[str, dict[str, object]] = {}
    for case_id in selected_case_ids:
        case_entry = case_folders[case_id]
        case_entries = list_drive_folder(case_entry.entry_id)
        case_dir = output_path / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        file_names = sorted(entry.name for entry in case_entries if not entry.is_folder)
        for entry in case_entries:
            if entry.is_folder:
                continue
            destination = case_dir / entry.name
            if overwrite or not destination.exists():
                download_drive_file(entry.entry_id, destination)
            downloaded.append(str(destination))
        case_summary[case_id] = {
            "downloaded_files": file_names,
            "missing_expected_files": sorted(set(SYNTHETIC_REQUIRED_CASE_FILES) - set(file_names)),
        }

    return {
        "root_folder_id": folder_id,
        "output_dir": str(output_path),
        "cases": case_summary,
        "downloaded_paths": downloaded,
    }


def synthetic_case_dir_from_config(config: dict) -> Path | None:
    paths = config.get("paths", {})
    if paths.get("synthetic_case_dir"):
        return Path(paths["synthetic_case_dir"])
    if paths.get("synthetic_root_dir") and config.get("synthetic_3d", {}).get("case_id"):
        return Path(paths["synthetic_root_dir"]) / str(config["synthetic_3d"]["case_id"])
    return None


def synthetic_case_ids_from_config(config: dict) -> list[str]:
    synthetic_cfg = config.get("synthetic_3d", {})
    train_case_ids = synthetic_cfg.get("train_case_ids")
    validation_case_ids = synthetic_cfg.get("validation_case_ids")
    if train_case_ids or validation_case_ids:
        combined = [str(case_id) for case_id in (train_case_ids or [])]
        combined.extend(str(case_id) for case_id in (validation_case_ids or []))
        return list(dict.fromkeys(combined))
    case_ids = synthetic_cfg.get("case_ids")
    if case_ids:
        return [str(case_id) for case_id in case_ids]
    if synthetic_cfg.get("case_id"):
        return [str(synthetic_cfg["case_id"])]
    case_dir = synthetic_case_dir_from_config(config)
    if case_dir is not None:
        return [case_dir.name]
    return []


def synthetic_case_dirs_from_config(config: dict) -> list[Path]:
    paths = config.get("paths", {})
    case_dir = synthetic_case_dir_from_config(config)
    if case_dir is not None and not synthetic_case_ids_from_config(config):
        return [case_dir]
    root_dir = paths.get("synthetic_root_dir")
    if root_dir and synthetic_case_ids_from_config(config):
        return [Path(root_dir) / case_id for case_id in synthetic_case_ids_from_config(config)]
    if case_dir is not None:
        return [case_dir]
    return []


def uses_synthetic_3d_data(config: dict) -> bool:
    return bool(synthetic_case_dirs_from_config(config))


def synthetic_case_split_ids_from_config(config: dict) -> tuple[list[str], list[str]]:
    synthetic_cfg = config.get("synthetic_3d", {})
    train_case_ids = [str(case_id) for case_id in synthetic_cfg.get("train_case_ids", [])]
    validation_case_ids = [str(case_id) for case_id in synthetic_cfg.get("validation_case_ids", [])]
    return train_case_ids, validation_case_ids


def _polarize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["r"] = np.sqrt(result["x"] ** 2 + result["y"] ** 2)
    result["phi_rad"] = np.arctan2(result["y"], result["x"])
    result["phi_cos"] = np.cos(result["phi_rad"])
    result["phi_sin"] = np.sin(result["phi_rad"])
    return result


def _deterministic_group_split(frame: pd.DataFrame, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_columns = [column for column in ["case_id", "boundary", "z", "phi_rad", "r"] if column in frame.columns]
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    ordered["split"] = "train"
    validation_indices: list[int] = []
    group_columns = [column for column in ["case_id", "boundary"] if column in ordered.columns]
    for _, group in ordered.groupby(group_columns, sort=False):
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


def _split_by_case_holdout(
    frame: pd.DataFrame,
    train_case_ids: list[str],
    validation_case_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "case_id" not in frame.columns:
        raise ValueError("Case-holdout splitting requires a 'case_id' column.")
    available_case_ids = [str(case_id) for case_id in frame["case_id"].drop_duplicates().tolist()]
    available_case_id_set = set(available_case_ids)
    train_case_id_set = set(train_case_ids)
    validation_case_id_set = set(validation_case_ids)
    overlap = train_case_id_set & validation_case_id_set
    if overlap:
        raise ValueError(f"Train/validation case split overlaps: {sorted(overlap)}")
    missing_train = sorted(train_case_id_set - available_case_id_set)
    missing_validation = sorted(validation_case_id_set - available_case_id_set)
    if missing_train:
        raise ValueError(f"Unknown train_case_ids in synthetic config: {missing_train}")
    if missing_validation:
        raise ValueError(f"Unknown validation_case_ids in synthetic config: {missing_validation}")

    if train_case_ids and validation_case_ids:
        assigned_case_ids = train_case_id_set | validation_case_id_set
        unassigned = sorted(available_case_id_set - assigned_case_ids)
        if unassigned:
            raise ValueError(
                "Synthetic case-holdout split left unassigned cases. "
                f"Specify all selected cases explicitly or omit one side of the split: {unassigned}"
            )
    elif validation_case_ids:
        train_case_id_set = available_case_id_set - validation_case_id_set
    elif train_case_ids:
        validation_case_id_set = available_case_id_set - train_case_id_set
    else:
        raise ValueError("Case-holdout split requested without train_case_ids or validation_case_ids.")

    ordered = frame.sort_values(["case_id", "boundary", "z", "phi_rad", "r"]).reset_index(drop=True)
    train = ordered[ordered["case_id"].isin(sorted(train_case_id_set))].reset_index(drop=True)
    validation = ordered[ordered["case_id"].isin(sorted(validation_case_id_set))].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Case-holdout split produced an empty train or validation set.")
    return train, validation


def _load_surface_frame(path: Path, boundary: str, flip_z_axis: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    result = frame.copy()
    result["boundary"] = boundary
    result["source_file"] = path.name
    if flip_z_axis:
        result["z"] = -result["z"]
        if "w" in result.columns:
            result["w"] = -result["w"]
        if "nz" in result.columns:
            result["nz"] = -result["nz"]
    return _polarize(result)


def _read_case_globals(case_dir: Path) -> dict[str, float | str]:
    globals_path = case_dir / "globals.csv"
    frame = pd.read_csv(globals_path)
    if frame.empty:
        raise ValueError(f"Synthetic globals file is empty: {globals_path}")
    row = frame.iloc[0].to_dict()
    return {str(key): value for key, value in row.items()}


def _attach_case_metadata(frame: pd.DataFrame, case_id: str, globals_row: dict[str, float | str]) -> pd.DataFrame:
    result = frame.copy()
    result["case_id"] = case_id
    for column, value in globals_row.items():
        if column == "case_id":
            continue
        result[column] = value
    result["temperature_span_K"] = float(globals_row["Th_in_K"]) - float(globals_row["Tc_in_K"])
    return result


def _derive_geometry(case_dir: Path, flip_z_axis: bool, surface_tolerance_m: float) -> dict[str, float]:
    hot_inlet = _load_surface_frame(case_dir / "hot_inlet.csv", "hot_inlet", flip_z_axis)
    hot_outlet = _load_surface_frame(case_dir / "hot_outlet.csv", "hot_outlet", flip_z_axis)
    hot_wall = _load_surface_frame(case_dir / "hot_wall_interface.csv", "hot_wall", flip_z_axis)
    cold_inlet = _load_surface_frame(case_dir / "cold_inlet.csv", "cold_inlet", flip_z_axis)
    cold_outlet = _load_surface_frame(case_dir / "cold_outlet.csv", "cold_outlet", flip_z_axis)
    cold_inner = _load_surface_frame(case_dir / "wall_cold_interface.csv", "cold_inner_wall", flip_z_axis)

    hot_half_length_m = float(max(abs(hot_inlet["z"].iloc[0]), abs(hot_outlet["z"].iloc[0]), hot_wall["z"].abs().max()))
    cold_half_length_m = float(
        max(abs(cold_inlet["z"].iloc[0]), abs(cold_outlet["z"].iloc[0]), cold_inner["z"].abs().max())
    )
    return {
        "hot_half_length_m": hot_half_length_m,
        "cold_half_length_m": cold_half_length_m,
        "hot_radius_m": float(hot_wall["r"].mean()),
        "cold_inner_radius_m": float(cold_inner["r"].mean()),
        "cold_outer_radius_m": float(max(cold_inlet["r"].max(), cold_outlet["r"].max())),
        "surface_tolerance_m": surface_tolerance_m,
    }


def resolve_synthetic_3d_config(config: dict) -> dict:
    if not uses_synthetic_3d_data(config):
        return config

    resolved = deepcopy(config)
    case_dirs = synthetic_case_dirs_from_config(resolved)
    if not case_dirs:
        return resolved
    missing_dirs = [case_dir for case_dir in case_dirs if not case_dir.exists()]
    if missing_dirs:
        missing = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(
            f"Synthetic case directory does not exist: {missing}. "
            f"Download it first with scripts/download_synthetic_dataset.py."
        )

    synthetic_cfg = resolved.setdefault("synthetic_3d", {})
    flip_z_axis = bool(synthetic_cfg.get("flip_z_axis", True))
    surface_tolerance_m = float(resolved.get("geometry_3d", {}).get("surface_tolerance_m", 1.0e-4))
    geometry_rows = [_derive_geometry(case_dir, flip_z_axis, surface_tolerance_m) for case_dir in case_dirs]
    globals_rows = [_read_case_globals(case_dir) for case_dir in case_dirs]
    geometry_section = resolved.setdefault("geometry_3d", {})
    geometry_section.update(
        {
            "hot_half_length_m": float(np.mean([row["hot_half_length_m"] for row in geometry_rows])),
            "cold_half_length_m": float(np.mean([row["cold_half_length_m"] for row in geometry_rows])),
            "hot_radius_m": float(np.mean([row["hot_radius_m"] for row in geometry_rows])),
            "cold_inner_radius_m": float(np.mean([row["cold_inner_radius_m"] for row in geometry_rows])),
            "cold_outer_radius_m": float(np.mean([row["cold_outer_radius_m"] for row in geometry_rows])),
            "surface_tolerance_m": surface_tolerance_m,
        }
    )

    reference = resolved.setdefault("reference_conditions", {})
    reference["hot_inlet_temperature_K"] = float(np.mean([float(row["Th_in_K"]) for row in globals_rows]))
    reference["cold_inlet_temperature_K"] = float(np.mean([float(row["Tc_in_K"]) for row in globals_rows]))
    reference["inlet_velocity_hot_m_per_s"] = float(np.mean([float(row["uh_in_mps"]) for row in globals_rows]))
    reference["inlet_velocity_cold_m_per_s"] = float(np.mean([float(row["uc_in_mps"]) for row in globals_rows]))
    reference.setdefault("initial_temperature_K", 293.15)

    paths = resolved.setdefault("paths", {})
    if len(case_dirs) == 1:
        paths["synthetic_case_dir"] = str(case_dirs[0])
        synthetic_cfg.setdefault("case_id", case_dirs[0].name)
    else:
        paths.pop("synthetic_case_dir", None)
        synthetic_cfg["case_ids"] = [case_dir.name for case_dir in case_dirs]
    return resolved


def build_synthetic_3d_case_artifacts(config: dict) -> dict:
    case_dirs = synthetic_case_dirs_from_config(config)
    if not case_dirs:
        raise ValueError("Synthetic 3D case directory was not configured.")
    missing_dirs = [case_dir for case_dir in case_dirs if not case_dir.exists()]
    if missing_dirs:
        missing = ", ".join(str(path) for path in missing_dirs)
        raise FileNotFoundError(f"Synthetic case directory does not exist: {missing}")

    synthetic_cfg = config.get("synthetic_3d", {})
    flip_z_axis = bool(synthetic_cfg.get("flip_z_axis", True))
    hot_frames: list[pd.DataFrame] = []
    cold_frames: list[pd.DataFrame] = []
    per_case_summary: dict[str, dict[str, object]] = {}
    for case_dir in case_dirs:
        case_id = case_dir.name
        globals_row = _read_case_globals(case_dir)
        hot_case = pd.concat(
            [
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "hot_inlet.csv", "hot_inlet", flip_z_axis),
                    case_id,
                    globals_row,
                ),
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "hot_outlet.csv", "hot_outlet", flip_z_axis),
                    case_id,
                    globals_row,
                ),
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "hot_wall_interface.csv", "hot_wall", flip_z_axis),
                    case_id,
                    globals_row,
                ),
            ],
            ignore_index=True,
        )
        cold_case = pd.concat(
            [
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "cold_inlet.csv", "cold_inlet", flip_z_axis),
                    case_id,
                    globals_row,
                ),
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "cold_outlet.csv", "cold_outlet", flip_z_axis),
                    case_id,
                    globals_row,
                ),
                _attach_case_metadata(
                    _load_surface_frame(case_dir / "wall_cold_interface.csv", "cold_inner_wall", flip_z_axis),
                    case_id,
                    globals_row,
                ),
            ],
            ignore_index=True,
        )
        hot_frames.append(hot_case)
        cold_frames.append(cold_case)
        available_case_files = sorted(path.name for path in case_dir.glob("*.csv"))
        per_case_summary[case_id] = {
            "case_dir": str(case_dir),
            "available_case_files": available_case_files,
            "missing_expected_files": sorted(set(SYNTHETIC_REQUIRED_CASE_FILES) - set(available_case_files)),
            "globals": {
                key: float(value) if isinstance(value, (np.floating, float, int, np.integer)) else value
                for key, value in globals_row.items()
            },
            "hot_surface_count": int(len(hot_case)),
            "cold_surface_count": int(len(cold_case)),
        }

    hot_surface = pd.concat(hot_frames, ignore_index=True)
    cold_surface = pd.concat(cold_frames, ignore_index=True)

    train_case_ids, validation_case_ids = synthetic_case_split_ids_from_config(config)
    if train_case_ids or validation_case_ids:
        split_strategy = "case_holdout"
        hot_train, hot_val = _split_by_case_holdout(hot_surface, train_case_ids, validation_case_ids)
        cold_train, cold_val = _split_by_case_holdout(cold_surface, train_case_ids, validation_case_ids)
    else:
        split_strategy = "point_holdout"
        split_ratio = float(config["preprocessing_3d"]["validation_ratio"])
        hot_train, hot_val = _deterministic_group_split(hot_surface, split_ratio)
        cold_train, cold_val = _deterministic_group_split(cold_surface, split_ratio)

    train_case_ids_resolved = [str(case_id) for case_id in hot_train["case_id"].drop_duplicates().tolist()]
    validation_case_ids_resolved = [str(case_id) for case_id in hot_val["case_id"].drop_duplicates().tolist()]
    summary = {
        "source_format": "synthetic_boundary_case",
        "case_ids": [case_dir.name for case_dir in case_dirs],
        "case_id": str(synthetic_cfg.get("case_id", case_dirs[0].name)),
        "flip_z_axis": flip_z_axis,
        "split_strategy": split_strategy,
        "train_case_ids": train_case_ids_resolved,
        "validation_case_ids": validation_case_ids_resolved,
        "conditioning_columns": list(OPERATING_COLUMNS),
        "per_case": per_case_summary,
        "surface_counts": {
            "hot_total": int(len(hot_surface)),
            "cold_total": int(len(cold_surface)),
            "hot_train": int(len(hot_train)),
            "hot_validation": int(len(hot_val)),
            "cold_train": int(len(cold_train)),
            "cold_validation": int(len(cold_val)),
        },
        "hot_boundary_counts": {str(key): int(value) for key, value in hot_surface["boundary"].value_counts().items()},
        "cold_boundary_counts": {str(key): int(value) for key, value in cold_surface["boundary"].value_counts().items()},
        "hot_boundary_temperature_stats": _boundary_temperature_stats(hot_surface),
        "cold_boundary_temperature_stats": _boundary_temperature_stats(cold_surface),
        "observed_geometry": {
            "hot_half_length_m": float(hot_surface["z"].abs().max()),
            "cold_half_length_m": float(cold_surface["z"].abs().max()),
            "hot_radius_m": float(hot_surface.loc[hot_surface["boundary"] == "hot_wall", "r"].mean()),
            "cold_inner_radius_m": float(
                cold_surface.loc[cold_surface["boundary"] == "cold_inner_wall", "r"].mean()
            ),
            "cold_outer_radius_m": float(cold_surface["r"].max()),
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


def save_download_summary(summary: dict[str, object], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path
