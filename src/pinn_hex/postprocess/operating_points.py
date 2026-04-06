from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_operating_point(
    config: dict,
    case_id: str | None,
    Th_in: float | None,
    Tc_in: float | None,
    uh_in: float | None,
    uc_in: float | None,
) -> dict[str, float | str]:
    if case_id:
        root_dir = Path(config["paths"]["synthetic_root_dir"])
        manifest_path = root_dir / "case_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            match = manifest[manifest["case_id"] == case_id]
            if match.empty:
                raise ValueError(f"Unknown synthetic case id in manifest: {case_id}")
            row = match.iloc[0]
            return {
                "case_id": str(case_id),
                "Th_in_K": float(row["Th_in_K"]),
                "Tc_in_K": float(row["Tc_in_K"]),
                "uh_in_mps": float(row["uh_in_mps"]),
                "uc_in_mps": float(row["uc_in_mps"]),
            }
        globals_path = root_dir / str(case_id) / "globals.csv"
        if globals_path.exists():
            globals_row = pd.read_csv(globals_path).iloc[0]
            return {
                "case_id": str(case_id),
                "Th_in_K": float(globals_row["Th_in_K"]),
                "Tc_in_K": float(globals_row["Tc_in_K"]),
                "uh_in_mps": float(globals_row["uh_in_mps"]),
                "uc_in_mps": float(globals_row["uc_in_mps"]),
            }
        raise FileNotFoundError(f"Could not resolve operating point for case '{case_id}'.")

    if None in {Th_in, Tc_in, uh_in, uc_in}:
        raise ValueError("Provide either --case-id or all of --Th-in --Tc-in --uh-in --uc-in.")
    return {
        "case_id": "custom",
        "Th_in_K": float(Th_in),
        "Tc_in_K": float(Tc_in),
        "uh_in_mps": float(uh_in),
        "uc_in_mps": float(uc_in),
    }
