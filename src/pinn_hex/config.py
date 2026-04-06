from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise KeyError(f"Unknown config section '{key}'.")
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return config
