from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    studio_path = root / "app" / "boundary_studio.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(studio_path),
        "--browser.gatherUsageStats",
        "false",
    ]
    completed = subprocess.run(command, cwd=root)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
