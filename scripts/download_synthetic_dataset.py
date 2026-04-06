from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_hex.data.synthetic import (
    SYNTHETIC_DATASET_FOLDER_ID,
    download_synthetic_dataset,
    save_download_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the shared synthetic boundary-case dataset from Google Drive.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "synthetic" / "synthetic_comsol_pinn_dataset"),
        help="Local directory where the dataset should be stored.",
    )
    parser.add_argument(
        "--folder-id",
        default=SYNTHETIC_DATASET_FOLDER_ID,
        help="Google Drive folder id for the shared dataset.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=["case_001"],
        help="Optional case ids to download. Defaults to the baseline case_001.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Download every case folder listed in the shared Drive dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_ids = None if args.all_cases else list(args.case_ids)
    summary = download_synthetic_dataset(
        output_dir=args.output_dir,
        folder_id=args.folder_id,
        case_ids=case_ids,
        overwrite=bool(args.overwrite),
    )
    summary_path = save_download_summary(summary, args.output_dir)
    print(f"Downloaded synthetic dataset to: {Path(args.output_dir).resolve()}")
    print(f"Saved download summary to: {summary_path.resolve()}")
    for case_id, case_summary in summary["cases"].items():
        missing = case_summary["missing_expected_files"]
        if missing:
            print(f"{case_id}: missing expected files in shared folder: {', '.join(missing)}")


if __name__ == "__main__":
    main()
