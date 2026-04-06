from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "case_matrix" / "comsol_case_matrix_dataset"

DEFAULT_CV_SUMMARY = ROOT / "outputs_3d_case_matrix_conditioned_case_cv_final" / "case_cv_summary.json"
DEFAULT_CV_CASES = ROOT / "outputs_3d_case_matrix_conditioned_case_cv_final" / "case_cv_case_summary.csv"
DEFAULT_VALIDATION_SUMMARY = (
    ROOT
    / "outputs_3d_case_matrix_final_validation_optphys2_10ep_dpcal_walltune_tempcal"
    / "final_validation_summary.json"
)
DEFAULT_VALIDATION_CASES = (
    ROOT
    / "outputs_3d_case_matrix_final_validation_optphys2_10ep_dpcal_walltune_tempcal"
    / "final_validation_case_summary.csv"
)
DEFAULT_INTERIOR_AUDIT = ROOT / "outputs_3d_case_matrix_interior_probe" / "case_016_interior_physics_audit.json"
DEFAULT_REPORT_ROOT = ROOT / "reports"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _status_color(value: float, target: float, *, lower_better: bool = True) -> str:
    if lower_better:
        if value <= target:
            return "#146c43"
        if value <= target * 1.2:
            return "#b26a00"
        return "#b42318"
    if value >= target:
        return "#146c43"
    if value >= target * 0.9:
        return "#b26a00"
    return "#b42318"


def _card(ax, x: float, y: float, w: float, h: float, title: str, value: str, subtitle: str, face: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.0,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03, y + h - 0.09, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#f8fafc")
    ax.text(x + 0.03, y + h - 0.23, value, transform=ax.transAxes, fontsize=24, fontweight="bold", color="#ffffff")
    ax.text(x + 0.03, y + 0.06, subtitle, transform=ax.transAxes, fontsize=10, color="#dbe4ee")


def _plot_scorecard(cv_summary: dict, validation_summary: dict, output_path: Path) -> None:
    aggregate = cv_summary["aggregate"]
    physics = validation_summary["physics_checks"]
    metrics = validation_summary["surface_validation_metrics"]
    rmse = float(metrics["combined_rmse_K"])
    q_error = float(physics["mean_Q_total_rel_error_pct"])
    energy_gap = float(physics["mean_energy_balance_gap_pct"])
    cv_rmse = float(aggregate["mean_combined_rmse_K"])

    fig = plt.figure(figsize=(13.5, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.patch.set_facecolor("#f4f1ea")
    ax.set_facecolor("#f4f1ea")

    ax.text(0.03, 0.93, "Boundary-Validated PINN Release", fontsize=22, fontweight="bold", color="#102a43", transform=ax.transAxes)
    ax.text(
        0.03,
        0.875,
        "Final scope: case matrix boundary surrogate with physics-audited interior behavior. Not a fully volumetric CFD validation.",
        fontsize=11,
        color="#3d556b",
        transform=ax.transAxes,
    )

    _card(ax, 0.03, 0.53, 0.22, 0.24, "Holdout Boundary RMSE", f"{rmse:.3f} K", "Target < 0.8 K on the reserved holdout split.", _status_color(rmse, 0.8))
    _card(ax, 0.28, 0.53, 0.22, 0.24, "Mean Heat-Duty Error", f"{q_error:.2f}%", "Target < 10% on holdout cases.", _status_color(q_error, 10.0))
    _card(ax, 0.53, 0.53, 0.22, 0.24, "Mean Energy Gap", f"{energy_gap:.2f}%", "Near the target, but still the main physics gap.", _status_color(energy_gap, 10.0))
    _card(ax, 0.78, 0.53, 0.19, 0.24, "4-Fold CV Mean RMSE", f"{cv_rmse:.3f} K", "Unseen-case average across all 16 case matrix cases.", _status_color(cv_rmse, 0.8))

    bullets = [
        f"Best holdout case: {physics['best_case_id']} at {physics['best_case_combined_rmse_K']:.3f} K combined RMSE.",
        f"Worst holdout case: {physics['worst_case_id']} at {physics['worst_case_combined_rmse_K']:.3f} K combined RMSE.",
        f"Pressure-drop trend correlation: hot {physics['hot_dp_correlation']:.3f}, cold {physics['cold_dp_correlation']:.3f}.",
        f"All held-out cases preserve hot cooling and cold heating: {physics['all_hot_cooling_positive'] and physics['all_cold_heating_positive']}.",
    ]
    ax.text(0.03, 0.40, "What To Trust", fontsize=14, fontweight="bold", color="#102a43", transform=ax.transAxes)
    for index, line in enumerate(bullets):
        ax.text(0.05, 0.35 - index * 0.06, f"- {line}", fontsize=11, color="#243b53", transform=ax.transAxes)

    ax.text(0.03, 0.08, "Core release artifacts live in the existing final output folders; the reports layer only summarizes them.", fontsize=10, color="#52606d", transform=ax.transAxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_case_cv(cv_case_df: pd.DataFrame, output_path: Path) -> None:
    ordered = cv_case_df.sort_values("combined_rmse_K", ascending=True).reset_index(drop=True)
    colors = []
    for _, row in ordered.iterrows():
        if str(row["case_id"]) == "case_016":
            colors.append("#b42318")
        elif float(row["combined_rmse_K"]) > 0.8:
            colors.append("#d97706")
        else:
            colors.append("#1976d2")

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")
    ax.bar(ordered["case_id"], ordered["combined_rmse_K"], color=colors, width=0.72)
    ax.axhline(0.8, color="#146c43", linestyle="--", linewidth=1.5, label="Target 0.8 K")
    ax.set_title("4-Fold Unseen-Case RMSE By Case Matrix Case", fontsize=16, fontweight="bold", color="#102a43")
    ax.set_ylabel("Combined RMSE [K]")
    ax.set_xlabel("Case Matrix Case")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.99, 0.95, "Red = persistent outlier", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#52606d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_holdout_breakdown(validation_case_df: pd.DataFrame, output_path: Path) -> None:
    ordered = validation_case_df.sort_values("combined_rmse_K", ascending=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f8f5ef")

    ax = axes[0]
    ax.set_facecolor("#f8f5ef")
    x = np.arange(len(ordered))
    width = 0.24
    ax.bar(x - width, ordered["hot_rmse_K"], width=width, label="Hot RMSE", color="#c0392b")
    ax.bar(x, ordered["cold_rmse_K"], width=width, label="Cold RMSE", color="#1d6fa5")
    ax.bar(x + width, ordered["combined_rmse_K"], width=width, label="Combined RMSE", color="#7b8a8b")
    ax.axhline(0.8, color="#146c43", linestyle="--", linewidth=1.3, label="Target 0.8 K")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["case_id"])
    ax.set_ylabel("RMSE [K]")
    ax.set_title("Reserved Holdout Surface Accuracy", fontsize=15, fontweight="bold", color="#102a43")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=4, loc="upper left")

    ax = axes[1]
    ax.set_facecolor("#f8f5ef")
    ax.bar(x - width / 2, ordered["Q_total_rel_error_pct"], width=width, label="Heat-duty error", color="#d97706")
    ax.bar(x + width / 2, ordered["energy_balance_gap_pct"], width=width, label="Energy gap", color="#6b7280")
    ax.axhline(10.0, color="#146c43", linestyle="--", linewidth=1.3, label="Target 10%")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["case_id"])
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("Reserved Holdout Physics Checks", fontsize=15, fontweight="bold", color="#102a43")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_case_016_focus(validation_case_df: pd.DataFrame, interior_audit: dict, output_path: Path) -> None:
    case_row = validation_case_df.loc[validation_case_df["case_id"] == "case_016"]
    if case_row.empty:
        raise ValueError("case_016 not found in the validation case summary.")
    row = case_row.iloc[0]
    interface = interior_audit["interface_residuals"]

    left_labels = ["Combined RMSE [K]", "Hot outlet abs err [K]", "Cold outlet abs err [K]", "Hot interface RMSE [K]", "Cold interface RMSE [K]"]
    left_values = [
        float(row["combined_rmse_K"]),
        float(row["hot_outlet_bulk_abs_error_K"]),
        float(row["cold_outlet_bulk_abs_error_K"]),
        float(row["hot_interface_temp_rmse_K"]),
        float(row["cold_interface_temp_rmse_K"]),
    ]
    right_labels = ["Heat-duty err [%]", "Energy gap [%]", "Hot dp err [%]", "Cold dp err [%]", "Interface temp audit RMSE [K]"]
    right_values = [
        float(row["Q_total_rel_error_pct"]),
        float(row["energy_balance_gap_pct"]),
        float(row["hot_dp_rel_error_pct"]),
        float(row["cold_dp_rel_error_pct"]),
        float(interface["temp_wall_cold"]["rmse"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("#fcfbf7")

    axes[0].set_facecolor("#fcfbf7")
    axes[0].barh(left_labels, left_values, color=["#b42318", "#d97706", "#1976d2", "#8b5cf6", "#6d28d9"])
    axes[0].invert_yaxis()
    axes[0].set_title("case_016 Error Profile", fontsize=15, fontweight="bold", color="#102a43")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].set_facecolor("#fcfbf7")
    axes[1].barh(right_labels, right_values, color=["#d97706", "#6b7280", "#c0392b", "#1d6fa5", "#7c3aed"])
    axes[1].invert_yaxis()
    axes[1].set_title("case_016 Physics Stress Points", fontsize=15, fontweight="bold", color="#102a43")
    axes[1].grid(axis="x", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _surface_heatmap_matrix(frame: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plot_frame = frame.copy()
    plot_frame["phi_deg"] = np.degrees(np.arctan2(plot_frame["y"], plot_frame["x"]))
    plot_frame.loc[plot_frame["phi_deg"] < 0.0, "phi_deg"] += 360.0
    phi_edges = np.linspace(0.0, 360.0, 73)
    z_edges = np.linspace(plot_frame["z"].min(), plot_frame["z"].max(), 73)
    plot_frame["phi_bin"] = pd.cut(plot_frame["phi_deg"], bins=phi_edges, include_lowest=True)
    plot_frame["z_bin"] = pd.cut(plot_frame["z"], bins=z_edges, include_lowest=True)
    binned = (
        plot_frame.groupby(["phi_bin", "z_bin"], observed=False)[value_column]
        .mean()
        .reset_index()
    )
    binned["phi_center"] = binned["phi_bin"].apply(lambda interval: float(interval.mid))
    binned["z_center"] = binned["z_bin"].apply(lambda interval: float(interval.mid))
    matrix = binned.pivot(index="phi_center", columns="z_center", values=value_column).sort_index()
    return matrix.columns.to_numpy(dtype=float), matrix.index.to_numpy(dtype=float), matrix.to_numpy(dtype=float)


def _plot_dataset_operating_space(case_manifest: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    figure.patch.set_facecolor("#faf7f2")
    axes[0].set_facecolor("#faf7f2")
    axes[1].set_facecolor("#faf7f2")

    scatter = axes[0].scatter(
        case_manifest["Th_in_K"],
        case_manifest["Tc_in_K"],
        c=case_manifest["Q_total"],
        s=case_manifest["effectiveness"] * 700.0,
        cmap="viridis",
        edgecolors="white",
        linewidths=1.0,
    )
    for _, row in case_manifest.iterrows():
        axes[0].text(float(row["Th_in_K"]) + 0.08, float(row["Tc_in_K"]) + 0.05, str(row["case_id"]), fontsize=8, color="#243b53")
    axes[0].set_title("Case Matrix Operating Matrix", fontsize=15, fontweight="bold", color="#102a43")
    axes[0].set_xlabel("Hot inlet [K]")
    axes[0].set_ylabel("Cold inlet [K]")
    axes[0].grid(alpha=0.22)
    cbar = figure.colorbar(scatter, ax=axes[0], pad=0.02)
    cbar.set_label("Heat duty [W]")

    pivot = (
        case_manifest.groupby(["Tc_in_K", "Th_in_K"], as_index=False)["Q_total"]
        .mean()
        .pivot(index="Tc_in_K", columns="Th_in_K", values="Q_total")
        .sort_index(ascending=True)
    )
    mesh = axes[1].imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Mean Heat Duty by Inlet Temperatures", fontsize=15, fontweight="bold", color="#102a43")
    axes[1].set_xticks(np.arange(len(pivot.columns)))
    axes[1].set_xticklabels([f"{value:.1f}" for value in pivot.columns.to_numpy(dtype=float)])
    axes[1].set_yticks(np.arange(len(pivot.index)))
    axes[1].set_yticklabels([f"{value:.1f}" for value in pivot.index.to_numpy(dtype=float)])
    axes[1].set_xlabel("Hot inlet [K]")
    axes[1].set_ylabel("Cold inlet [K]")
    cbar = figure.colorbar(mesh, ax=axes[1], pad=0.02)
    cbar.set_label("Mean heat duty [W]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_dataset_boundary_heatmaps(output_path: Path) -> None:
    cases = [("case_001", "Reference-like"), ("case_016", "Stress case")]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf7")
    for row_index, (case_id, label) in enumerate(cases):
        hot_frame = pd.read_csv(DATA_ROOT / case_id / "hot_wall_interface.csv")
        cold_frame = pd.read_csv(DATA_ROOT / case_id / "wall_cold_interface.csv")
        hot_x, hot_y, hot_matrix = _surface_heatmap_matrix(hot_frame, "T")
        cold_x, cold_y, cold_matrix = _surface_heatmap_matrix(cold_frame, "T")

        hot_ax = axes[row_index, 0]
        cold_ax = axes[row_index, 1]
        hot_ax.set_facecolor("#fbfaf7")
        cold_ax.set_facecolor("#fbfaf7")

        hot_mesh = hot_ax.imshow(hot_matrix, origin="lower", aspect="auto", cmap="turbo")
        cold_mesh = cold_ax.imshow(cold_matrix, origin="lower", aspect="auto", cmap="turbo")

        hot_ax.set_title(f"{case_id} | {label} | Hot wall T", fontsize=12, fontweight="bold", color="#102a43")
        cold_ax.set_title(f"{case_id} | {label} | Cold wall T", fontsize=12, fontweight="bold", color="#102a43")
        hot_ax.set_ylabel("Circumference bin")
        cold_ax.set_ylabel("Circumference bin")
        hot_ax.set_xlabel("Axial bin")
        cold_ax.set_xlabel("Axial bin")
        figure.colorbar(hot_mesh, ax=hot_ax, fraction=0.046, pad=0.03)
        figure.colorbar(cold_mesh, ax=cold_ax, fraction=0.046, pad=0.03)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_manifest(manifest: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_summary_markdown(manifest: dict, output_path: Path) -> None:
    lines = [
        "# Heat Exchanger Studio",
        "",
        "## What This Workspace Does",
        "",
        "- Browse preset operating points from the reference library.",
        "- Generate fresh surface previews for a selected operating point.",
        "- Create a slice view and review temperature balance when needed.",
        "- Launch a new run without editing scripts directly.",
        "",
        "## Start Here",
        "",
        "- Open the scenario library to compare saved operating points.",
        "- Use the live preview page to test a new condition and inspect the wall maps.",
        "- Use the build page when you want a fresh run from the same workspace.",
        "- Use the flow view page to export a slice view or review the temperature balance.",
        "",
        "## Visual Assets",
        "",
        f"- Dataset operating matrix: `{manifest['figures']['dataset_operating_matrix']}`",
        f"- Dataset boundary heatmaps: `{manifest['figures']['dataset_boundary_heatmaps']}`",
        "",
        "## Launch",
        "",
        "- Launch with `python app/final_boundary_gui.py`.",
        "- Or run `streamlit run app/boundary_studio.py` directly.",
        "",
        "## Notes",
        "",
        "- The slice view is intended for exploration and review.",
        "- Additional solver-backed interior fields can be connected later without changing the workspace flow.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard_html(manifest: dict, output_path: Path) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Heat Exchanger Studio</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: linear-gradient(180deg, #f5f1e8 0%, #ffffff 100%); color: #102a43; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }}
    .hero {{ margin-bottom: 24px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .hero p {{ margin: 0; color: #486581; max-width: 820px; line-height: 1.55; }}
    .steps {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 14px; margin: 24px 0 32px; }}
    .card {{ background: #102a43; color: white; border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(16, 42, 67, 0.12); }}
    .card .label {{ font-size: 18px; font-weight: 700; }}
    .card .value {{ font-size: 15px; margin-top: 10px; line-height: 1.6; color: rgba(255,255,255,0.82); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    figure {{ margin: 0; background: white; border-radius: 18px; padding: 14px; box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08); }}
    img {{ width: 100%; border-radius: 12px; }}
    figcaption {{ margin-top: 10px; color: #486581; font-size: 14px; }}
    .links {{ margin-top: 28px; padding: 18px; background: white; border-radius: 18px; box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08); }}
    .links a {{ display: inline-block; margin-right: 16px; margin-bottom: 10px; color: #0f609b; text-decoration: none; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Heat Exchanger Studio</h1>
      <p>A calmer front end for browsing presets, creating new previews, and opening slice views from one place.</p>
    </section>
    <section class="steps">
      <div class="card"><div class="label">1. Browse presets</div><div class="value">Explore the operating library and compare saved surface maps.</div></div>
      <div class="card"><div class="label">2. Create a preview</div><div class="value">Enter an operating point or pick a preset to generate a fresh wall-temperature view.</div></div>
      <div class="card"><div class="label">3. Start a build</div><div class="value">Launch a quick refresh or a full rebuild directly from the workspace.</div></div>
    </section>
    <section class="grid">
      <figure><img src="figures/dataset_operating_matrix.png" alt="dataset operating matrix"><figcaption>Operating-space view of the preset library.</figcaption></figure>
      <figure><img src="figures/dataset_boundary_heatmaps.png" alt="dataset boundary heatmaps"><figcaption>Reference surface patterns from the preset library.</figcaption></figure>
    </section>
    <section class="links">
      <a href="final_boundary_model_summary.md">Studio Guide</a>
      <a href="../app/final_boundary_gui.py">Launch Studio</a>
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    report_root = DEFAULT_REPORT_ROOT
    figures_root = report_root / "figures"
    report_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    cv_summary = _load_json(DEFAULT_CV_SUMMARY)
    validation_summary = _load_json(DEFAULT_VALIDATION_SUMMARY)
    interior_audit = _load_json(DEFAULT_INTERIOR_AUDIT)
    cv_case_df = pd.read_csv(DEFAULT_CV_CASES)
    validation_case_df = pd.read_csv(DEFAULT_VALIDATION_CASES)
    case_manifest_df = pd.read_csv(DATA_ROOT / "case_manifest.csv")

    scorecard_path = figures_root / "release_scorecard.png"
    cv_plot_path = figures_root / "case_cv_rmse.png"
    holdout_plot_path = figures_root / "holdout_breakdown.png"
    case_016_plot_path = figures_root / "case_016_focus.png"
    dataset_operating_path = figures_root / "dataset_operating_matrix.png"
    dataset_heatmaps_path = figures_root / "dataset_boundary_heatmaps.png"

    _plot_scorecard(cv_summary, validation_summary, scorecard_path)
    _plot_case_cv(cv_case_df, cv_plot_path)
    _plot_holdout_breakdown(validation_case_df, holdout_plot_path)
    _plot_case_016_focus(validation_case_df, interior_audit, case_016_plot_path)
    _plot_dataset_operating_space(case_manifest_df, dataset_operating_path)
    _plot_dataset_boundary_heatmaps(dataset_heatmaps_path)

    manifest = {
        "headline_metrics": {
            "holdout_combined_rmse_K": float(validation_summary["surface_validation_metrics"]["combined_rmse_K"]),
            "mean_Q_total_rel_error_pct": float(validation_summary["physics_checks"]["mean_Q_total_rel_error_pct"]),
            "mean_energy_balance_gap_pct": float(validation_summary["physics_checks"]["mean_energy_balance_gap_pct"]),
            "cv_mean_combined_rmse_K": float(cv_summary["aggregate"]["mean_combined_rmse_K"]),
        },
        "artifacts": {
            "checkpoint": str((ROOT / "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune" / "checkpoints" / "best_model_3d.pt").resolve()),
            "temperature_calibration": str((ROOT / "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune" / "boundary_temperature_calibration.json").resolve()),
            "validation_summary": str(DEFAULT_VALIDATION_SUMMARY.resolve()),
            "validation_case_summary": str(DEFAULT_VALIDATION_CASES.resolve()),
            "cv_summary": str(DEFAULT_CV_SUMMARY.resolve()),
            "cv_case_summary": str(DEFAULT_CV_CASES.resolve()),
            "interior_audit": str(DEFAULT_INTERIOR_AUDIT.resolve()),
            "dashboard_html": str((report_root / "index.html").resolve()),
            "release_notes": str((report_root / "final_boundary_model_summary.md").resolve()),
        },
        "figures": {
            "scorecard": str(scorecard_path.resolve()),
            "cv_case_rmse": str(cv_plot_path.resolve()),
            "holdout_breakdown": str(holdout_plot_path.resolve()),
            "case_016_focus": str(case_016_plot_path.resolve()),
            "dataset_operating_matrix": str(dataset_operating_path.resolve()),
            "dataset_boundary_heatmaps": str(dataset_heatmaps_path.resolve()),
        },
        "notes": {
            "scope": "Boundary-validated case matrix surrogate with interior physics audit.",
            "limitation": "No independent interior CFD field holdout is included yet.",
        },
    }

    manifest_path = report_root / "release_manifest.json"
    summary_md_path = report_root / "final_boundary_model_summary.md"
    dashboard_path = report_root / "index.html"
    _write_manifest(manifest, manifest_path)
    _write_summary_markdown(manifest, summary_md_path)
    _write_dashboard_html(manifest, dashboard_path)

    print(f"Release assets written to: {report_root.resolve()}")
    print(f"Dashboard: {dashboard_path.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
