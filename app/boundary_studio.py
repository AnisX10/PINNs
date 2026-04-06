from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports"
GUI_RUN_ROOT = REPORT_ROOT / "gui_runs"
DATA_ROOT = ROOT / "data" / "case_matrix" / "comsol_case_matrix_dataset"

CASE_MANIFEST_PATH = DATA_ROOT / "case_manifest.csv"
DEFAULT_CONFIG = "configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml"
DEFAULT_CHECKPOINT = (
    "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt"
)
DEFAULT_CALIBRATION = (
    "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/boundary_temperature_calibration.json"
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url("https://fonts.cdnfonts.com/css/aeonik");

        :root {
            --bg-cream: #f5f1ea;
            --bg-ice: #f8fbfd;
            --ink: #102536;
            --muted: #5f7384;
            --line: rgba(16, 37, 54, 0.09);
            --card: rgba(255, 255, 255, 0.88);
            --accent: #133d5a;
        }

        #MainMenu,
        footer,
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stHeaderActionElements"],
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }

        html, body, [class*="st-"], [data-testid="stAppViewContainer"] * {
            font-family: "Aeonik", "Segoe UI", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(227, 233, 241, 0.85), transparent 26%),
                radial-gradient(circle at bottom right, rgba(255, 233, 214, 0.65), transparent 26%),
                linear-gradient(180deg, var(--bg-cream) 0%, #ffffff 40%, var(--bg-ice) 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 26, 40, 0.94) 0%, rgba(16, 37, 54, 0.98) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f5f8fb !important;
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }

        .hero-shell,
        .glass-card,
        .metric-tile,
        .action-tile,
        .note-card {
            background: var(--card);
            border: 1px solid var(--line);
            box-shadow: 0 24px 56px rgba(15, 35, 52, 0.08);
            border-radius: 28px;
        }

        .hero-shell {
            padding: 1.7rem 1.8rem;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.18), transparent 30%),
                linear-gradient(135deg, #102536 0%, #143d5a 52%, #1f5d80 100%);
            color: #ffffff;
            min-height: 210px;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0.9rem 0 0;
            font-size: 2.55rem;
            line-height: 0.98;
            letter-spacing: -0.045em;
        }

        .hero-copy {
            margin: 0.85rem 0 0;
            color: rgba(255, 255, 255, 0.84);
            line-height: 1.65;
            max-width: 42rem;
            font-size: 1rem;
        }

        .section-kicker {
            color: #5c7386;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }

        .section-title {
            font-size: 1.55rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
            color: var(--ink);
            font-weight: 700;
            margin: 0.12rem 0 0;
        }

        .section-copy {
            color: var(--muted);
            margin-top: 0.42rem;
            line-height: 1.72;
            font-size: 0.98rem;
        }

        .glass-card,
        .note-card {
            padding: 1.15rem 1.2rem;
        }

        .metric-tile {
            padding: 1.05rem 1.1rem;
            min-height: 132px;
        }

        .metric-tile .label {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #61788b;
            font-weight: 700;
        }

        .metric-tile .value {
            color: var(--ink);
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 0.26rem;
            letter-spacing: -0.03em;
        }

        .metric-tile .note {
            color: var(--muted);
            margin-top: 0.3rem;
            line-height: 1.55;
            font-size: 0.9rem;
        }

        .action-tile {
            padding: 1.1rem 1.15rem;
            min-height: 180px;
        }

        .action-title {
            color: var(--ink);
            font-size: 1.04rem;
            font-weight: 700;
            margin-bottom: 0.32rem;
        }

        .action-copy {
            color: var(--muted);
            line-height: 1.62;
            font-size: 0.94rem;
        }

        .chip {
            display: inline-block;
            padding: 0.24rem 0.72rem;
            border-radius: 999px;
            background: rgba(19, 61, 90, 0.08);
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.62rem;
        }

        .sidebar-chip {
            display: inline-block;
            padding: 0.24rem 0.68rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .soft-note {
            color: var(--muted);
            font-size: 0.93rem;
            line-height: 1.65;
        }

        .quick-list {
            margin: 0;
            padding-left: 1.15rem;
            color: var(--muted);
            line-height: 1.8;
        }

        .recent-item {
            padding: 0.7rem 0.85rem;
            border-radius: 18px;
            background: rgba(19, 61, 90, 0.05);
            border: 1px solid rgba(19, 61, 90, 0.08);
            margin-bottom: 0.6rem;
        }

        .recent-item strong {
            color: var(--ink);
            display: block;
            margin-bottom: 0.18rem;
        }

        .recent-item span {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .status-good,
        .status-watch,
        .status-review {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.72rem;
            border-radius: 999px;
            font-size: 0.79rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .status-good {
            color: #0b7a55;
            background: rgba(11, 122, 85, 0.10);
        }

        .status-watch {
            color: #a35b08;
            background: rgba(163, 91, 8, 0.10);
        }

        .status-review {
            color: #9f1d35;
            background: rgba(159, 29, 53, 0.10);
        }

        .stButton button,
        .stDownloadButton button {
            border-radius: 999px;
            border: 0;
            background: #102536;
            color: white;
            font-weight: 700;
            min-height: 3rem;
            padding: 0.68rem 1.08rem;
            box-shadow: 0 16px 34px rgba(16, 37, 54, 0.16);
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: #173b54;
            color: white;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            border-radius: 18px !important;
            min-height: 3rem !important;
            border: 1px solid rgba(16, 37, 54, 0.10) !important;
            background: rgba(255,255,255,0.95) !important;
        }

        .stRadio [role="radiogroup"] {
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .stRadio [role="radio"] {
            border-radius: 999px;
            padding: 0.65rem 0.95rem;
            background: rgba(19, 61, 90, 0.05);
            border: 1px solid rgba(19, 61, 90, 0.08);
        }

        .stRadio [aria-checked="true"] {
            background: #102536 !important;
            color: white !important;
            border-color: #102536 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.56rem 0.96rem;
            background: rgba(19, 61, 90, 0.06);
        }

        .stTabs [aria-selected="true"] {
            background: #102536 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_tile(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="label">{title}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_action_tile(title: str, text: str, chip: str) -> None:
    st.markdown(
        f"""
        <div class="action-tile">
          <div class="chip">{chip}</div>
          <div class="action-title">{title}</div>
          <div class="action-copy">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="recent-item">
          <strong>{title}</strong>
          <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(label: str, tone: str) -> str:
    css_class = {
        "good": "status-good",
        "watch": "status-watch",
        "review": "status-review",
    }.get(tone, "status-watch")
    return f'<div class="{css_class}">{label}</div>'


def _scenario_title(case_id: str) -> str:
    return f"Preset {str(case_id).replace('case_', '')}"


def _scenario_label(case_manifest: pd.DataFrame, case_id: str) -> str:
    row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
    return (
        f"{_scenario_title(case_id)} | "
        f"{float(row['Th_in_K']):.1f} K to {float(row['Tc_in_K']):.1f} K | "
        f"{float(row['uh_in_mps']):.1f} / {float(row['uc_in_mps']):.1f} m/s"
    )


def _scenario_story(row: pd.Series) -> str:
    span = float(row["Th_in_K"] - row["Tc_in_K"])
    flow_gap = abs(float(row["uh_in_mps"] - row["uc_in_mps"]))
    if span >= 24.0:
        thermal_note = "a wide temperature gap"
    elif span >= 18.0:
        thermal_note = "a balanced temperature gap"
    else:
        thermal_note = "a gentle temperature gap"
    if flow_gap <= 0.15:
        flow_note = "with matched flow on both sides"
    elif float(row["uh_in_mps"]) > float(row["uc_in_mps"]):
        flow_note = "with a stronger hot-side push"
    else:
        flow_note = "with a stronger cold-side push"
    return f"This setup uses {thermal_note} {flow_note}."


def _current_case(case_manifest: pd.DataFrame, fallback_index: int = 0) -> str:
    options = case_manifest["case_id"].tolist()
    saved = st.session_state.get("active_case_id")
    if saved in options:
        return str(saved)
    return str(options[fallback_index])


def _remember_case(case_id: str) -> None:
    st.session_state["active_case_id"] = str(case_id)


def _goto_page(page_name: str) -> None:
    st.session_state["studio_page"] = page_name
    st.rerun()


def _record_recent_activity(kind: str, title: str, path: Path) -> None:
    items = list(st.session_state.get("recent_activity", []))
    items.insert(
        0,
        {
            "kind": kind,
            "title": title,
            "path": str(path),
            "filename": path.name,
        },
    )
    st.session_state["recent_activity"] = items[:6]


def _load_case_inputs(case_manifest: pd.DataFrame, case_id: str) -> dict[str, float]:
    row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
    return {
        "Th_in_K": float(row["Th_in_K"]),
        "Tc_in_K": float(row["Tc_in_K"]),
        "uh_in_mps": float(row["uh_in_mps"]),
        "uc_in_mps": float(row["uc_in_mps"]),
    }


def _surface_heatmap(frame: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    plot_frame = frame.copy()
    if "phi_rad" not in plot_frame.columns:
        plot_frame["phi_rad"] = np.arctan2(plot_frame["y"], plot_frame["x"])
    plot_frame["phi_deg"] = np.degrees(plot_frame["phi_rad"])
    plot_frame.loc[plot_frame["phi_deg"] < 0.0, "phi_deg"] += 360.0
    phi_edges = np.linspace(0.0, 360.0, 73)
    z_edges = np.linspace(float(plot_frame["z"].min()), float(plot_frame["z"].max()), 73)
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
    figure = px.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        color_continuous_scale="Turbo",
        labels={
            "x": "Length position [m]",
            "y": "Around the surface [deg]",
            "color": value_column,
        },
    )
    figure.update_layout(
        title=title,
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _operating_matrix_figure(case_manifest: pd.DataFrame) -> go.Figure:
    plot_frame = case_manifest.copy()
    plot_frame["flow_label"] = (
        plot_frame["uh_in_mps"].map(lambda value: f"{value:.1f}")
        + " / "
        + plot_frame["uc_in_mps"].map(lambda value: f"{value:.1f}")
        + " m/s"
    )
    figure = px.scatter(
        plot_frame,
        x="Th_in_K",
        y="Tc_in_K",
        color="Q_total",
        size="effectiveness",
        symbol="flow_label",
        text="case_id",
        color_continuous_scale="Tealgrn",
        labels={
            "Th_in_K": "Hot-side starting temperature [K]",
            "Tc_in_K": "Cold-side starting temperature [K]",
            "Q_total": "Transfer level [W]",
            "flow_label": "Flow setting",
        },
    )
    figure.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    figure.update_layout(
        title="Preset map",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Flow setting",
    )
    return figure


def _transfer_trend_figure(case_manifest: pd.DataFrame) -> go.Figure:
    plot_frame = case_manifest.copy()
    plot_frame["temperature_gap"] = plot_frame["Th_in_K"] - plot_frame["Tc_in_K"]
    plot_frame["flow_balance"] = plot_frame["uh_in_mps"] / plot_frame["uc_in_mps"]
    figure = px.scatter(
        plot_frame,
        x="temperature_gap",
        y="Q_total",
        size="effectiveness",
        color="flow_balance",
        text="case_id",
        color_continuous_scale="Brwnyl",
        labels={
            "temperature_gap": "Starting temperature gap [K]",
            "Q_total": "Transfer level [W]",
            "flow_balance": "Hot-to-cold flow ratio",
        },
    )
    figure.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    figure.update_layout(
        title="Transfer trend across presets",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _boundary_distribution_figure(case_dir: Path, value_column: str, title: str) -> go.Figure:
    boundary_names = {
        "hot_inlet": "Hot inlet",
        "hot_outlet": "Hot outlet",
        "hot_wall_interface": "Inner wall",
        "wall_cold_interface": "Outer wall",
        "cold_inlet": "Cold inlet",
        "cold_outlet": "Cold outlet",
    }
    parts: list[pd.DataFrame] = []
    for filename, label in boundary_names.items():
        file_path = case_dir / f"{filename}.csv"
        if not file_path.exists():
            continue
        frame = _load_csv(file_path)
        if value_column not in frame.columns:
            continue
        plot_frame = frame[[value_column]].copy()
        plot_frame["surface"] = label
        parts.append(plot_frame)
    combined = pd.concat(parts, ignore_index=True)
    figure = px.box(
        combined,
        x="surface",
        y=value_column,
        color="surface",
        color_discrete_sequence=["#c76b34", "#8d4a27", "#6b7fa0", "#2d719a", "#6fb2d6", "#1f4f72"],
        points="all",
    )
    figure.update_layout(
        title=title,
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title="Boundary",
    )
    return figure


def _preset_comparison_figure(case_manifest: pd.DataFrame, left_case: str, right_case: str) -> go.Figure:
    compare_metrics = [
        ("Hot start [K]", "Th_in_K"),
        ("Cold start [K]", "Tc_in_K"),
        ("Hot flow [m/s]", "uh_in_mps"),
        ("Cold flow [m/s]", "uc_in_mps"),
        ("Transfer level [W]", "Q_total"),
        ("Effectiveness", "effectiveness"),
    ]
    frames: list[pd.DataFrame] = []
    for case_id in [left_case, right_case]:
        row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
        case_frame = pd.DataFrame(
            {
                "metric": [label for label, _ in compare_metrics],
                "value": [float(row[column]) for _, column in compare_metrics],
                "preset": _scenario_title(case_id),
            }
        )
        frames.append(case_frame)
    combined = pd.concat(frames, ignore_index=True)
    figure = px.bar(
        combined,
        x="metric",
        y="value",
        color="preset",
        barmode="group",
        color_discrete_sequence=["#133d5a", "#c76b34"],
    )
    figure.update_layout(
        title="Side-by-side preset comparison",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        legend_title_text="",
    )
    return figure


def _prediction_story_figure(frame: pd.DataFrame) -> go.Figure:
    first_row = frame.iloc[0]
    story = pd.DataFrame(
        [
            {"stream": "Hot side", "stage": "Start", "temperature_K": float(first_row["Th_in_K"])},
            {
                "stream": "Hot side",
                "stage": "End",
                "temperature_K": float(frame.loc[frame["boundary"] == "hot_outlet", "T_pred_mean_K"].mean()),
            },
            {"stream": "Cold side", "stage": "Start", "temperature_K": float(first_row["Tc_in_K"])},
            {
                "stream": "Cold side",
                "stage": "End",
                "temperature_K": float(frame.loc[frame["boundary"] == "cold_outlet", "T_pred_mean_K"].mean()),
            },
        ]
    )
    figure = px.line(
        story,
        x="stage",
        y="temperature_K",
        color="stream",
        markers=True,
        color_discrete_map={"Hot side": "#c76b34", "Cold side": "#25749b"},
        labels={"temperature_K": "Temperature [K]", "stage": ""},
    )
    figure.update_traces(line=dict(width=4), marker=dict(size=10))
    figure.update_layout(
        title="Temperature path through the exchanger",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return figure


def _surface_mean_figure(frame: pd.DataFrame) -> go.Figure:
    summary = pd.DataFrame(
        [
            {
                "surface": "Inner surface",
                "temperature_K": float(frame.loc[frame["boundary"] == "hot_wall", "T_pred_mean_K"].mean()),
            },
            {
                "surface": "Outer surface",
                "temperature_K": float(frame.loc[frame["boundary"] == "cold_inner_wall", "T_pred_mean_K"].mean()),
            },
        ]
    )
    figure = px.bar(
        summary,
        x="surface",
        y="temperature_K",
        color="surface",
        color_discrete_sequence=["#c76b34", "#25749b"],
    )
    figure.update_layout(
        title="Surface temperature summary",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Temperature [K]",
    )
    return figure


def _interior_slice_figure(frame: pd.DataFrame, domain: str, z_value: float) -> go.Figure:
    slice_frame = frame.loc[(frame["domain"] == domain) & np.isclose(frame["z"], z_value)].copy()
    figure = px.scatter(
        slice_frame,
        x="x",
        y="y",
        color="T_pred",
        color_continuous_scale="Turbo",
        render_mode="webgl",
        labels={"x": "x [m]", "y": "y [m]", "T_pred": "Temperature [K]"},
    )
    figure.update_traces(marker=dict(size=6, opacity=0.94))
    figure.update_layout(
        title=f"{domain.replace('_', ' ').title()} slice at z = {z_value:.3f} m",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_scaleanchor="x",
    )
    return figure


def _temperature_band_figure(audit: dict[str, Any]) -> go.Figure:
    ranges = audit["temperature_ranges"]
    labels = ["Hot channel", "Wall", "Cold channel"]
    values = [
        float(ranges["hot_fluid_K"]["mean"]),
        float(ranges["wall_K"]["mean"]),
        float(ranges["cold_fluid_K"]["mean"]),
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#c76b34", "#7b8898", "#25749b"],
        )
    )
    figure.update_layout(
        title="Average temperature across the core",
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Temperature [K]",
    )
    return figure


def _quality_band(value: float, good: float, watch: float) -> tuple[str, str]:
    if value <= good:
        return "Smooth", "good"
    if value <= watch:
        return "Watch", "watch"
    return "Needs review", "review"


def _show_report_image(path: Path) -> None:
    if path.exists():
        st.image(str(path), use_container_width=True)


def _run_command(command: list[str], label: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        message = completed.stdout.strip()
        if completed.stderr.strip():
            message = f"{message}\n{completed.stderr.strip()}".strip()
        return True, message or f"{label} completed."
    except subprocess.CalledProcessError as exc:
        combined = (exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")
        return False, combined.strip() or f"{label} failed."


def _render_home(case_manifest: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-kicker">No-code workspace</div>
          <div class="hero-title">Heat Flow Studio</div>
          <div class="hero-copy">
            Open a ready-made setup, try your own operating point, start a fresh model run,
            or inspect a slice through the exchanger from one calm workspace.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    action_cols = st.columns(4)
    with action_cols[0]:
        _render_action_tile(
            "Open the library",
            "Browse the built-in operating presets, compare two setups, and inspect wall maps before you change anything.",
            "Start here",
        )
        if st.button("Open presets", key="home_library", use_container_width=True):
            _goto_page("Preset Library")
    with action_cols[1]:
        _render_action_tile(
            "Try a fresh setup",
            "Enter your own starting temperatures and flow settings, then generate a new preview in one click.",
            "Most used",
        )
        if st.button("Create preview", key="home_preview", use_container_width=True):
            _goto_page("Try a Setup")
    with action_cols[2]:
        _render_action_tile(
            "Create a new model",
            "Choose a light, balanced, or deep build without opening any scripts or configuration files.",
            "Guided run",
        )
        if st.button("Open build tools", key="home_build", use_container_width=True):
            _goto_page("Model Workshop")
    with action_cols[3]:
        _render_action_tile(
            "Look inside the flow",
            "Create a slice view through the exchanger and review how temperature changes from one side to the other.",
            "Explore",
        )
        if st.button("Open inside view", key="home_inside", use_container_width=True):
            _goto_page("Inside View")

    st.write("")
    info_left, info_right = st.columns([1.15, 0.85])
    with info_left:
        _render_section_intro(
            "Quick start",
            "A simple way to move through the workspace",
            "If this is your first time here, these three steps will get you from a preset to a usable preview quickly.",
        )
        st.markdown(
            """
            <div class="glass-card">
              <ol class="quick-list">
                <li>Open <strong>Preset Library</strong> to choose a starting point.</li>
                <li>Go to <strong>Try a Setup</strong> to generate a fresh surface preview.</li>
                <li>Use <strong>Inside View</strong> when you want a slice through the exchanger.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        library_cases = [
            case_manifest.sort_values("Q_total", ascending=False).iloc[0]["case_id"],
            case_manifest.sort_values("effectiveness", ascending=False).iloc[0]["case_id"],
            case_manifest.iloc[len(case_manifest) // 2]["case_id"],
        ]
        suggestion_cols = st.columns(3)
        for column, case_id in zip(suggestion_cols, library_cases, strict=False):
            row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
            with column:
                _render_action_tile(
                    _scenario_title(case_id),
                    _scenario_story(row),
                    "Suggested preset",
                )
                if st.button(f"Use {_scenario_title(case_id)}", key=f"home_case_{case_id}", use_container_width=True):
                    _remember_case(case_id)
                    _goto_page("Try a Setup")
    with info_right:
        _render_section_intro(
            "Recent activity",
            "What you created most recently",
            "Your latest previews, builds, and slice files appear here during the current session.",
        )
        recent_items = st.session_state.get("recent_activity", [])
        if recent_items:
            for item in recent_items:
                _render_recent_card(item["title"], item["filename"])
        else:
            st.markdown(
                """
                <div class="note-card">
                  <div class="soft-note">
                    Nothing has been created in this session yet. Start with a preset, preview a setup, or open the inside view.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    visual_left, visual_right = st.columns(2)
    with visual_left:
        _show_report_image(ROOT / "reports" / "figures" / "dataset_operating_matrix.png")
    with visual_right:
        _show_report_image(ROOT / "reports" / "figures" / "dataset_boundary_heatmaps.png")


def _render_preset_library(case_manifest: pd.DataFrame) -> None:
    _render_section_intro(
        "Preset library",
        "Browse ready-made operating setups",
        "Pick a starting point, compare it with another preset, and inspect how the exchanger surface behaves before you launch a new run.",
    )

    selection_left, selection_right = st.columns([1.3, 1.0])
    with selection_left:
        selected_case = st.selectbox(
            "Primary preset",
            options=case_manifest["case_id"].tolist(),
            index=case_manifest["case_id"].tolist().index(_current_case(case_manifest)),
            format_func=lambda case_id: _scenario_label(case_manifest, case_id),
            key="library_case",
        )
        _remember_case(selected_case)
    with selection_right:
        compare_options = [case_id for case_id in case_manifest["case_id"].tolist() if case_id != selected_case]
        compare_case = st.selectbox(
            "Compare with",
            options=compare_options,
            index=0,
            format_func=lambda case_id: _scenario_label(case_manifest, case_id),
            key="library_compare_case",
        )

    row = case_manifest.loc[case_manifest["case_id"] == selected_case].iloc[0]
    metrics = st.columns(4)
    with metrics[0]:
        _render_metric_tile("Preset style", _scenario_title(selected_case), _scenario_story(row))
    with metrics[1]:
        span = float(row["Th_in_K"] - row["Tc_in_K"])
        _render_metric_tile("Temperature gap", f"{span:.1f} K", "Difference between the two starting temperatures.")
    with metrics[2]:
        _render_metric_tile(
            "Flow setting",
            f"{float(row['uh_in_mps']):.1f} / {float(row['uc_in_mps']):.1f} m/s",
            "Hot side first, cold side second.",
        )
    with metrics[3]:
        _render_metric_tile(
            "Transfer level",
            f"{float(row['Q_total']):.0f} W",
            "Average heat transfer for this preset.",
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(_operating_matrix_figure(case_manifest), use_container_width=True)
    with chart_right:
        st.plotly_chart(_transfer_trend_figure(case_manifest), use_container_width=True)

    case_dir = DATA_ROOT / selected_case
    hot_wall = _load_csv(case_dir / "hot_wall_interface.csv")
    cold_wall = _load_csv(case_dir / "wall_cold_interface.csv")
    tabs = st.tabs(["Wall maps", "Boundary spread", "Preset compare"])
    with tabs[0]:
        map_left, map_right = st.columns(2)
        with map_left:
            st.plotly_chart(
                _surface_heatmap(hot_wall, "T", f"{_scenario_title(selected_case)} | Inner surface temperature"),
                use_container_width=True,
            )
        with map_right:
            st.plotly_chart(
                _surface_heatmap(cold_wall, "T", f"{_scenario_title(selected_case)} | Outer surface temperature"),
                use_container_width=True,
            )
        heat_left, heat_right = st.columns(2)
        with heat_left:
            st.plotly_chart(
                _surface_heatmap(hot_wall, "qn", f"{_scenario_title(selected_case)} | Inner surface heat flow"),
                use_container_width=True,
            )
        with heat_right:
            st.plotly_chart(
                _surface_heatmap(cold_wall, "qn", f"{_scenario_title(selected_case)} | Outer surface heat flow"),
                use_container_width=True,
            )
    with tabs[1]:
        spread_left, spread_right = st.columns(2)
        with spread_left:
            st.plotly_chart(
                _boundary_distribution_figure(case_dir, "T", "Temperature spread across boundaries"),
                use_container_width=True,
            )
        with spread_right:
            st.plotly_chart(
                _boundary_distribution_figure(case_dir, "qn", "Heat flow spread across boundaries"),
                use_container_width=True,
            )
    with tabs[2]:
        st.plotly_chart(
            _preset_comparison_figure(case_manifest, selected_case, compare_case),
            use_container_width=True,
        )
        compare_row = case_manifest.loc[case_manifest["case_id"] == compare_case].iloc[0]
        st.markdown(
            f"""
            <div class="glass-card">
              <div class="soft-note">
                <strong>{_scenario_title(selected_case)}</strong> uses {_scenario_story(row).lower()}
                <br><br>
                <strong>{_scenario_title(compare_case)}</strong> uses {_scenario_story(compare_row).lower()}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_try_setup(case_manifest: pd.DataFrame) -> None:
    _render_section_intro(
        "Live preview",
        "Create a fresh surface preview",
        "Start from a saved preset or enter your own operating conditions. The studio will generate a boundary preview and keep the latest files ready to download.",
    )

    mode = st.radio(
        "How would you like to start?",
        ["Use a saved preset", "Enter my own values"],
        horizontal=True,
        key="preview_mode",
    )

    if mode == "Use a saved preset":
        case_id = st.selectbox(
            "Preset",
            case_manifest["case_id"].tolist(),
            index=case_manifest["case_id"].tolist().index(_current_case(case_manifest)),
            format_func=lambda value: _scenario_label(case_manifest, value),
            key="preview_case",
        )
        _remember_case(case_id)
        defaults = _load_case_inputs(case_manifest, case_id)
    else:
        case_id = "custom"
        defaults = {"Th_in_K": 303.0, "Tc_in_K": 283.5, "uh_in_mps": 1.0, "uc_in_mps": 1.0}

    control_left, control_right = st.columns([1.05, 0.95])
    with control_left:
        with st.form("preview_form"):
            temp_left, temp_right = st.columns(2)
            hot_temp = temp_left.slider(
                "Hot-side starting temperature [K]",
                min_value=300.0,
                max_value=314.0,
                step=0.5,
                value=float(defaults["Th_in_K"]),
            )
            cold_temp = temp_right.slider(
                "Cold-side starting temperature [K]",
                min_value=282.0,
                max_value=289.0,
                step=0.5,
                value=float(defaults["Tc_in_K"]),
            )
            flow_left, flow_right = st.columns(2)
            hot_flow = flow_left.slider(
                "Hot-side flow [m/s]",
                min_value=0.5,
                max_value=1.5,
                step=0.1,
                value=float(defaults["uh_in_mps"]),
            )
            cold_flow = flow_right.slider(
                "Cold-side flow [m/s]",
                min_value=0.5,
                max_value=1.5,
                step=0.1,
                value=float(defaults["uc_in_mps"]),
            )
            submitted = st.form_submit_button("Generate preview")

    with control_right:
        _render_action_tile(
            "What this view gives you",
            "A new boundary preview, wall heatmaps, a start-to-end temperature story, and downloadable files for the latest run.",
            "Preview output",
        )
        preview_cards = st.columns(2)
        with preview_cards[0]:
            _render_metric_tile(
                "Temperature gap",
                f"{hot_temp - cold_temp:.1f} K",
                "Wider gaps usually drive stronger exchange.",
            )
        with preview_cards[1]:
            _render_metric_tile(
                "Flow balance",
                f"{hot_flow / cold_flow:.2f}",
                "A value near 1.00 means both sides move at a similar rate.",
            )

    prediction_dir = GUI_RUN_ROOT / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = prediction_dir / f"{case_id}_boundary_prediction.csv"
    prediction_json = prediction_csv.with_suffix(".json")

    if submitted:
        command = [
            sys.executable,
            "scripts/predict_boundary_3d.py",
            "--config",
            DEFAULT_CONFIG,
            "--checkpoint",
            DEFAULT_CHECKPOINT,
            "--temperature-calibration-json",
            DEFAULT_CALIBRATION,
            "--Th-in",
            str(hot_temp),
            "--Tc-in",
            str(cold_temp),
            "--uh-in",
            str(hot_flow),
            "--uc-in",
            str(cold_flow),
            "--output",
            str(prediction_csv),
        ]
        with st.spinner("Creating a fresh preview..."):
            ok, message = _run_command(command, "Preview")
        if ok:
            st.success("Your preview is ready.")
            _record_recent_activity("preview", "Surface preview created", prediction_csv)
        else:
            st.error("The preview could not be created.")
        with st.expander("Technical log", expanded=not ok):
            st.code(message)

    if prediction_csv.exists():
        frame = _load_csv(prediction_csv)
        cards = st.columns(4)
        with cards[0]:
            _render_metric_tile(
                "Hot-side exit",
                f"{float(frame.loc[frame['boundary'] == 'hot_outlet', 'T_pred_mean_K'].mean()):.2f} K",
                "Average predicted exit temperature.",
            )
        with cards[1]:
            _render_metric_tile(
                "Cold-side exit",
                f"{float(frame.loc[frame['boundary'] == 'cold_outlet', 'T_pred_mean_K'].mean()):.2f} K",
                "Average predicted exit temperature.",
            )
        with cards[2]:
            _render_metric_tile(
                "Inner surface",
                f"{float(frame.loc[frame['boundary'] == 'hot_wall', 'T_pred_mean_K'].mean()):.2f} K",
                "Average inner wall temperature.",
            )
        with cards[3]:
            _render_metric_tile(
                "Outer surface",
                f"{float(frame.loc[frame['boundary'] == 'cold_inner_wall', 'T_pred_mean_K'].mean()):.2f} K",
                "Average outer wall temperature.",
            )

        tabs = st.tabs(["Surface maps", "Temperature story", "Downloads"])
        with tabs[0]:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    _surface_heatmap(
                        frame.loc[frame["boundary"] == "hot_wall"],
                        "T_pred_mean_K",
                        "Inner surface temperature",
                    ),
                    use_container_width=True,
                )
            with right:
                st.plotly_chart(
                    _surface_heatmap(
                        frame.loc[frame["boundary"] == "cold_inner_wall"],
                        "T_pred_mean_K",
                        "Outer surface temperature",
                    ),
                    use_container_width=True,
                )
        with tabs[1]:
            story_left, story_right = st.columns(2)
            with story_left:
                st.plotly_chart(_prediction_story_figure(frame), use_container_width=True)
            with story_right:
                st.plotly_chart(_surface_mean_figure(frame), use_container_width=True)
        with tabs[2]:
            st.download_button(
                "Download preview table",
                data=prediction_csv.read_bytes(),
                file_name=prediction_csv.name,
                use_container_width=True,
            )
            if prediction_json.exists():
                st.download_button(
                    "Download preview summary",
                    data=prediction_json.read_bytes(),
                    file_name=prediction_json.name,
                    use_container_width=True,
                )


def _render_model_workshop(case_manifest: pd.DataFrame) -> None:
    _render_section_intro(
        "Model workshop",
        "Start a new run without touching code",
        "Choose the amount of work you want the studio to do, pick a preset, and the workspace will prepare the right run for you.",
    )

    mode_left, mode_mid, mode_right = st.columns(3)
    with mode_left:
        _render_action_tile(
            "Quick check",
            "Best when you want a fast confidence check on one preset.",
            "Fastest",
        )
    with mode_mid:
        _render_action_tile(
            "Balanced refresh",
            "A fuller refresh that gives a cleaner training history without taking as long as a deep run.",
            "Recommended",
        )
    with mode_right:
        _render_action_tile(
            "Deep validation run",
            "Use this when you want the broader reserved validation workflow instead of a one-case refresh.",
            "Longest",
        )

    with st.form("build_form"):
        build_mode = st.radio(
            "Run style",
            ["Quick check", "Balanced refresh", "Deep validation run"],
            horizontal=True,
            key="build_mode",
        )
        case_id = st.selectbox(
            "Preset",
            options=case_manifest["case_id"].tolist(),
            index=case_manifest["case_id"].tolist().index(_current_case(case_manifest)),
            format_func=lambda value: _scenario_label(case_manifest, value),
            key="build_case",
        )
        _remember_case(case_id)
        run_now = st.form_submit_button("Start run")

    training_root = GUI_RUN_ROOT / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    if run_now:
        if build_mode == "Quick check":
            output_dir = training_root / f"quick_{case_id}"
            command = [
                sys.executable,
                "scripts/train_pinn_3d.py",
                "--config",
                "configs/double_pipe_3d_case_matrix_case001.yaml",
                "--adam-epochs",
                "8",
                "--set",
                f"case_matrix_3d.case_id={case_id}",
                "--set",
                f"paths.output_dir={output_dir.as_posix()}",
                "--set",
                "training_3d.lbfgs_steps=0",
            ]
        elif build_mode == "Balanced refresh":
            output_dir = training_root / f"fit_{case_id}"
            command = [
                sys.executable,
                "scripts/train_pinn_3d.py",
                "--config",
                "configs/double_pipe_3d_case_matrix_case001.yaml",
                "--adam-epochs",
                "25",
                "--set",
                f"case_matrix_3d.case_id={case_id}",
                "--set",
                f"paths.output_dir={output_dir.as_posix()}",
                "--set",
                "training_3d.lbfgs_steps=2",
            ]
        else:
            output_dir = training_root / "deep_validation_run"
            command = [
                sys.executable,
                "scripts/validate_final_pinn_3d.py",
                "--config",
                DEFAULT_CONFIG,
                "--output-dir",
                str(output_dir),
                "--no-reuse-existing",
            ]

        with st.spinner("Starting the run..."):
            ok, message = _run_command(command, "Build")
        if ok:
            st.success("The run completed.")
            _record_recent_activity("build", "Model run completed", output_dir)
        else:
            st.error("The run stopped before finishing.")
        with st.expander("Technical log", expanded=not ok):
            st.code(message)
        st.session_state["last_training_dir"] = str(output_dir)

    latest_dir = st.session_state.get("last_training_dir")
    if latest_dir:
        training_dir = Path(latest_dir)
        history_path = training_dir / "training_history_3d.csv"
        checkpoint_path = training_dir / "checkpoints" / "best_model_3d.pt"
        st.write("")
        _render_section_intro(
            "Latest run",
            "Your most recent build output",
            "If the run produced a training history or a fresh model file, they appear below.",
        )
        if history_path.exists():
            history = _load_csv(history_path)
            value_columns = [
                column
                for column in history.columns
                if any(token in column.lower() for token in ["total", "val", "surface"])
            ]
            if value_columns:
                figure = px.line(
                    history,
                    x="epoch",
                    y=value_columns[:3],
                    title="Run progress",
                )
                figure.update_layout(
                    margin=dict(l=18, r=18, t=50, b=18),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="",
                )
                st.plotly_chart(figure, use_container_width=True)
        if checkpoint_path.exists():
            st.download_button(
                "Download latest model file",
                data=checkpoint_path.read_bytes(),
                file_name=checkpoint_path.name,
                use_container_width=True,
            )


def _render_inside_view(case_manifest: pd.DataFrame) -> None:
    _render_section_intro(
        "Inside view",
        "Explore the exchanger from the inside",
        "Create an interior slice view for a preset or review how smoothly temperature passes through the wall from one side to the other.",
    )

    case_id = st.selectbox(
        "Preset",
        options=case_manifest["case_id"].tolist(),
        index=case_manifest["case_id"].tolist().index(
            _current_case(case_manifest, fallback_index=len(case_manifest) - 1)
        ),
        format_func=lambda value: _scenario_label(case_manifest, value),
        key="inside_case",
    )
    _remember_case(case_id)
    defaults = _load_case_inputs(case_manifest, case_id)

    with st.form("inside_form"):
        top_left, top_right = st.columns(2)
        hot_temp = top_left.slider(
            "Hot-side starting temperature [K]",
            min_value=300.0,
            max_value=314.0,
            step=0.5,
            value=float(defaults["Th_in_K"]),
        )
        cold_temp = top_right.slider(
            "Cold-side starting temperature [K]",
            min_value=282.0,
            max_value=289.0,
            step=0.5,
            value=float(defaults["Tc_in_K"]),
        )
        flow_left, flow_right = st.columns(2)
        hot_flow = flow_left.slider(
            "Hot-side flow [m/s]",
            min_value=0.5,
            max_value=1.5,
            step=0.1,
            value=float(defaults["uh_in_mps"]),
        )
        cold_flow = flow_right.slider(
            "Cold-side flow [m/s]",
            min_value=0.5,
            max_value=1.5,
            step=0.1,
            value=float(defaults["uc_in_mps"]),
        )
        button_left, button_right = st.columns(2)
        export_now = button_left.form_submit_button("Create slice view")
        audit_now = button_right.form_submit_button("Review temperature handoff")

    audit_dir = GUI_RUN_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    export_csv = audit_dir / f"{case_id}_interior_fields.csv"
    audit_json = audit_dir / f"{case_id}_interior_audit.json"

    if export_now:
        command = [
            sys.executable,
            "scripts/export_interior_fields_3d.py",
            "--config",
            DEFAULT_CONFIG,
            "--checkpoint",
            DEFAULT_CHECKPOINT,
            "--Th-in",
            str(hot_temp),
            "--Tc-in",
            str(cold_temp),
            "--uh-in",
            str(hot_flow),
            "--uc-in",
            str(cold_flow),
            "--output",
            str(export_csv),
        ]
        with st.spinner("Creating the slice view..."):
            ok, message = _run_command(command, "Slice view")
        if ok:
            st.success("The slice view is ready.")
            _record_recent_activity("slice", "Inside view exported", export_csv)
        else:
            st.error("The slice view could not be created.")
        with st.expander("Technical log", expanded=not ok):
            st.code(message)

    if audit_now:
        command = [
            sys.executable,
            "scripts/audit_interior_physics_3d.py",
            "--config",
            DEFAULT_CONFIG,
            "--checkpoint",
            DEFAULT_CHECKPOINT,
            "--Th-in",
            str(hot_temp),
            "--Tc-in",
            str(cold_temp),
            "--uh-in",
            str(hot_flow),
            "--uc-in",
            str(cold_flow),
            "--output-json",
            str(audit_json),
        ]
        with st.spinner("Reviewing the temperature handoff..."):
            ok, message = _run_command(command, "Inside view review")
        if ok:
            st.success("The review is ready.")
            _record_recent_activity("audit", "Inside review created", audit_json)
        else:
            st.error("The review could not be completed.")
        with st.expander("Technical log", expanded=not ok):
            st.code(message)

    if audit_json.exists():
        audit = _load_json(audit_json)
        metrics = audit["interface_residuals"]
        hot_value = float(metrics["temp_hot_wall"]["rmse"])
        cold_value = float(metrics["temp_wall_cold"]["rmse"])
        order_value = float(metrics["ordering"]["rmse"])
        hot_label, hot_tone = _quality_band(hot_value, good=2.0, watch=4.0)
        cold_label, cold_tone = _quality_band(cold_value, good=2.0, watch=4.0)
        order_label = "Order looks right" if order_value == 0.0 else "Order needs review"
        order_tone = "good" if order_value == 0.0 else "review"

        cards = st.columns(3)
        with cards[0]:
            _render_metric_tile("Inner handoff", f"{hot_value:.2f} K", "Average mismatch between hot fluid and the wall.")
            st.markdown(_status_badge(hot_label, hot_tone), unsafe_allow_html=True)
        with cards[1]:
            _render_metric_tile("Outer handoff", f"{cold_value:.2f} K", "Average mismatch between the wall and the cold side.")
            st.markdown(_status_badge(cold_label, cold_tone), unsafe_allow_html=True)
        with cards[2]:
            _render_metric_tile("Flow order", order_label, "Checks whether one side cools while the other side warms.")
            st.markdown(_status_badge(order_label, order_tone), unsafe_allow_html=True)

        st.plotly_chart(_temperature_band_figure(audit), use_container_width=True)

    if export_csv.exists():
        frame = _load_csv(export_csv)
        domains = frame["domain"].dropna().unique().tolist()
        control_left, control_right = st.columns(2)
        domain = control_left.selectbox("Region", domains, key="inside_domain")
        z_values = np.sort(frame.loc[frame["domain"] == domain, "z"].unique())
        default_index = int(len(z_values) // 2)
        selected_z = control_right.select_slider(
            "Slice depth",
            options=[float(value) for value in z_values],
            value=float(z_values[default_index]),
            key="inside_slice_z",
        )
        st.plotly_chart(_interior_slice_figure(frame, domain, float(selected_z)), use_container_width=True)
        st.download_button(
            "Download slice table",
            data=export_csv.read_bytes(),
            file_name=export_csv.name,
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Heat Flow Studio",
        page_icon="H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    GUI_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    case_manifest = _load_csv(CASE_MANIFEST_PATH)

    st.sidebar.markdown("### Heat Flow Studio")
    st.sidebar.markdown('<div class="sidebar-chip">No-code workspace</div>', unsafe_allow_html=True)
    st.sidebar.write(
        "Explore presets, test new operating conditions, start a fresh run, or open an inside view without working through scripts."
    )
    st.sidebar.markdown("#### Current preset")
    current_case = _current_case(case_manifest)
    st.sidebar.write(_scenario_label(case_manifest, current_case))

    quick_cols = st.sidebar.columns(2)
    if quick_cols[0].button("Presets", use_container_width=True):
        _goto_page("Preset Library")
    if quick_cols[1].button("Preview", use_container_width=True):
        _goto_page("Try a Setup")
    if quick_cols[0].button("Build", use_container_width=True):
        _goto_page("Model Workshop")
    if quick_cols[1].button("Inside", use_container_width=True):
        _goto_page("Inside View")

    st.sidebar.markdown("#### Need a simple path?")
    st.sidebar.markdown(
        """
        <div class="soft-note">
        1. Pick a preset<br>
        2. Generate a preview<br>
        3. Open the inside view
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_options = ["Welcome", "Preset Library", "Try a Setup", "Model Workshop", "Inside View"]
    current_page = st.session_state.get("studio_page", "Welcome")
    page = st.radio(
        "Navigation",
        page_options,
        index=page_options.index(current_page) if current_page in page_options else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["studio_page"] = page

    st.write("")
    if page == "Welcome":
        _render_home(case_manifest)
    elif page == "Preset Library":
        _render_preset_library(case_manifest)
    elif page == "Try a Setup":
        _render_try_setup(case_manifest)
    elif page == "Model Workshop":
        _render_model_workshop(case_manifest)
    else:
        _render_inside_view(case_manifest)


if __name__ == "__main__":
    main()
