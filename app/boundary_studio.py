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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.cdnfonts.com/css/aeonik');
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stHeaderActionElements"], [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        html, body, [class*="st-"], [data-testid="stAppViewContainer"] * {
            font-family: "Aeonik", "Segoe UI", sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(215, 226, 240, 0.48), transparent 24%),
                radial-gradient(circle at bottom right, rgba(249, 230, 213, 0.42), transparent 24%),
                linear-gradient(180deg, #f8f5ef 0%, #ffffff 44%, #f7fafc 100%);
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102a43 0%, #132f4c 55%, #0f2238 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        .hero-card, .metric-card, .panel-card {
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 22px 54px rgba(16, 42, 67, 0.08);
            padding: 1.15rem 1.2rem;
        }
        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.12), transparent 28%),
                linear-gradient(135deg, #102a43 0%, #143d5c 58%, #1d577b 100%);
            color: white;
            padding: 1.55rem 1.6rem;
        }
        .hero-card h1 {
            margin: 0;
            font-size: 2.15rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
        }
        .hero-card p {
            margin: 0.55rem 0 0;
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
        }
        .metric-card .label {
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #486581;
        }
        .metric-card .value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #102a43;
            margin-top: 0.25rem;
        }
        .metric-card .note {
            color: #627d98;
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }
        .pill {
            display: inline-block;
            padding: 0.24rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.14);
            color: #ffffff;
            margin-bottom: 0.55rem;
        }
        .section-title {
            font-size: 1.28rem;
            font-weight: 700;
            color: #102a43;
            margin-bottom: 0.2rem;
        }
        .section-note {
            color: #627d98;
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
        }
        .stButton button, .stDownloadButton button {
            border-radius: 999px;
            border: 0;
            background: #102a43;
            color: white;
            font-weight: 600;
            padding: 0.68rem 1.08rem;
            box-shadow: 0 12px 30px rgba(16, 42, 67, 0.16);
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background: #143d5c;
            color: white;
        }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            border-radius: 18px !important;
            border: 1px solid rgba(16, 42, 67, 0.10) !important;
            min-height: 3rem !important;
            background: rgba(255,255,255,0.92) !important;
        }
        .stRadio > div {
            gap: 1rem;
        }
        .stRadio [role="radiogroup"] {
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .stRadio [role="radio"] {
            border-radius: 999px;
            padding: 0.62rem 0.95rem;
            background: rgba(16, 42, 67, 0.06);
            border: 1px solid rgba(16, 42, 67, 0.08);
        }
        .stRadio [aria-checked="true"] {
            background: #102a43 !important;
            color: #ffffff !important;
            border-color: #102a43 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.55rem 0.95rem;
            background: rgba(16, 42, 67, 0.06);
        }
        .stTabs [aria-selected="true"] {
            background: #102a43 !important;
            color: white !important;
        }
        .soft-note {
            color: #627d98;
            font-size: 0.92rem;
            line-height: 1.6;
        }
        .feature-card {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(16, 42, 67, 0.08);
            box-shadow: 0 18px 44px rgba(16, 42, 67, 0.06);
        }
        .feature-card h3 {
            margin: 0 0 0.35rem;
            color: #102a43;
            font-size: 1.02rem;
        }
        .feature-card p {
            margin: 0;
            color: #627d98;
            font-size: 0.92rem;
            line-height: 1.6;
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


def _metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="label">{title}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
          <h3>{title}</h3>
          <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _scenario_title(case_id: str) -> str:
    suffix = str(case_id).replace("case_", "")
    return f"Preset {suffix}"


def _scenario_label(case_manifest: pd.DataFrame, case_id: str) -> str:
    row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
    preset_name = _scenario_title(case_id)
    return (
        f"{preset_name} | {float(row['Th_in_K']):.1f} K / {float(row['Tc_in_K']):.1f} K | "
        f"{float(row['uh_in_mps']):.1f} / {float(row['uc_in_mps']):.1f} m/s"
    )


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


def _surface_heatmap(frame: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    plot_frame = frame.copy()
    if "phi_rad" not in plot_frame.columns:
        plot_frame["phi_rad"] = np.arctan2(plot_frame["y"], plot_frame["x"])
    plot_frame["phi_deg"] = np.degrees(plot_frame["phi_rad"])
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
    fig = px.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        color_continuous_scale="Turbo",
        labels={"x": "Axial position z [m]", "y": "Circumference [deg]", "color": value_column},
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title=value_column),
    )
    return fig


def _operating_matrix_figure(case_manifest: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        case_manifest,
        x="Th_in_K",
        y="Tc_in_K",
        color="Q_total",
        size="effectiveness",
        symbol=case_manifest["uh_in_mps"].astype(str) + "/" + case_manifest["uc_in_mps"].astype(str),
        text="case_id",
        color_continuous_scale="Tealgrn",
        labels={
            "Th_in_K": "Hot inlet [K]",
            "Tc_in_K": "Cold inlet [K]",
            "Q_total": "Transfer level [W]",
            "symbol": "Flow setting [m/s]",
        },
    )
    figure.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
    figure.update_layout(
        title="Preset Library",
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _q_heatmap(case_manifest: pd.DataFrame) -> go.Figure:
    heat = (
        case_manifest.groupby(["Th_in_K", "Tc_in_K"], as_index=False)["Q_total"]
        .mean()
        .pivot(index="Tc_in_K", columns="Th_in_K", values="Q_total")
        .sort_index(ascending=True)
    )
    fig = px.imshow(
        heat,
        aspect="auto",
        origin="lower",
        color_continuous_scale="Sunsetdark",
        labels={"x": "Hot inlet [K]", "y": "Cold inlet [K]", "color": "Average transfer level [W]"},
    )
    fig.update_layout(
        title="Operating Map",
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _load_case_inputs(case_manifest: pd.DataFrame, case_id: str) -> dict[str, float]:
    row = case_manifest.loc[case_manifest["case_id"] == case_id].iloc[0]
    return {
        "Th_in_K": float(row["Th_in_K"]),
        "Tc_in_K": float(row["Tc_in_K"]),
        "uh_in_mps": float(row["uh_in_mps"]),
        "uc_in_mps": float(row["uc_in_mps"]),
    }


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
    figure.update_traces(marker=dict(size=6, opacity=0.92))
    figure.update_layout(
        title=f"{domain.replace('_', ' ').title()} | Slice view at z = {z_value:.3f} m",
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_scaleanchor="x",
    )
    return figure


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


def _render_overview() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="pill">User Workspace</div>
          <h1>Heat Exchanger Studio</h1>
          <p>Browse ready-made operating presets, create a fresh temperature preview, start a new build, or open a slice view from one calm interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feature_cols = st.columns(3)
    with feature_cols[0]:
        _feature_card("Browse presets", "Start with the built-in library to compare operating conditions and inspect real surface patterns.")
        if st.button("Open library", key="home_library", use_container_width=True):
            _goto_page("Scenario Library")
    with feature_cols[1]:
        _feature_card("Create a preview", "Generate a fresh wall-temperature preview for any operating point and download the result in one step.")
        if st.button("Create preview", key="home_preview", use_container_width=True):
            _goto_page("Live Preview")
    with feature_cols[2]:
        _feature_card("Build and explore", "Launch a quick refresh or open a slice view without touching any backend files.")
        action_left, action_right = st.columns(2)
        if action_left.button("Start run", key="home_build", use_container_width=True):
            _goto_page("Build")
        if action_right.button("Open slice", key="home_flow", use_container_width=True):
            _goto_page("Flow View")

    st.markdown('<div class="section-title">Start Here</div><div class="section-note">These views help you get oriented before you move into preview or a new build.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.image(str(ROOT / "reports" / "figures" / "dataset_operating_matrix.png"), use_container_width=True)
    with right:
        st.image(str(ROOT / "reports" / "figures" / "dataset_boundary_heatmaps.png"), use_container_width=True)

    st.markdown(
        """
        <div class="panel-card">
          <div class="section-title">A simple way to use the studio</div>
          <div class="soft-note">
            1. Open <strong>Scenario Library</strong> to browse presets and surface maps.<br>
            2. Use <strong>Live Preview</strong> to test a preset or your own operating point.<br>
            3. Use <strong>Build</strong> when you want a fresh run without touching the backend.<br>
            4. Open <strong>Flow View</strong> to create a slice view or review the temperature balance.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataset_explorer(case_manifest: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Scenario Library</div><div class="section-note">Browse the built-in operating presets and inspect the surface patterns they produce.</div>', unsafe_allow_html=True)
    top_left, top_right = st.columns([1.2, 1.0])
    with top_left:
        st.plotly_chart(_operating_matrix_figure(case_manifest), use_container_width=True)
    with top_right:
        st.plotly_chart(_q_heatmap(case_manifest), use_container_width=True)

    selected_case = st.selectbox(
        "Preset",
        options=case_manifest["case_id"].tolist(),
        index=case_manifest["case_id"].tolist().index(_current_case(case_manifest)),
        key="dataset_case",
        format_func=lambda case_id: _scenario_label(case_manifest, case_id),
    )
    _remember_case(selected_case)
    inputs = _load_case_inputs(case_manifest, selected_case)
    scenario_title = _scenario_title(selected_case)
    cards = st.columns(4)
    with cards[0]:
        _metric_card("Hot inlet", f"{inputs['Th_in_K']:.1f} K", "Selected preset")
    with cards[1]:
        _metric_card("Cold inlet", f"{inputs['Tc_in_K']:.1f} K", "Selected preset")
    with cards[2]:
        _metric_card("Hot flow", f"{inputs['uh_in_mps']:.1f} m/s", "Selected preset")
    with cards[3]:
        _metric_card("Cold flow", f"{inputs['uc_in_mps']:.1f} m/s", "Selected preset")

    hot_frame = _load_csv(DATA_ROOT / selected_case / "hot_wall_interface.csv")
    cold_frame = _load_csv(DATA_ROOT / selected_case / "wall_cold_interface.csv")
    tabs = st.tabs(["Temperature maps", "Heat flow maps"])
    with tabs[0]:
        hot_cols, cold_cols = st.columns(2)
        with hot_cols:
            st.plotly_chart(_surface_heatmap(hot_frame, "T", f"{scenario_title} | Inner surface temperature"), use_container_width=True)
        with cold_cols:
            st.plotly_chart(_surface_heatmap(cold_frame, "T", f"{scenario_title} | Outer surface temperature"), use_container_width=True)
    with tabs[1]:
        hot_cols, cold_cols = st.columns(2)
        with hot_cols:
            st.plotly_chart(_surface_heatmap(hot_frame, "qn", f"{scenario_title} | Inner surface heat flow"), use_container_width=True)
        with cold_cols:
            st.plotly_chart(_surface_heatmap(cold_frame, "qn", f"{scenario_title} | Outer surface heat flow"), use_container_width=True)


def _render_prediction_lab(case_manifest: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Live Preview</div><div class="section-note">Choose a saved preset or enter your own conditions, then create a fresh surface preview.</div>', unsafe_allow_html=True)
    source_mode = st.radio("Start from", ["Preset library", "Custom setup"], horizontal=True, key="predict_mode")
    if source_mode == "Preset library":
        default_case = _current_case(case_manifest)
        case_id = st.selectbox(
            "Preset",
            case_manifest["case_id"].tolist(),
            index=case_manifest["case_id"].tolist().index(default_case),
            key="predict_case",
            format_func=lambda value: _scenario_label(case_manifest, value),
        )
        _remember_case(case_id)
        defaults = _load_case_inputs(case_manifest, case_id)
    else:
        case_id = "custom"
        defaults = {"Th_in_K": 303.0, "Tc_in_K": 283.5, "uh_in_mps": 1.0, "uc_in_mps": 1.0}

    with st.form("preview_form"):
        cols = st.columns(4)
        Th = cols[0].number_input("Hot inlet [K]", value=float(defaults["Th_in_K"]), step=0.5, key="pred_Th")
        Tc = cols[1].number_input("Cold inlet [K]", value=float(defaults["Tc_in_K"]), step=0.5, key="pred_Tc")
        uh = cols[2].number_input("Hot flow [m/s]", value=float(defaults["uh_in_mps"]), step=0.1, key="pred_uh")
        uc = cols[3].number_input("Cold flow [m/s]", value=float(defaults["uc_in_mps"]), step=0.1, key="pred_uc")
        submitted = st.form_submit_button("Generate preview")

    prediction_dir = GUI_RUN_ROOT / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = prediction_dir / f"{case_id}_boundary_prediction.csv"
    prediction_json = prediction_csv.with_suffix(".json")

    if submitted:
        command = [
            sys.executable,
            "scripts/predict_boundary_3d.py",
            "--config",
            "configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml",
            "--checkpoint",
            "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt",
            "--temperature-calibration-json",
            "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/boundary_temperature_calibration.json",
            "--Th-in",
            str(Th),
            "--Tc-in",
            str(Tc),
            "--uh-in",
            str(uh),
            "--uc-in",
            str(uc),
            "--output",
            str(prediction_csv),
        ]
        with st.spinner("Creating the preview..."):
            ok, message = _run_command(command, "Preview")
        if ok:
            st.success("Preview is ready.")
        else:
            st.error("The preview could not be created.")
        with st.expander("Activity notes", expanded=not ok):
            st.code(message)

    if prediction_csv.exists():
        frame = _load_csv(prediction_csv)
        hot_outlet = frame.loc[frame["boundary"] == "hot_outlet", "T_pred_mean_K"].mean()
        cold_outlet = frame.loc[frame["boundary"] == "cold_outlet", "T_pred_mean_K"].mean()
        hot_wall = frame.loc[frame["boundary"] == "hot_wall", "T_pred_mean_K"].mean()
        cold_wall = frame.loc[frame["boundary"] == "cold_inner_wall", "T_pred_mean_K"].mean()
        cards = st.columns(4)
        with cards[0]:
            _metric_card("Hot outlet", f"{hot_outlet:.2f} K", "Estimated outlet level")
        with cards[1]:
            _metric_card("Cold outlet", f"{cold_outlet:.2f} K", "Estimated outlet level")
        with cards[2]:
            _metric_card("Inner surface", f"{hot_wall:.2f} K", "Average surface temperature")
        with cards[3]:
            _metric_card("Outer surface", f"{cold_wall:.2f} K", "Average surface temperature")

        tabs = st.tabs(["Surface heatmaps", "Files"])
        with tabs[0]:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(_surface_heatmap(frame.loc[frame["boundary"] == "hot_wall"], "T_pred_mean_K", "Inner surface temperature"), use_container_width=True)
            with right:
                st.plotly_chart(_surface_heatmap(frame.loc[frame["boundary"] == "cold_inner_wall"], "T_pred_mean_K", "Outer surface temperature"), use_container_width=True)
        with tabs[1]:
            st.download_button("Download surface data", data=prediction_csv.read_bytes(), file_name=prediction_csv.name)
            if prediction_json.exists():
                st.download_button("Download summary file", data=prediction_json.read_bytes(), file_name=prediction_json.name)

        st.markdown('<div class="soft-note">Tip: open <strong>Flow View</strong> with the same values if you want to inspect a slice through the exchanger.</div>', unsafe_allow_html=True)


def _render_training_studio(case_manifest: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Build</div><div class="section-note">Choose how much work you want the studio to do, then let it prepare the run for you.</div>', unsafe_allow_html=True)
    with st.form("training_form"):
        preset = st.selectbox(
            "Run style",
            [
                "Quick check",
                "Balanced refresh",
                "Full refresh",
            ],
            key="train_preset",
        )
        case_id = st.selectbox(
            "Operating preset",
            options=case_manifest["case_id"].tolist(),
            index=case_manifest["case_id"].tolist().index(_current_case(case_manifest)),
            key="train_case",
            format_func=lambda value: _scenario_label(case_manifest, value),
        )
        _remember_case(case_id)
        submitted = st.form_submit_button("Start build")
    training_root = GUI_RUN_ROOT / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    if submitted:
        if preset == "Quick check":
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
        elif preset == "Balanced refresh":
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
            output_dir = training_root / "final_holdout_rebuild"
            command = [
                sys.executable,
                "scripts/validate_final_pinn_3d.py",
                "--config",
                "configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml",
                "--output-dir",
                str(output_dir),
                "--no-reuse-existing",
            ]

        with st.spinner("Preparing the build..."):
            ok, message = _run_command(command, "Build")
        if ok:
            st.success("Build completed.")
        else:
            st.error("The build stopped before finishing.")
        with st.expander("Activity notes", expanded=not ok):
            st.code(message)
        st.session_state["last_training_dir"] = str(output_dir)

    training_dir_value = st.session_state.get("last_training_dir")
    if training_dir_value:
        training_dir = Path(training_dir_value)
        history_path = training_dir / "training_history_3d.csv"
        st.markdown('<div class="soft-note">Your latest build is ready below.</div>', unsafe_allow_html=True)
        if history_path.exists():
            history = _load_csv(history_path)
            path_cols = [col for col in history.columns if any(token in col.lower() for token in ["total", "val", "surface"])]
            if path_cols:
                fig = px.line(history, x="epoch", y=path_cols[:3], title="Build progress", markers=False)
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
        checkpoint_path = training_dir / "checkpoints" / "best_model_3d.pt"
        if checkpoint_path.exists():
            st.success("A fresh model file is available from this build.")
            st.download_button("Download model file", data=checkpoint_path.read_bytes(), file_name=checkpoint_path.name)


def _render_audit_studio(case_manifest: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Flow View</div><div class="section-note">Create an interior slice view for a chosen preset or review the temperature balance across the exchanger.</div>', unsafe_allow_html=True)
    case_id = st.selectbox(
        "Preset",
        options=case_manifest["case_id"].tolist(),
        index=case_manifest["case_id"].tolist().index(_current_case(case_manifest, fallback_index=15)),
        key="audit_case",
        format_func=lambda value: _scenario_label(case_manifest, value),
    )
    _remember_case(case_id)
    defaults = _load_case_inputs(case_manifest, case_id)
    with st.form("audit_form"):
        cols = st.columns(4)
        Th = cols[0].number_input("Hot inlet [K]", value=float(defaults["Th_in_K"]), step=0.5, key="audit_Th")
        Tc = cols[1].number_input("Cold inlet [K]", value=float(defaults["Tc_in_K"]), step=0.5, key="audit_Tc")
        uh = cols[2].number_input("Hot flow [m/s]", value=float(defaults["uh_in_mps"]), step=0.1, key="audit_uh")
        uc = cols[3].number_input("Cold flow [m/s]", value=float(defaults["uc_in_mps"]), step=0.1, key="audit_uc")
        left_button, right_button = st.columns(2)
        export_now = left_button.form_submit_button("Create slice view")
        audit_now = right_button.form_submit_button("Review flow balance")

    audit_dir = GUI_RUN_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    export_csv = audit_dir / f"{case_id}_interior_fields.csv"
    audit_json = audit_dir / f"{case_id}_interior_audit.json"

    if export_now:
        command = [
            sys.executable,
            "scripts/export_interior_fields_3d.py",
            "--config",
            "configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml",
            "--checkpoint",
            "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt",
            "--Th-in",
            str(Th),
            "--Tc-in",
            str(Tc),
            "--uh-in",
            str(uh),
            "--uc-in",
            str(uc),
            "--output",
            str(export_csv),
        ]
        with st.spinner("Creating the slice view..."):
            ok, message = _run_command(command, "Slice view")
        if ok:
            st.success("Slice view is ready.")
        else:
            st.error("The slice view could not be created.")
        with st.expander("Activity notes", expanded=not ok):
            st.code(message)

    if audit_now:
        command = [
            sys.executable,
            "scripts/audit_interior_physics_3d.py",
            "--config",
            "configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml",
            "--checkpoint",
            "outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt",
            "--Th-in",
            str(Th),
            "--Tc-in",
            str(Tc),
            "--uh-in",
            str(uh),
            "--uc-in",
            str(uc),
            "--output-json",
            str(audit_json),
        ]
        with st.spinner("Reviewing the flow balance..."):
            ok, message = _run_command(command, "Flow balance review")
        if ok:
            st.success("Flow balance review is ready.")
        else:
            st.error("The flow balance review could not be completed.")
        with st.expander("Activity notes", expanded=not ok):
            st.code(message)

    if audit_json.exists():
        audit = _load_json(audit_json)
        metrics = audit["interface_residuals"]
        ranges = audit["temperature_ranges"]
        cols = st.columns(3)
        with cols[0]:
            _metric_card("Inside surface match", f"{metrics['temp_hot_wall']['rmse']:.2f} K", "Temperature transition")
        with cols[1]:
            _metric_card("Outside surface match", f"{metrics['temp_wall_cold']['rmse']:.2f} K", "Temperature transition")
        with cols[2]:
            ordering = float(metrics["ordering"]["rmse"])
            _metric_card("Flow direction", "Looks right" if ordering == 0.0 else "Review", "Warm side cools while the cool side warms")
        band_labels = ["Hot channel", "Solid wall", "Cold channel"]
        band_values = [
            ranges["hot_fluid_K"]["mean"],
            ranges["wall_K"]["mean"],
            ranges["cold_fluid_K"]["mean"],
        ]
        fig = go.Figure(go.Bar(x=band_labels, y=band_values, marker_color=["#c0392b", "#7c3aed", "#1d6fa5"]))
        fig.update_layout(
            title="Temperature profile across the core",
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Average temperature [K]",
        )
        st.plotly_chart(fig, use_container_width=True)

    if export_csv.exists():
        frame = _load_csv(export_csv)
        domains = frame["domain"].dropna().unique().tolist()
        select_cols = st.columns(2)
        domain = select_cols[0].selectbox("Region", domains, key="slice_domain")
        domain_frame = frame.loc[frame["domain"] == domain].copy()
        z_values = np.sort(domain_frame["z"].unique())
        z_index = int(len(z_values) // 2)
        selected_z = select_cols[1].select_slider(
            "Slice depth",
            options=[float(value) for value in z_values],
            value=float(z_values[z_index]),
            key="slice_z",
        )
        st.plotly_chart(_interior_slice_figure(frame, domain, float(selected_z)), use_container_width=True)
        st.download_button("Download slice data", data=export_csv.read_bytes(), file_name=export_csv.name)


def main() -> None:
    st.set_page_config(page_title="Heat Exchanger Studio", page_icon="H", layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    GUI_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    case_manifest = _load_csv(CASE_MANIFEST_PATH)

    st.sidebar.markdown("## Heat Exchanger Studio")
    st.sidebar.write("A clear front end for browsing presets, creating previews, and starting new runs.")
    st.sidebar.markdown(
        """
        <div class="soft-note">
        Choose a view to match the job:
        <br><br>
        - <strong>Home</strong> for a quick starting point
        <br>
        - <strong>Scenario Library</strong> for preset browsing
        <br>
        - <strong>Live Preview</strong> for new surface maps
        <br>
        - <strong>Build</strong> for a fresh run
        <br>
        - <strong>Flow View</strong> for slices and balance review
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Quick actions")
    quick_cols = st.sidebar.columns(2)
    if quick_cols[0].button("Library", use_container_width=True):
        _goto_page("Scenario Library")
    if quick_cols[1].button("Preview", use_container_width=True):
        _goto_page("Live Preview")
    if quick_cols[0].button("Build", use_container_width=True):
        _goto_page("Build")
    if quick_cols[1].button("Flow View", use_container_width=True):
        _goto_page("Flow View")

    page_options = ["Home", "Scenario Library", "Live Preview", "Build", "Flow View"]
    current_page = st.session_state.get("studio_page", "Home")
    page = st.radio(
        "Workspace",
        page_options,
        index=page_options.index(current_page) if current_page in page_options else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["studio_page"] = page

    if page == "Home":
        _render_overview()
    elif page == "Scenario Library":
        _render_dataset_explorer(case_manifest)
    elif page == "Live Preview":
        _render_prediction_lab(case_manifest)
    elif page == "Build":
        _render_training_studio(case_manifest)
    else:
        _render_audit_studio(case_manifest)


if __name__ == "__main__":
    main()
