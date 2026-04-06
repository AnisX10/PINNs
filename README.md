# Boundary Flow Studio

Boundary Flow Studio is a user-facing workspace for exploring a double-pipe heat exchanger with a physics-informed neural network.

The goal is simple: give someone a clean way to test operating conditions, preview boundary behavior, run guided training, and inspect results without needing to understand the internal project history.

## What You Can Do

- Open a polished local studio for scenario testing and guided workflows.
- Preview boundary temperature and flow behavior for saved or custom operating points.
- Retrain or refresh the model from guided scripts.
- Export boundary predictions for downstream use.
- Generate interior field previews and physics-audit reports for engineering review.
- Use the bundled synthetic dataset and final release artifacts directly from the repo.

## Who This Is For

This repository is built for:

- engineers who want a faster way to explore exchanger behavior
- teams preparing for a future COMSOL or OpenFOAM validation phase
- anyone who needs a packaged PINN workflow instead of a raw research notebook stack

## Quick Start

If you want the fastest path, install the dependencies and launch the studio.

Create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already use the local project environment, that works too:

```powershell
.\.venv_ssp\Scripts\Activate.ps1
```

Launch the studio:

```powershell
python app\final_boundary_gui.py
```

You can also start Streamlit directly:

```powershell
streamlit run app\boundary_studio.py
```

## Main Workflows

### 1. Open the Studio

The local app is the main entry point:

- `Home` gives a clean landing page
- `Scenario Library` helps browse operating presets
- `Live Preview` generates fresh boundary previews
- `Build` runs guided training flows
- `Flow View` opens interior preview and audit tools

Main launcher:

```powershell
python app\final_boundary_gui.py
```

### 2. Run a Boundary Prediction

Use the prediction script when you want a direct export instead of the studio:

```powershell
python scripts\predict_boundary_3d.py --Th-in 303.0 --Tc-in 283.5 --uh-in 1.0 --uc-in 1.0 --output outputs_3d_synthetic_boundary_inference\custom_boundary_prediction.csv
```

This writes a prediction bundle you can inspect or pass to another workflow.

### 3. Train or Refresh the Model

For a direct training run:

```powershell
python scripts\train_pinn_3d.py --config configs\double_pipe_3d_synthetic_conditioned_validation_optphys.yaml
```

For the case-based sweep:

```powershell
python scripts\run_synthetic_case_cv.py --config configs\double_pipe_3d_synthetic_conditioned_final.yaml --output-root outputs_3d_synthetic_conditioned_case_cv_final --no-reuse-existing
```

### 4. Export an Interior Preview

When you want a dense interior field preview:

```powershell
python scripts\export_interior_fields_3d.py --case-id case_016 --output outputs_3d_synthetic_interior_probe\case_016_interior_fields.csv
```

When you want a physics-consistency check:

```powershell
python scripts\audit_interior_physics_3d.py --case-id case_016 --output-json outputs_3d_synthetic_interior_probe\case_016_interior_physics_audit.json
```

## Included in This Repository

This branch is packaged as a clean release snapshot. It includes:

- the local studio app
- the full source code
- training and export scripts
- the synthetic dataset
- final model artifacts
- public reports and figures

You do not need to reconstruct the project from old experiment folders.

## Project Structure

```text
app/                         local user-facing studio
configs/                     reusable configuration files
data/synthetic/              bundled synthetic dataset
docs/                        small supporting reference files
reports/                     human-readable figures and summary pages
scripts/                     runnable training, export, and validation tools
src/pinn_hex/                package source code
outputs_3d_synthetic_*/      released model bundles and generated examples
```

## Model Summary

At the core of the project is a conditioned 3D Physics-Informed Neural Network for a counter-current double-pipe heat exchanger.

In practical terms, the model is used here as:

- a boundary prediction engine
- a scenario exploration tool
- a training and export workflow
- a bridge toward future solver-backed validation

## Reports and Figures

The `reports/` folder contains the public-facing output layer:

- `reports/index.html` for a lightweight summary page
- `reports/final_boundary_model_summary.md` for the release note version
- `reports/release_manifest.json` for the artifact map
- `reports/figures/` for operating maps, heatmaps, and overview figures

You can regenerate those assets with:

```powershell
python scripts\generate_release_assets.py
```

## Dataset

The bundled dataset lives under:

```text
data/synthetic/synthetic_comsol_pinn_dataset/
```

It includes:

- operating-point metadata
- hot and cold inlet and outlet boundary files
- wall interface files
- volume files for the synthetic cases included in this release

If you want to refresh or extend the dataset later, the downloader is still included:

```powershell
python scripts\download_synthetic_dataset.py
```

## Current Scope

This project is packaged to be useful now, while still leaving room for a future external validation stage.

Today it is best used as:

- a polished PINN studio
- a synthetic-data workflow for training and testing
- a boundary prediction tool
- a preparation layer for later COMSOL or OpenFOAM comparison

## Next Step

When you are ready to continue the project with new solver data, this repo is already organized for it:

- keep using the studio for scenario testing
- add new unseen solver exports
- compare them against the existing prediction and audit pipeline
- extend the final model without rebuilding the whole workspace
