# PINN for a Heat Exchanger

This workspace contains a research-grade scaffold for a Physics-Informed Neural Network.

## Environment

The active local environment created for this folder is `.venv_ssp`, which inherits the scientific stack already installed on this machine.

PowerShell activation:

```powershell
.venv_ssp\Scripts\Activate.ps1
```

If you want a fully isolated environment later, create a fresh venv and install `requirements.txt`.

## Project Layout

```text
configs/                     experiment configuration
data/raw/                    copied source files
data/processed/              reduced-order profiles and summaries
docs/                        reference material and research blueprint
app/                         local Streamlit studio for testing and training
reports/                     generated release dashboard, notes, and figures
scripts/                     runnable entry points
src/pinn_hex/                package code
outputs_3d_case_matrix_*/      locked final model artifacts and validation bundles
```

## Main Commands

Download the shared case matrix dataset baseline (`case_001`) into the repo:

```powershell
.venv_ssp\Scripts\python.exe scripts\download_case_matrix_dataset.py
```

Prepare the case matrix 3D boundary case:

```powershell
.venv_ssp\Scripts\python.exe scripts\prepare_case_3d.py
```

Run a short 3D PINN smoke training on the case matrix baseline:

```powershell
.venv_ssp\Scripts\python.exe scripts\train_pinn_3d.py --adam-epochs 25 --set training_3d.lbfgs_steps=0
```

Run the final 4-fold conditioned case matrix case CV benchmark:

```powershell
.venv_ssp\Scripts\python.exe scripts\run_case_matrix_cv.py --config configs\double_pipe_3d_case_matrix_conditioned_final.yaml --output-root outputs_3d_case_matrix_conditioned_case_cv_final --no-reuse-existing
```

Predict full boundary states for a new operating point with the 4-fold ensemble:

```powershell
.venv_ssp\Scripts\python.exe scripts\predict_boundary_3d.py --Th-in 303.0 --Tc-in 283.5 --uh-in 1.0 --uc-in 1.0 --output outputs_3d_case_matrix_boundary_inference\case_001_boundary_state_predictions.csv
```

Run the locked final holdout validation on reserved cases:

```powershell
.venv_ssp\Scripts\python.exe scripts\validate_final_pinn_3d.py --checkpoint outputs_3d_case_matrix_conditioned_case_cv_final\fold_3\checkpoints\best_model_3d.pt --output-dir outputs_3d_case_matrix_final_validation
```

Run the validation-optimization recipe on the reserved holdout split:

```powershell
.venv_ssp\Scripts\python.exe scripts\train_pinn_3d.py --config configs\double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml --resume-checkpoint outputs_3d_case_matrix_qagg_positivep_hotout\checkpoints\best_model_3d.pt
```

Calibrate pressure gains on training cases and then fine-tune only the wall branch:

```powershell
.venv_ssp\Scripts\python.exe scripts\calibrate_pressure_heads_3d.py --config configs\double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml --checkpoint outputs_3d_case_matrix_qagg_positivep_optphys2_10ep\checkpoints\best_model_3d.pt --output-dir outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal
.venv_ssp\Scripts\python.exe scripts\tune_wall_branch_3d.py --config configs\double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml --checkpoint outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal\checkpoints\best_model_3d.pt --output-dir outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune
```

Fit the final boundary-temperature calibration and validate the calibrated model:

```powershell
.venv_ssp\Scripts\python.exe scripts\fit_boundary_temperature_calibration_3d.py --config configs\double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml --checkpoint outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\checkpoints\best_model_3d.pt --output-json outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\boundary_temperature_calibration.json
.venv_ssp\Scripts\python.exe scripts\validate_final_pinn_3d.py --config configs\double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml --checkpoint outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\checkpoints\best_model_3d.pt --temperature-calibration-json outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\boundary_temperature_calibration.json --output-dir outputs_3d_case_matrix_final_validation_optphys2_10ep_dpcal_walltune_tempcal
```

Generate the final human-readable report bundle:

```powershell
.venv_ssp\Scripts\python.exe scripts\generate_release_assets.py
```

Launch the user-facing local studio:

```powershell
.venv_ssp\Scripts\python.exe app\final_boundary_gui.py
```

Direct Streamlit entry also works:

```powershell
.venv_ssp\Scripts\streamlit.exe run app\boundary_studio.py
```

Profile the original COMSOL case and generate reduced hot/cold axial data:

```powershell
.venv_ssp\Scripts\python.exe scripts\profile_case.py --config configs\double_pipe_countercurrent.yaml
```

Run a short PINN training smoke test:

```powershell
.venv_ssp\Scripts\python.exe scripts\train_pinn.py --config configs\double_pipe_countercurrent.yaml --adam-epochs 25 --skip-lbfgs
```

Run a longer inverse PINN fit:

```powershell
.venv_ssp\Scripts\python.exe scripts\train_pinn.py --config configs\double_pipe_countercurrent.yaml
```

## Reference Interpretation

The supplied engineering reference was a Word export (`docs/reference_report.docx`), not a PDF. The project treats that COMSOL report as the authoritative problem definition and uses the CSVs as the practical simulation dataset.

The implemented reduced model assumes:

- Inner hot tube and outer cold annulus.
- Counter-current flow.
- Stationary CFD source data with turbulent nonisothermal flow.
- Effective heat-transfer coupling over the shared overlap region only.

The detailed project rationale is in [docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md).

## Case Matrix Dataset Notes

The shared Google Drive case matrix dataset is now the default 3D workflow input. The current shared folders contain boundary CSVs plus `globals.csv` for each case; the volume CSVs referenced in the Drive README were not present in the folder at the time this repo was updated.

The case matrix 3D config is [configs/double_pipe_3d_case_matrix_case001.yaml](configs/double_pipe_3d_case_matrix_case001.yaml). After downloading more cases, switch cases with an override such as:

```powershell
.venv_ssp\Scripts\python.exe scripts\train_pinn_3d.py --set case_matrix_3d.case_id=case_010
```

## Final Case-Matrix Baseline

The locked final conditioned config is [configs/double_pipe_3d_case_matrix_conditioned_final.yaml](configs/double_pipe_3d_case_matrix_conditioned_final.yaml).

The final unseen-case 4-fold CV summary is [outputs_3d_case_matrix_conditioned_case_cv_final/case_cv_summary.json](outputs_3d_case_matrix_conditioned_case_cv_final/case_cv_summary.json) with:

- Mean unseen-case combined RMSE: `0.7238 K`
- Mean hot RMSE: `0.3572 K`
- Mean cold RMSE: `0.3666 K`
- Best case: `case_005` at `0.6381 K`
- Worst case: `case_016` at `1.5067 K`

This model is strong as a case matrix boundary-temperature surrogate. It is not yet validated as a full volumetric field surrogate because the shared dataset still lacks the promised volume CSVs.

## Boundary Inference

Use [scripts/predict_boundary_3d.py](scripts/predict_boundary_3d.py) to export boundary predictions for arbitrary inlet temperatures and velocities. By default it loads the 4 final CV fold checkpoints from [outputs_3d_case_matrix_conditioned_case_cv_final](outputs_3d_case_matrix_conditioned_case_cv_final) and writes:

- a CSV with `u`, `v`, `w`, `p`, and `T` ensemble means and standard deviations on the hot and cold fluid boundaries
- a CSV section for wall-interface temperature mean and standard deviation
- a JSON metadata file with the operating point, grid, geometry, and checkpoint list

The default output root is `outputs_3d_case_matrix_boundary_inference/`.

For the final validation-optimized single-checkpoint path, pass:

- `--checkpoint outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\checkpoints\best_model_3d.pt`
- `--temperature-calibration-json outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune\boundary_temperature_calibration.json`

## Final Holdout Validation

Use [scripts/validate_final_pinn_3d.py](scripts/validate_final_pinn_3d.py) to evaluate the locked final conditioned model on a reserved holdout split and emit:

- held-out boundary predictions
- per-case validation summaries
- one aggregate validation JSON with surface RMSE and physics checks

The default conservative holdout is `case_003`, `case_007`, `case_009`, and `case_016`. This is an internal holdout validation, not an external dataset validation.

## Validation-Optimized Candidate

The strongest validation-focused candidate is built from:

- [configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml](configs/double_pipe_3d_case_matrix_conditioned_validation_optphys.yaml)
- [scripts/calibrate_pressure_heads_3d.py](scripts/calibrate_pressure_heads_3d.py)
- [scripts/tune_wall_branch_3d.py](scripts/tune_wall_branch_3d.py)
- [scripts/fit_boundary_temperature_calibration_3d.py](scripts/fit_boundary_temperature_calibration_3d.py)

Final reserved-holdout summary:

- checkpoint: [outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt](outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/checkpoints/best_model_3d.pt)
- temperature calibration: [outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/boundary_temperature_calibration.json](outputs_3d_case_matrix_qagg_positivep_optphys2_10ep_dpcal_walltune/boundary_temperature_calibration.json)
- report: [outputs_3d_case_matrix_final_validation_optphys2_10ep_dpcal_walltune_tempcal/final_validation_summary.json](outputs_3d_case_matrix_final_validation_optphys2_10ep_dpcal_walltune_tempcal/final_validation_summary.json)
- combined boundary RMSE: `0.7802 K`
- mean `Q_total` relative error: `8.17%`
- mean energy-balance gap: `11.68%`
- mean hot / cold pressure-drop relative error: `14.55% / 14.81%`
- mean hot / cold interface temperature RMSE: `2.60 K / 3.22 K`

This is the current best boundary-validated PINN candidate in the repo and it meets the `< 0.8 K` strict holdout RMSE target on the reserved internal split. It is still not a fully validated volumetric field model because the case matrix dataset does not include interior volume fields.

## Release Layer

The repo now includes a human-readable release layer in `reports/`:

- `reports/index.html`: quick dashboard
- `reports/final_boundary_model_summary.md`: concise release notes
- `reports/release_manifest.json`: machine-readable artifact map
- `reports/figures/`: summary figures, operating-matrix visuals, and boundary heatmaps

This layer is generated from the locked final artifacts by [scripts/generate_release_assets.py](scripts/generate_release_assets.py).

## Studio

Use [app/final_boundary_gui.py](app/final_boundary_gui.py) as the local Streamlit studio. It is designed as a user-facing workspace rather than a project console:

- browse the preset library and operating maps
- create fresh surface previews for saved or custom operating points
- start a quick check, balanced refresh, or full build from guided options
- open slice views and review the temperature balance across the exchanger

## Missing Interior Fields

Do not generate target-derived case matrix interior labels from the same operating point and then call that validation. That would leak the answer back into the benchmark.

The defensible fallback now in the repo is:

- export dense model-predicted pseudo-volume fields for visualization and inspection
- run an interior physics-consistency audit over PDE and interface residuals

Use [scripts/export_interior_fields_3d.py](scripts/export_interior_fields_3d.py) when you need a dense interior field from the locked final model:

```powershell
.venv_ssp\Scripts\python.exe scripts\export_interior_fields_3d.py --case-id case_016 --output outputs_3d_case_matrix_interior_probe\case_016_interior_fields.csv
```

Use [scripts/audit_interior_physics_3d.py](scripts/audit_interior_physics_3d.py) when you need a physics-consistency report over interior probe points:

```powershell
.venv_ssp\Scripts\python.exe scripts\audit_interior_physics_3d.py --case-id case_016 --output-json outputs_3d_case_matrix_interior_probe\case_016_interior_physics_audit.json
```

The export output is not independent ground truth. It is suitable for plots, qualitative inspection, and downstream engineering review. The audit JSON is the honest fallback signal for missing interior volume CSVs.
