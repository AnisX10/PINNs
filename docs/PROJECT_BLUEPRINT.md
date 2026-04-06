# Executive Summary

This project turns the supplied COMSOL report and exported temperature fields into a PINN digital twin for a double-pipe heat exchanger. The source material is not a clean textbook benchmark: the engineering reference is a COMSOL Word report rather than a PDF, and the data are raw temperature-point exports rather than labeled sensor histories. Even with that limitation, the files are rich enough to define a realistic reduced-order inverse PINN.

The strongest interpretation is a counter-current double-pipe exchanger with:

- Inner hot tube extending to `z = ±0.205 m`
- Shared heat-transfer section for the annulus over `z = ±0.165 m`
- Approximate inner-tube radius `r_h ≈ 0.011 m`
- Approximate annulus outer radius `r_o ≈ 0.020 m`
- Water as the fluid, `1 m/s` inlet velocity on both streams, hot inlet `303 K`, cold inlet `277 K`
- Stationary turbulent nonisothermal flow solved in COMSOL with `k-ω` RANS and Kays-Crawford turbulent heat transport

The implementation in this folder uses two coupled axial temperature fields:

- `T_h(z,t)` for the hot inner tube
- `T_c(z,t)` for the cold annulus

The effective coupling term is only active in the shared overlap region. Unknown transfer behavior is captured with a learnable `U`, and the code is structured to extend that into `U(t)` or `R_f(t)` when true multi-snapshot degradation data become available.

# PDF and Dataset Interpretation

## Document reality check

The user asked for a PDF-based analysis, but the attached authoritative reference was actually a COMSOL Word export:

- `docs/reference_report.docx`

That mismatch is explicit and important. The project still treats the COMSOL report as authoritative because it contains the exact physics interfaces, solver mode, materials, and boundary conditions used to generate the simulation data.

## What the COMSOL report establishes

Observed directly from the report:

- Model name: `Heat exchanger AL.mph`
- Solver type: stationary
- Physics: `Heat Transfer in Fluids`, `Turbulent Flow, k-ω`, `Nonisothermal Flow`
- Fluid material: water
- Flow model: turbulent RANS `k-ω`
- Heat transport turbulence model: `Kays - Crawford`
- Thermal wall function: `Standard`
- Mesh vertices: `27023`
- Imported CAD source: `Part11.SLDPRT`

Boundary conditions extracted from the report:

- Hot thermal inflow temperature: `303 K` on boundaries `8–12`
- Cold thermal inflow temperature: `277 K` on boundaries `13–14`
- Reference/open-boundary upstream temperature: `293.15 K`
- Velocity inlet on boundary `10`: `1 m/s`
- Velocity inlet on boundaries `13–14`: `1 m/s`
- Pressure/open outlet groups: boundaries `15–16` and boundary `7`
- Initial temperature: `293.15 K`

Material and modeling notes:

- Water occupies the fluid domain(s)
- Boundary material assignments include aluminum and copper selections
- No solid wall thickness data are directly recoverable from the export
- The reduced-order twin should therefore infer an effective `U` rather than assume a precise wall-resistance stackup

## What the CSV dataset establishes

Three CSVs were provided:

- `Datasol1.csv`
- `Datasol2.csv`
- `DataEW.csv`

Observed structure:

- All three files are COMSOL-style CSV exports with metadata rows prefixed by `%`
- Numeric columns are `x, y, z, T`
- Units are meters for `x, y, z` and Kelvin for `T`

File-specific interpretation:

1. `Datasol1.csv`
   - Shape: `9510 x 4`
   - Non-uniform temperature field, `T in [276.94, 303.23] K`
   - This is the usable stationary solution export

2. `DataEW.csv`
   - Exactly identical to `Datasol1.csv`
   - This is a duplicate of the same stationary field, not an independent scenario

3. `Datasol2.csv`
   - Shape: `27023 x 4`
   - Temperature essentially constant at `293.15 K`
   - `27023` matches the COMSOL mesh-vertex count in the report
   - Best interpretation: geometry/mesh-support export at the initial/reference temperature

## Geometry inferred from the point cloud

From the CSV coordinates and face slices:

- Hot stream exists to `z = ±0.205 m`
- Cold/exchange region exists to `z = ±0.165 m`
- Hot inlet face at `z = +0.205 m` is uniformly `303 K`
- Hot outlet face at `z = -0.205 m` is around `288 K` mean
- Cold-side annulus is colder near negative `z` and warmer near positive `z`

This supports the following engineering interpretation:

- Counter-current flow
- Hot fluid enters from positive `z` and exits at negative `z`
- Cold fluid enters near negative `z` and exits near positive `z`
- The hot tube has entrance and exit extensions beyond the shared heat-transfer region

## Data quality and usability

- Missing values: none observed
- Duplicate rows inside each file: none
- Duplicate operating-condition files: `DataEW.csv` duplicates `Datasol1.csv`
- Noise level: negligible numerical noise; the exports are CFD, not plant sensors
- Time dependence: none in the provided solution file
- Practical implication: true fouling evolution or parameter drift cannot be identified from this single operating point alone

# Physics Formulation

## Reduced-order state variables

- `T_h(z,t)`: bulk hot-stream temperature in the inner tube
- `T_c(z,t)`: bulk cold-stream temperature in the annulus
- `U`: effective overall heat-transfer coefficient

Optional future extension:

- `R_f(t)` or separate `R_{f,h}(t), R_{f,c}(t)` for fouling
- `T_w(z,t)` if a wall-state model is later introduced

## Coordinate choice

Use the physical axial coordinate `z` from the COMSOL export, with:

- Hot domain: `z in [-L_h/2, L_h/2]`, `L_h = 0.41 m`
- Cold domain: `z in [-L_c/2, L_c/2]`, `L_c = 0.33 m`

Define an overlap mask:

```math
M(z)=
\begin{cases}
1, & |z|\le L_c/2 \\
0, & |z|>L_c/2
\end{cases}
```

This is critical because the hot tube extends beyond the heat-transfer region.

## Governing equations

For the hot stream, which flows in the negative `z` direction:

```math
\rho_h A_h c_{p,h}\frac{\partial T_h}{\partial t}
- \dot{m}_h c_{p,h}\frac{\partial T_h}{\partial z}
- \rho_h A_h c_{p,h}\alpha_h \frac{\partial^2 T_h}{\partial z^2}
+ U P_i M(z)\left(T_h - T_c\right)=0
```

For the cold stream, which flows in the positive `z` direction:

```math
\rho_c A_c c_{p,c}\frac{\partial T_c}{\partial t}
+ \dot{m}_c c_{p,c}\frac{\partial T_c}{\partial z}
- \rho_c A_c c_{p,c}\alpha_c \frac{\partial^2 T_c}{\partial z^2}
- U P_i\left(T_h - T_c\right)=0,
\quad z \in [-L_c/2,L_c/2]
```

Where:

- `A_h = \pi r_h^2`
- `A_c = \pi (r_o^2-r_h^2)`
- `P_i = 2\pi r_h`
- `\dot{m}_h = \rho_h A_h v_h`
- `\dot{m}_c = \rho_c A_c v_c`

## Term-by-term interpretation

- Convection:
  - `-\dot{m}_h c_p dT_h/dz` for the hot stream because it moves toward decreasing `z`
  - `+\dot{m}_c c_p dT_c/dz` for the cold stream because it moves toward increasing `z`
- Diffusion:
  - `-\rho A c_p \alpha d^2T/dz^2`
  - This acts as effective axial dispersion or turbulent mixing in the reduced model
- Inter-stream exchange:
  - `U P_i (T_h-T_c)`
  - This lumps wall conduction plus convective film effects into an inferable effective `U`
- Learned quantities:
  - `U`
  - Optionally `\alpha_h`, `\alpha_c`
  - Future `R_f(t)`

## Boundary conditions

Hot inlet:

```math
T_h(z=+0.205,t)=303 \text{ K}
```

Cold inlet:

```math
T_c(z=-0.165,t)=277 \text{ K}
```

Outflow conditions for the diffusion-augmented reduced model:

```math
\frac{\partial T_h}{\partial z}(z=-0.205,t)=0
```

```math
\frac{\partial T_c}{\partial z}(z=+0.165,t)=0
```

Initial condition for transient extension:

```math
T_h(z,0)=T_c(z,0)=293.15 \text{ K}
```

That initial condition is directly supported by `Datasol2.csv`.

## Wall-state extension

If a future wall temperature is introduced:

```math
C_w \frac{\partial T_w}{\partial t}
= h_h P_i (T_h-T_w)+h_c P_o (T_c-T_w)+k_w A_w \frac{\partial^2 T_w}{\partial z^2}
```

But the present data do not support reliable identification of separate `h_h`, `h_c`, wall thickness, and wall conductivity. Therefore the base implementation estimates an effective `U`.

# Data Mapping

## PINN inputs

Base stationary case:

- `z`

Extended digital twin:

- `z`
- `t`
- `T_{h,in}`
- `T_{c,in}`
- `v_h`
- `v_c`

## PINN outputs

- `T_h(z,t)`
- `T_c(z,t)`
- `U` as a learnable global parameter

## Measured outputs from the provided data

- Hot-stream axial profile reconstructed from points with `r <= 0.011 m`
- Cold-stream axial profile reconstructed from points with `0.011 < r <= 0.020 m` inside the shared region

## Boundary-condition mapping

- `303 K` hot inlet from the COMSOL report and confirmed by the `z = +0.205 m` face in `Datasol1.csv`
- `277 K` cold inlet from the COMSOL report
- `293.15 K` from `Datasol2.csv` used as the transient initial condition

## Collocation-point mapping

- `Datasol2.csv` is ideal for reduced-order collocation support because it contains the full `27023` mesh vertices
- The preprocessing pipeline classifies those points into hot/cold geometry subsets and reuses their `z` coordinates as physics collocation support

## Unknown parameters to infer

- Primary: `U`
- Secondary: effective axial diffusivities `\alpha_h`, `\alpha_c`
- Future: fouling resistance `R_f(t)`

## Preprocessing steps

1. Strip COMSOL `%` metadata rows.
2. Parse `x, y, z, T`.
3. Compute `r = sqrt(x^2 + y^2)`.
4. Classify points into hot inner-tube and cold annulus regions.
5. Bin along `z` to build denoised axial supervision profiles.
6. Normalize `z` to `[-1,1]` for network input stability.
7. Use profile bins for data supervision and mesh-derived `z` support for PDE residual sampling.

## Train and validation split

For the current single-case stationary dataset, use profile bins rather than raw points:

- Training: random `80%` of hot bins and `80%` of cold bins
- Validation: remaining `20%`

For future multi-condition datasets, split by operating condition rather than by random point.

# Network Architecture

Use two separate MLPs with the same input encoding:

- `f_h(z,t) -> T_h`
- `f_c(z,t) -> T_c`

Recommended base architecture:

- Hidden layers: `6`
- Hidden width: `128`
- Activation: `tanh`
- Fourier features: `16`
- Fourier scale: `4.0`
- Initialization: Xavier

Why this architecture:

- Separate heads avoid forcing identical representation structure on physically different streams
- `tanh` is robust for PINNs because it is smooth and differentiable
- Fourier features help with entrance-region curvature without requiring a very deep network

# Loss Functions

```math
\mathcal{L}_{total}=
\lambda_d \mathcal{L}_{data}
+\lambda_p \mathcal{L}_{PDE}
+\lambda_b \mathcal{L}_{BC}
+\lambda_i \mathcal{L}_{IC}
+\lambda_r \mathcal{L}_{reg}
```

Data loss:

```math
\mathcal{L}_{data}=
\frac{1}{N_h}\sum_i \left(T_h^{pred}(z_i)-T_h^{data}(z_i)\right)^2
+
\frac{1}{N_c}\sum_j \left(T_c^{pred}(z_j)-T_c^{data}(z_j)\right)^2
```

PDE loss:

```math
\mathcal{L}_{PDE}=
\frac{1}{N_{ph}}\sum_k r_h(z_k,t_k)^2
+
\frac{1}{N_{pc}}\sum_l r_c(z_l,t_l)^2
```

Boundary loss:

```math
\mathcal{L}_{BC}=
\left(T_h(+0.205)-303\right)^2
+
\left(T_c(-0.165)-277\right)^2
+
\left(\partial_z T_h(-0.205)\right)^2
+
\left(\partial_z T_c(+0.165)\right)^2
```

Initial-condition loss:

```math
\mathcal{L}_{IC}=
\frac{1}{N_{ih}}\sum \left(T_h(z,0)-293.15\right)^2
+
\frac{1}{N_{ic}}\sum \left(T_c(z,0)-293.15\right)^2
```

Regularization:

- Soft prior on `U`
- Soft priors on `\alpha_h`, `\alpha_c`
- Optional smoothness penalty for `U(t)` if fouling drift is introduced later

For fouling-aware extension:

```math
\frac{1}{U_{eff}(t)}=\frac{1}{U_{clean}}+R_f(t)
```

and:

```math
\mathcal{L}_{reg,foul}=
\gamma_1 \|R_f(t)\|_2^2+\gamma_2 \left\|\frac{dR_f}{dt}\right\|_2^2
```

Autograd is used for:

- `dT/dz`
- `dT/dt`
- `d²T/dz²`

# Training Strategy

1. Profile the raw COMSOL exports.
2. Reconstruct hot and cold axial profiles from the 3D field.
3. Train a stationary forward PINN with fixed nominal `U`.
4. Turn on inverse estimation for `U`.
5. Turn on learnable `\alpha_h`, `\alpha_c` if residuals remain biased.
6. Extend to transient runs using the `293.15 K` initial condition when time data exist.
7. Add `R_f(t)` only after obtaining multi-snapshot or multi-day operating data.

Optimizer sequence:

- Phase 1: Adam
- Phase 2: L-BFGS

Recommended defaults:

- Adam learning rate: `1e-3`
- Gradient clipping: `1.0`
- Adaptive loss weighting: optional if PDE and data terms become imbalanced

Because the current dataset is a single stationary CFD case, the inverse problem is weakly identifiable if too many parameters are free. The base fit should infer only:

- `U`
- optionally one hot and one cold axial dispersion coefficient

# Experiment Plan

| Experiment | Purpose | Inputs | Outputs | Success criterion | Expected failure mode |
| --- | --- | --- | --- | --- | --- |
| Forward PINN | Reproduce stationary profiles | `z`, fixed BCs | `T_h, T_c` | Low profile RMSE and low PDE residual | Boundary-layer mismatch near entrance regions |
| Inverse PINN for `U` | Estimate effective transfer strength | `z`, BCs, profile data | `T_h, T_c, U` | Stable `U` across random seeds | `U` absorbs geometry/model mismatch |
| Dispersion-aware PINN | Capture entrance effects | `z`, BCs | `T_h, T_c, U, \alpha_h, \alpha_c` | Lower residual near overlap edges | Overfitting through inflated dispersion |
| Transient PINN | Prepare digital twin for future time data | `z, t` | `T_h, T_c` | Stable training with IC term | No benefit with only stationary data |
| Fouling-aware PINN | Track degradation | `z, t`, repeated campaigns | `T_h, T_c, R_f(t)` | `R_f(t)` monotonic or interpretable | Unidentifiable with single snapshot |
| Noisy-data test | Robustness check | injected noise on profiles | predicted temperatures, `U` | graceful error growth | parameter collapse under high noise |
| Missing-data test | Sparse-sensor robustness | masked bins or outlets only | reconstructed fields | outlet error remains acceptable | poor field recovery with too little supervision |
| Unseen operating conditions | Generalization | new inlet temperatures or flow rates | predicted profiles | low error on held-out runs | poor extrapolation beyond training manifold |

# Evaluation Metrics

ML metrics:

- MSE
- RMSE
- MAE
- Relative error
- PDE residual norm

Engineering metrics:

- Hot outlet temperature error
- Cold outlet temperature error
- Estimated `U` consistency across seeds
- Heat-duty mismatch
- LMTD consistency check
- Effectiveness error if outlet data are trusted

For fouling-aware extension:

- `R_f` estimation error
- degradation trend monotonicity

# Implementation Roadmap

Recommended framework: PyTorch

Folder responsibilities:

- `scripts/profile_case.py`: inspect reference and raw COMSOL exports, then build reduced datasets
- `scripts/train_pinn.py`: train the forward and inverse PINN
- `src/pinn_hex/data/comsol.py`: parsing, classification, reduction, summaries
- `src/pinn_hex/physics/double_pipe.py`: geometry and governing-equation constants
- `src/pinn_hex/models/pinn.py`: neural architecture
- `src/pinn_hex/training/losses.py`: PINN residuals and loss assembly
- `src/pinn_hex/training/trainer.py`: training loop and checkpointing

Reproducibility:

- Seed control in training config
- Config-driven experiments
- Saved summaries, plots, and checkpoints

Plotting strategy:

- hot and cold axial profile plots
- data vs PINN prediction plots
- training loss curves
- residual-vs-z plots

# Risks, Limitations, and Assumptions

What is strong:

- Boundary temperatures and velocities come directly from the COMSOL report
- The stationary solution field strongly supports a counter-current two-stream interpretation
- `Datasol2.csv` cleanly supports geometry-aware collocation and transient IC setup

What is weak:

- The source document is not a narrative paper; it is an exported solver report
- The current data contain one stationary operating case only
- `DataEW.csv` is not an extra scenario; it duplicates `Datasol1.csv`
- Full 3D wall geometry and wall thickness are not recoverable enough to separate wall resistance from convective resistance

Assumptions used in this project:

- Counter-current configuration
- Hot stream in the inner tube
- Cold stream in the annulus
- Effective overlap length `0.33 m`
- Uniform inlet velocities of `1 m/s`
- Water properties approximated with constant `rho=1000 kg/m³` and `cp=4180 J/(kg·K)` in the reduced-order twin
- `U` represents an effective clean or degraded global transfer coefficient depending on future calibration data

# Suggested Next Steps

1. Run `scripts/profile_case.py` and confirm the reduced hot and cold profiles visually.
2. Run a short smoke training to verify the PINN pipeline.
3. Train the inverse PINN and inspect the learned `U`.
4. Add explicit measured outlet temperatures or CFD exports from additional operating conditions.
5. Only then extend to true fouling detection or time-varying parameter tracking.
