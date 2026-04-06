# Synthetic COMSOL-style PINN dataset

This package was generated as a **synthetic surrogate dataset** from the uploaded run order and the attached COMSOL report/CSV files.
It mirrors the **requested export structure and operating-point order** without running a new COMSOL solve.

## What is included
- 16 stationary case folders (`case_001` ... `case_016`)
- Per-case files:
  - hot_volume.csv
  - cold_volume.csv
  - wall_volume.csv
  - hot_inlet.csv
  - hot_outlet.csv
  - cold_inlet.csv
  - cold_outlet.csv
  - hot_wall_interface.csv
  - wall_cold_interface.csv
  - globals.csv
- Root summary:
  - case_manifest.csv

## How the surrogate was built
- Baseline and pilot operating points follow the uploaded run order exactly.
- Geometry extents follow the uploaded report/data envelope:
  - z in [-0.205, 0.205] m
  - x in [-0.02, 0.04] m
  - y in [-0.02, 0.02] m
- Flow/temperature fields are procedurally generated to remain physically consistent with a counterflow double-pipe exchanger:
  - hot stream enters at z = -0.205 m and exits at z = 0.205 m
  - cold stream enters at z = 0.205 m and exits at z = -0.205 m
  - `globals.csv` closes the hot/cold energy balance for each case

## Important note
These files are **synthetic** and suitable for pipeline development, PINN scaffolding, sanity checks, and loader testing.
They are **not** direct exports from COMSOL and no updated `.mph` file is included.
