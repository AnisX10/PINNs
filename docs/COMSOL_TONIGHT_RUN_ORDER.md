# COMSOL Tonight Run Order

## Objective

Generate a clean, PINN-ready CFD dataset for the double-pipe heat exchanger with:

- multiple stationary operating points
- volumetric hot/cold/wall fields
- separate boundary exports
- actual cold-side field values, not only report values

## Before You Start

Use these two files as the reference:

- [COMSOL_RESTART_CHECKLIST.md](/c:/Users/TL/Desktop/PINN/docs/COMSOL_RESTART_CHECKLIST.md)
- [comsol_operating_matrix_template.csv](/c:/Users/TL/Desktop/PINN/docs/comsol_operating_matrix_template.csv)

## Fastest Safe Execution Order

### 1. Duplicate the current COMSOL model

Create a new model copy for the data-generation campaign.

Recommended naming:

- `heat_exchanger_pinn_dataset.mph`

Do not overwrite your current working model.

### 2. Confirm geometry and domain selections

Before running any sweep, make sure you can select these separately:

- hot-fluid domain
- cold-fluid domain
- wall domain
- hot inlet boundary
- hot outlet boundary
- cold inlet boundary
- cold outlet boundary
- hot-wall interface
- wall-cold interface

If any of these are not separated cleanly, fix selections first.

### 3. Define the four sweep parameters

Create four global parameters:

- `Th_in`
- `Tc_in`
- `uh_in`
- `uc_in`

Set the baseline to:

- `Th_in = 303[K]`
- `Tc_in = 283.5[K]`
- `uh_in = 1[m/s]`
- `uc_in = 1[m/s]`

Important:

- for the cold side, use the actual field-consistent baseline around `283.5 K`
- do not force `277 K` just because it appeared in the report

### 4. Verify one baseline stationary run

Run one stationary case first:

- `Th_in = 303 K`
- `Tc_in = 283.5 K`
- `uh_in = 1.0 m/s`
- `uc_in = 1.0 m/s`

Check:

- solution converges
- temperature field looks physically correct
- no broken selections
- inlet and outlet definitions are correct

Do not start the full sweep until this baseline passes.

### 5. Create export datasets before the full sweep

Set up export nodes now, not after the simulations.

Create exports for these files per case:

- `hot_volume.csv`
- `cold_volume.csv`
- `wall_volume.csv`
- `hot_inlet.csv`
- `hot_outlet.csv`
- `cold_inlet.csv`
- `cold_outlet.csv`
- `hot_wall_interface.csv`
- `wall_cold_interface.csv`
- `globals.csv`

### 6. Put the correct expressions in each export

For hot-fluid and cold-fluid volume exports:

- `x`
- `y`
- `z`
- `T`
- `u`
- `v`
- `w`
- `p`

If available, also export:

- effective thermal conductivity
- density
- viscosity
- turbulent viscosity

For wall volume:

- `x`
- `y`
- `z`
- `T`

If available, also export:

- heat-flux components
- material properties

For fluid boundaries:

- `x`
- `y`
- `z`
- `T`
- `u`
- `v`
- `w`
- `p`

If available, also export:

- normal heat flux
- normal vector components

For `globals.csv`, export:

- `Th_in_bulk`
- `Th_out_bulk`
- `Tc_in_bulk`
- `Tc_out_bulk`
- `m_dot_hot`
- `m_dot_cold`
- `dp_hot`
- `dp_cold`
- `Q_total`
- effectiveness

### 7. Run a pilot set of 4 cases first

Do these four before the whole matrix:

1. baseline: `303, 283.5, 1.0, 1.0`
2. hot high: `311, 283.5, 1.0, 1.0`
3. cold high: `303, 287.0, 1.0, 1.0`
4. both fast: `303, 283.5, 1.4, 1.4`

After these four runs, confirm:

- all exports exist
- file naming is consistent
- CSV columns are correct
- no empty or corrupted export files

### 8. Run the full stationary matrix

Once the pilot is clean, run the full matrix in [comsol_operating_matrix_template.csv](/c:/Users/TL/Desktop/PINN/docs/comsol_operating_matrix_template.csv).

Target tonight:

- minimum acceptable: `8` cases
- good: `12` cases
- ideal: `16` cases

### 9. Use one folder per case

Store outputs like this:

```text
case_001/
  hot_volume.csv
  cold_volume.csv
  wall_volume.csv
  hot_inlet.csv
  hot_outlet.csv
  cold_inlet.csv
  cold_outlet.csv
  hot_wall_interface.csv
  wall_cold_interface.csv
  globals.csv
```

Keep the same structure for every case.

### 10. Save a case manifest

In each case folder, keep the actual parameter values.

If COMSOL does not write this automatically, add them to `globals.csv`:

- `case_id`
- `Th_in_K`
- `Tc_in_K`
- `uh_in_mps`
- `uc_in_mps`

### 11. If time remains, add transient data

After the stationary sweep, only if you still have time:

- choose `2-4` representative cases
- run time-dependent simulations
- export snapshots at consistent times

Suggested snapshot times:

- early
- mid
- late
- near steady state

### 12. Final check before you stop for the night

Before closing COMSOL, verify:

- every case folder exists
- every case has all expected CSVs
- the cold inlet field was exported explicitly
- hot/cold/wall domains are separated
- units are SI everywhere
- no file is empty

## Highest-Priority Deliverables

If you cannot finish everything tonight, prioritize in this order:

1. multiple stationary cases
2. volumetric hot and cold fields with `x,y,z,T,u,v,w,p`
3. wall temperature volume export
4. separate inlet, outlet, and interface boundary exports
5. global derived values

## What To Send Back

When the run is finished, bring back:

- the case folders
- the updated COMSOL model file
- any note about failed cases or convergence issues

That is enough to rebuild the PINN pipeline around the new data.
