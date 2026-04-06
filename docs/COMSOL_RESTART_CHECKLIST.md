# COMSOL Restart Checklist For The 3D PINN

## Goal

The current 3D PINN is now limited mainly by missing CFD supervision, not by the training code.

The next COMSOL run should produce:

1. volumetric fields, not just surface temperature
2. separate exports for hot fluid, cold fluid, and wall domains
3. multiple operating points, not one single case
4. boundary and global-value exports for validation

## Minimum Study Set

Run a **stationary parametric sweep** first.

Minimum useful set: `12` cases  
Recommended set: `16-20` cases

Use the operating-point template in [comsol_operating_matrix_template.csv](/c:/Users/TL/Desktop/PINN/docs/comsol_operating_matrix_template.csv).

If you have time after the stationary sweep, add a **time-dependent study** for `2-4` representative cases.

## Parameters To Sweep

Sweep these four inputs:

- hot inlet temperature `T_h,in`
- cold inlet temperature `T_c,in`
- hot inlet velocity `u_h,in`
- cold inlet velocity `u_c,in`

Recommended ranges:

- `T_h,in`: `295, 303, 311 K`
- `T_c,in`: `279, 283.5, 287 K`
- `u_h,in`: `0.6, 1.0, 1.4 m/s`
- `u_c,in`: `0.6, 1.0, 1.4 m/s`

If your true operating range is different, use your real range instead. Keep the center case close to the current baseline.

## Domains To Export

Export each of these separately:

1. hot-fluid volume
2. cold-fluid volume
3. wall volume

Do not merge all domains into one file unless you also export a reliable domain identifier.

## Volume Fields To Export

For **hot-fluid volume** and **cold-fluid volume**, export:

- `x, y, z`
- temperature `T`
- velocity components `u, v, w`
- pressure `p`
- density if it varies with temperature
- dynamic viscosity if it varies with temperature
- effective thermal conductivity or thermal diffusivity if available
- turbulent viscosity or turbulence-derived transport quantities if available

For **wall volume**, export:

- `x, y, z`
- wall temperature `T`
- wall heat-flux components if available
- wall material properties if temperature-dependent

Use SI units only.

## Boundary Exports

Export each boundary set separately:

1. hot inlet
2. hot outlet
3. cold inlet
4. cold outlet
5. hot-fluid to wall interface
6. wall to cold-fluid interface

For each boundary export:

- `x, y, z`
- temperature `T`
- velocity components on fluid boundaries
- pressure on fluid boundaries
- outward normal vector if available
- normal heat flux if available

Important:

- Do **not** assume the report cold inlet is enough.
- Export the **actual cold-inlet boundary field** from COMSOL.
- The current PINN should trust the field data for the cold side when the field and report disagree.

## Global / Derived Values To Export

For every case, save these scalar values:

- hot inlet bulk temperature
- hot outlet bulk temperature
- cold inlet bulk temperature
- cold outlet bulk temperature
- hot mass flow rate
- cold mass flow rate
- hot-side pressure drop
- cold-side pressure drop
- total heat duty
- exchanger effectiveness
- area-averaged wall temperature if available

These are needed for sanity checks, loss design, and engineering evaluation.

## Mesh / Sampling Guidance

Use the same geometry and, if possible, the same mesh strategy across all cases.

For export resolution:

- keep enough points in the full volume to resolve radial gradients
- keep enough points near inlets, outlets, and interfaces
- do not downsample away the boundary layers

If export size becomes too large, prefer:

1. full boundary exports
2. dense near-interface volume exports
3. moderate interior volume exports

Do not fall back to surface temperature only.

## File Organization

Use one folder per case:

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

If COMSOL makes separate exports difficult, keep the same naming pattern as closely as possible.

## Case Naming

Use stable case IDs:

- `case_001`
- `case_002`
- ...

In `globals.csv`, include the actual parameter values:

- `Th_in_K`
- `Tc_in_K`
- `uh_in_mps`
- `uc_in_mps`

## If You Can Add Only One Extra Thing

If tonight you can add only one improvement, make it this:

**export volumetric `u, v, w, p, T` for hot and cold domains plus wall temperature for multiple operating cases**

That would help the PINN more than any other single change.

## If You Can Add Two Extra Things

Add:

1. volumetric fields
2. multiple operating points

This is the minimum set that turns the current 3D PINN into a much stronger digital twin candidate.

## Optional But High Value

If you have time:

- add `2-4` transient cases with saved time snapshots
- export wall heat flux directly
- export turbulence transport quantities
- export pressure-drop and heat-duty derived values for every case

## What To Avoid

- only exporting one stationary case again
- only exporting temperature
- only exporting surfaces
- mixing hot, cold, and wall points without identifiers
- changing units across files
- using inconsistent file names across cases
