# Heat Exchanger Studio

## What This Workspace Does

- Browse preset operating points from the reference library.
- Generate fresh surface previews for a selected operating point.
- Create a slice view and review temperature balance when needed.
- Launch a new run without editing scripts directly.

## Start Here

- Open the scenario library to compare saved operating points.
- Use the live preview page to test a new condition and inspect the wall maps.
- Use the build page when you want a fresh run from the same workspace.
- Use the flow view page to export a slice view or review the temperature balance.

## Visual Assets

- Dataset operating matrix: `C:\Users\TL\Desktop\PINN\reports\figures\dataset_operating_matrix.png`
- Dataset boundary heatmaps: `C:\Users\TL\Desktop\PINN\reports\figures\dataset_boundary_heatmaps.png`

## Launch

- Launch with `python app/final_boundary_gui.py`.
- Or run `streamlit run app/boundary_studio.py` directly.

## Notes

- The slice view is intended for exploration and review.
- Additional solver-backed interior fields can be connected later without changing the workspace flow.
