# s01 — Data overview

Verifies the stage-01 normalized parquets and produces the dataset-overview
figure/table for the manuscript Methods section.

See `manifest.yaml` for inputs and outputs.

Notebooks (run in order):

1. `01_coverage.ipynb` — assay × kind coverage matrix, replicate counts.
2. `02_overlap.ipynb` — N3 ↔ N4 ↔ GDPa1 overlap and joinability.
