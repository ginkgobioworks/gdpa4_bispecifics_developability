# GDPa4 bispecific developability figures

This repository contains the analysis required to reproduce the code-backed
figures and source data for the revised PNAS manuscript. It describes 160
bispecific antibodies assembled from 65 unique monospecific parents and
profiled alongside 71 monospecific IgG1s.

The historical campaign labels remain only in immutable raw filenames and
input parsing. Processed data, code, and figure labels use **bispecific** and
**monospecific**.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[test]"
python scripts/ensure_editable_imports.py
```

## Reproduction

Build the normalized and analysis-ready data:

```bash
make data
```

Each code-backed manuscript figure has an independent, non-interactive Python
entry point. For example:

```bash
uv run python figures/main/figure_03.py
uv run python figures/supplementary/figure_s14.py
```

Each command writes a final-numbered PNG and its numerical source-data CSVs.
Generate all supported figures and assemble the PNAS source-data workbook with:

```bash
make all
```

The default workflow uses the committed supervised-model records in
`data/processed/05_modeling/`; it does not refit models. `make fit-local`
repeats that expensive sweep and can take hours.

## Final figure map

Main figures:

- Figure 1: `figures/main/figure_01.py` generates panels B and C; panels A and
  D are manually composed.
- Figure 2: `figures/main/figure_02.py`
- Figure 3: `figures/main/figure_03.py`
- Figure 4: `figures/main/figure_04.py`
- Figure 5: `figures/main/figure_05.py`
- Figure 6: `figures/main/figure_06.py`; panel A uses
  `reports/manual_input_materials/loo_diagram.png`.

Supplementary figures with in-repository generators:

- S1, S2, S4, S5, S7, S8, S9, S11, and S13–S16 are under
  `figures/supplementary/figure_sNN.py`.
- S3, S6, S10, and S12 are external or manually composed and have no analysis
  generator in this checkout. Their submitted records remain in
  `docs_for_updates/Supplementary_Information_PNAS_Revision_TRACKED.docx`.

Generated outputs are written under `reports/figures/main/`,
`reports/figures/supplementary/`, and
`reports/figures/figure_data_supplement/`; they are intentionally ignored by
Git.

## Preserved inputs

- `data/raw/`: immutable assay, design, production, and in-silico inputs.
- `data/processed/05_modeling/cv_metrics.parquet`
- `data/processed/05_modeling/cv_metrics_loo.parquet`
- `data/processed/05_modeling/cv_oof_predictions_loo.parquet`
- `data/processed/05_modeling/feature_importance_long.parquet`
- `datapoints_figures/`: the local manuscript plotting-style package.
- `docs_for_updates/`: the tracked manuscript and supplementary records.

Intermediate processed tables and rendered outputs are reproducible and are
not versioned.

## Verification

Rendering all figure scripts is the primary integration test. A minimal pytest
suite protects cohort counts, immutable inputs, and readability of the costly
model records:

```bash
make test
```

## Citation and license

Replace this section with the final article citation. Confirm data and code
redistribution terms before public release.
