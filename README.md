# GDPa4 bispecific developability figures

This repository contains the analysis required to reproduce the code-backed
figures and source data for [Decoding Bispecific Antibody Developability:
Design Rules and Predictive Models from a 160-Member Library](https://www.biorxiv.org/content/10.64898/2026.06.15.732449v1).
It describes 160 bispecific antibodies assembled from 65 unique monospecific
parents and profiled alongside 71 monospecific IgG1s.

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
Generate all supported figures and assemble the source-data workbook with:

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
  generator in this checkout.

Each script writes only its final-numbered image under `reports/figures/main/`
or `reports/figures/supplementary/`. Generated source-data files remain ignored.

## Data

The assay tables are in this repository. Open the files below; no separate request is required to review them.

- [Bispecific and monospecific assay measurements](<data/raw/GDPa4_N3_N4_Summary_tall.csv>), one row per replicate
- [GDPa1 monospecific reference workbook](data/raw/%5BExternal%5D%20AbDev%20peer-review%20246%20IgGs_Master%20data%20file_GDPa1.xlsx), including sequences, tidy assay data, and prior literature values
- [Bispecific production and purity](data/raw/production/Data_Summary_U594PPMRG0_03032026%20%281%29.xlsx)
- [Monospecific production](data/raw/production/N4_U126M421G0_AntibodyList_reformatted.xlsx)
- [In-silico parental features](data/raw/in_silico_gpa1/GDPa1/)
- [Library design tables](data/raw/bsab_design/)

`make data` reads these files and writes analysis tables under `data/processed/`. Those generated tables are not versioned. Four supervised-model records are committed so the default figure build does not refit models:

- `data/processed/05_modeling/cv_metrics.parquet`
- `data/processed/05_modeling/cv_metrics_loo.parquet`
- `data/processed/05_modeling/cv_oof_predictions_loo.parquet`
- `data/processed/05_modeling/feature_importance_long.parquet`

The same assay release is listed at [Ginkgo Datapoints](https://datapoints.ginkgo.bio/dataset-access).

## Verification

Rendering all figure scripts is the primary integration test. A minimal pytest
suite protects cohort counts, immutable inputs, and readability of the costly
model records:

```bash
make test
```

## Citation

Ritter S, Rand L, Karthick S, Bloomingdale T, Smith A, Ao X, Pierre Y,
Harris B, Moller J, Bhatt A, Bhatt R, Schwartz J, Grippo L, Cohen R,
Borhani DW, Tessier PM, Arsiwala A (2026). Decoding Bispecific Antibody
Developability: Design Rules and Predictive Models from a 160-Member
Library. *bioRxiv*. <https://www.biorxiv.org/content/10.64898/2026.06.15.732449v1>
