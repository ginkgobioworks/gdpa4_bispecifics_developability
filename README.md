# GDPa4 bispecific developability figures

This repository rebuilds the central figures for the GDPa4 campaign: **160
bispecific antibodies (N3)** assembled from **65 unique monospecific parents**,
profiled together with **71 monospecific IgG1s (N4)** on the PROPHET-Ab
platform. A subset of the N4 panel also appears in the published GDPa1
benchmark (Arsiwala et al. 2025, *MAbs*).

N3/N4 are format labels used in the tables and code. The assay table for this
campaign is `data/raw/GDPa4_N3_N4_Summary_tall.csv`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[analysis]"
```

For the numeric tests:

```bash
uv pip install -e ".[test]"
```

## Reproduce figures (default)

Default `make all` rebuilds processed assay tables and every listed figure
from the committed model caches. It does **not** re-fit models. Then run
`python -m pytest tests -q` (or `make test`).

```bash
make all
```

If `make` is not available, the same steps are `bash scripts/reproduce.sh`.

Runtime is typically 10–20 minutes on a laptop (s00 protein-property
computation and s07 figure rendering dominate). No GPU is required.

Then:

```bash
make test
```

### Optional model re-fit

The leave-one-bispecific-out sweep (Ridge, Lasso, ElasticNet, random forest,
XGBoost, HGBM, PLS × feature configs × labels) is already stored under
`data/processed/05_modeling/`. To overwrite those parquets locally:

```bash
make fit-local
```

This takes hours on a multi-core workstation.

## Inputs

| File | Role |
|---|---|
| `data/raw/GDPa4_N3_N4_Summary_tall.csv` | Primary GDPa4 assay table (tall) |
| `data/raw/[External] AbDev peer-review 246 IgGs_Master data file_GDPa1.xlsx` | GDPa1 companion file |
| `data/raw/bsab_design/bsabs_dec_2025/data/` | Design-space selection tables |
| `data/raw/bsab_design/umap_coords.csv` | Frozen UMAP coordinates for Figure 1B |
| `data/raw/in_silico_gpa1/GDPa1/` | In-silico predictor CSVs |
| `data/raw/production/` | N3 and N4 production summaries |
| `data/processed/05_modeling/cv_*.parquet` | Committed CV metrics / importance |

## Figure outputs

File names inside `reports/figures/` are the names used by the plotting
scripts. Manuscript numbering in
`reports/figures/figure_data_supplement/figures_for_data.md` can differ.

| Description | Path |
|---|---|
| Design-space UMAP (selected pairs) | `reports/figures/s00_umap_selected.png` |
| Parent-arm usage | `reports/figures/s00_arm_distribution.png` |
| Design-space coverage grid | `reports/figures/s00_coverage_consolidated.png` |
| Swap-pair orientation scatters | `reports/figures/s01_swap_pair_scatters_pearson.png` |
| Pooled Tm violins | `reports/figures/s01_violin_pts_if_tm_pooled.png` |
| GDPa4 vs GDPa1 scatter | `reports/figures/s02_cross_platform_scatter.png` |
| Compositional tiers | `reports/figures/main/figure_2_tiers.png` |
| Compositional tiers (wide panel A) | `reports/figures/main/figure_2a_wide.png` |
| AC-SINS classification + charge | `reports/figures/main/figure_3.png` |
| AC-SINS charge quadrants (band-only alt.) | `reports/figures/main/figure_3_version_2.png` |
| Charge-transform horserace | `reports/figures/main/figure_4.png` |
| Polyreactivity vs HIC/HAC | `reports/figures/s09_polyreactivity_vs_chromatography.png` |
| LOO design + model sweep | `reports/figures/main/figure_5_loo.png` |
| LOO vs parental bars | `reports/figures/main/figure_5_version_2.png` |
| LOO Δρ bars | `reports/figures/main/figure_5_version_3.png` |
| Feature-group importance + dumbbell | `reports/figures/main/figure_6_loo.png` |
| mAb ceiling crossers | `reports/figures/main/supplemental_figure_ceiling.png` |
| LOO top features (operator configs) | `reports/figures/s07_top_features_all_groups_reported_subset.png` |
| LOO top features (arm configs) | `reports/figures/s07_top_features_all_arm_groups_reported_subset.png` |
| Format-effect coefficients | `reports/figures/s08_coefficient_overlay_reported_subset.png` |

Source-data Excel workbook (one sheet per panel):

`reports/figures/figure_data_supplement/source_data.xlsx`

Rebuild a single section with `make s00`, `make s01`, …, `make s13`.

## How the pipeline is organized

```
data/raw/GDPa4_N3_N4_Summary_tall.csv
        → python -m prophet_ab.pipelines.build_01_normalized
        → build_02_joined → build_03_aggregated
        → build_04_features → build_05_wide
        → notebooks (figures)
```

Median over replicates is the per-antibody summary. Three N4 names carry an
`_IgG1` suffix (`bococizumab_IgG1`, `galcanezumab_IgG1`, `ixekizumab_IgG1`);
that suffix is stripped before joining N3 components to N4 parents.
Deprecated metrics excluded from modeling: `acsins_Lmax`, `pr_score_norm`.

Open a notebook in the browser with `marimo edit notebooks/sNN_.../*.py`,
or run headless with `python notebooks/sNN_.../*.py`.

## Citation

Replace with the published article citation when available.

## License

Data and code licensing for redistribution should be confirmed before a
public release.
