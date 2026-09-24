#!/usr/bin/env bash
# Equivalent to `make all` without requiring GNU/BSD make.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"

"$PY" -m prophet_ab.pipelines.build_01_normalized
"$PY" -m prophet_ab.pipelines.build_02_joined
"$PY" -m prophet_ab.pipelines.build_03_aggregated
"$PY" -m prophet_ab.pipelines.build_04_features
"$PY" -m prophet_ab.pipelines.build_05_wide

"$PY" notebooks/s00_bsab_design/00_design_coverage.py
"$PY" notebooks/s01_data_overview/01_coverage.py
"$PY" notebooks/s01_data_overview/02_distributions.py
"$PY" notebooks/s02_cross_platform/01_reproducibility.py
"$PY" notebooks/s03_compositional_baselines/01_baselines.py
"$PY" notebooks/s07_loo_models/01_metrics.py
"$PY" notebooks/s08_format_effects/01_format_effects.py
"$PY" notebooks/s09_multi_reactivity/01_reactivity_chromatography.py
"$PY" notebooks/s13_main_figures/01_figure2_tiers.py
"$PY" notebooks/s13_main_figures/05_figure2a_wide.py
"$PY" notebooks/s13_main_figures/02_figure3_acsins.py
"$PY" notebooks/s13_main_figures/04_supplemental_ceiling.py
"$PY" notebooks/s13_main_figures/03_figure5_loo.py

mkdir -p reports/figures/figure_data_supplement/extracted
for s in reports/figures/figure_data_supplement/scripts/extract_*.py; do
  echo "extract: $s"
  "$PY" "$s"
done
"$PY" reports/figures/figure_data_supplement/scripts/assemble_source_data.py
echo "done."
