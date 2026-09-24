PY := .venv/bin/python

LIB_SRC := src/prophet_ab/io.py \
           src/prophet_ab/normalize.py \
           src/prophet_ab/aggregate.py \
           src/prophet_ab/in_silico.py \
           src/prophet_ab/schema.py \
           src/prophet_ab/paths.py

S01_OUT := data/processed/01_normalized/n3n4_long.parquet \
           data/processed/01_normalized/gdpa1_tidy.parquet \
           data/processed/01_normalized/gdpa1_sequences.parquet \
           data/processed/01_normalized/gdpa1_prior_lit.parquet \
           data/processed/01_normalized/n3_production.parquet \
           data/processed/01_normalized/n4_production.parquet \
           data/processed/01_normalized/in_silico/Aggrescan3D.parquet \
           data/processed/01_normalized/in_silico/AntiFold.parquet \
           data/processed/01_normalized/in_silico/DeepViscosity.parquet \
           data/processed/01_normalized/in_silico/MOE.parquet \
           data/processed/01_normalized/in_silico/Saprot_VH.parquet \
           data/processed/01_normalized/in_silico/TAP.parquet

S02_OUT := data/processed/02_joined/n3_components.parquet \
           data/processed/02_joined/n4_gdpa1_map.parquet \
           data/processed/02_joined/n3_chains_long.parquet

S03_OUT := data/processed/03_aggregated/n3n4_per_antibody.parquet \
           data/processed/03_aggregated/gdpa1_per_antibody.parquet

S04_OUT := data/processed/04_features/in_silico_per_n4_long.parquet \
           data/processed/04_features/in_silico_per_n3_long.parquet

S05_WIDE_OUT := data/processed/05_modeling/n3_features_wide.parquet \
               data/processed/05_modeling/n3_labels_wide.parquet

S05_CV_CACHE := data/processed/05_modeling/cv_metrics.parquet \
                data/processed/05_modeling/cv_metrics_loo.parquet \
                data/processed/05_modeling/cv_oof_predictions_loo.parquet \
                data/processed/05_modeling/feature_importance_long.parquet

S00_NB_OUT := reports/figures/s00_umap_selected.png \
              reports/figures/s00_arm_distribution.png \
              reports/figures/s00_coverage_consolidated.png
S00_NB_SRC := notebooks/s00_bsab_design/00_design_coverage.py

S01_NB_OUT := reports/figures/s01_assay_coverage.png \
              reports/tables/s01_replicate_counts.csv
S01_NB_SRC := notebooks/s01_data_overview/01_coverage.py

S01_02_NB_OUT := reports/figures/s01_swap_pair_scatters_pearson.png \
                 reports/figures/s01_violin_pts_if_tm_pooled.png
S01_02_NB_SRC := notebooks/s01_data_overview/02_distributions.py

S02_NB_OUT := reports/figures/s02_cross_platform_scatter.png \
              reports/tables/s02_cross_platform_correlations.csv
S02_NB_SRC := notebooks/s02_cross_platform/01_reproducibility.py

S03_NB_OUT := reports/tables/s03_baseline_metrics.csv \
              data/processed/04_features/n3_compositional_predictions.parquet
S03_NB_SRC := notebooks/s03_compositional_baselines/01_baselines.py

S07_NB_OUT := reports/figures/s07_top_features_all_groups_reported_subset.png \
              reports/figures/s07_top_features_all_arm_groups_reported_subset.png
S07_NB_SRC := notebooks/s07_loo_models/01_metrics.py

S08_NB_OUT := reports/figures/s08_coefficient_overlay_reported_subset.png \
              reports/tables/s08_format_effect_coefficients.csv
S08_NB_SRC := notebooks/s08_format_effects/01_format_effects.py

S09_NB_OUT := reports/figures/s09_polyreactivity_vs_chromatography.png
S09_NB_SRC := notebooks/s09_multi_reactivity/01_reactivity_chromatography.py

S13_FIG2_OUT := reports/figures/main/figure_2_tiers.png
S13_FIG2_SRC := notebooks/s13_main_figures/01_figure2_tiers.py
S13_FIG2A_OUT := reports/figures/main/figure_2a_wide.png
S13_FIG2A_SRC := notebooks/s13_main_figures/05_figure2a_wide.py
S13_FIG3_OUT := reports/figures/main/figure_3.png \
                reports/figures/main/figure_3_version_2.png \
                reports/figures/main/figure_4.png
S13_FIG3_SRC := notebooks/s13_main_figures/02_figure3_acsins.py
S13_FIG5_OUT := reports/figures/main/figure_5_loo.png \
                reports/figures/main/figure_5_version_2.png \
                reports/figures/main/figure_5_version_3.png \
                reports/figures/main/figure_6_loo.png
S13_FIG5_SRC := notebooks/s13_main_figures/03_figure5_loo.py
S13_FIG5_DIAG := reports/manual_input_materials/loo_diagram.png
S13_SUPP_OUT := reports/figures/main/supplemental_figure_ceiling.png
S13_SUPP_SRC := notebooks/s13_main_figures/04_supplemental_ceiling.py

SOURCE_XLSX := reports/figures/figure_data_supplement/source_data.xlsx
EXTRACT_SCRIPTS := $(wildcard reports/figures/figure_data_supplement/scripts/extract_*.py)

.PHONY: all stages figures source-data \
        stage01 stage02 stage03 stage04 stage05 fit-local \
        s00 s01 s02 s03 s07 s08 s09 s13 test clean

all: stages figures source-data

stages: stage01 stage02 stage03 stage04 stage05

figures: s00 s01 s02 s03 s07 s08 s09 s13

stage01: $(S01_OUT)
$(S01_OUT): $(LIB_SRC) src/prophet_ab/pipelines/build_01_normalized.py
	$(PY) -m prophet_ab.pipelines.build_01_normalized

stage02: $(S02_OUT)
$(S02_OUT): $(LIB_SRC) src/prophet_ab/pipelines/build_02_joined.py $(S01_OUT)
	$(PY) -m prophet_ab.pipelines.build_02_joined

stage03: $(S03_OUT)
$(S03_OUT): $(LIB_SRC) src/prophet_ab/pipelines/build_03_aggregated.py $(S01_OUT)
	$(PY) -m prophet_ab.pipelines.build_03_aggregated

stage04: $(S04_OUT)
$(S04_OUT): $(LIB_SRC) src/prophet_ab/features/compositional.py src/prophet_ab/pipelines/build_04_features.py $(S01_OUT) $(S02_OUT)
	$(PY) -m prophet_ab.pipelines.build_04_features

stage05: $(S05_WIDE_OUT)
$(S05_WIDE_OUT): $(LIB_SRC) src/prophet_ab/features/compositional.py src/prophet_ab/features/wide.py src/prophet_ab/pipelines/build_05_wide.py $(S02_OUT) $(S03_OUT) $(S04_OUT)
	$(PY) -m prophet_ab.pipelines.build_05_wide

# Optional. Overwrites the committed LOO / k-fold metric parquets. Hours.
fit-local: | $(S05_WIDE_OUT)
	$(PY) -m prophet_ab.pipelines.build_05_cv

s00: $(S00_NB_OUT)
$(S00_NB_OUT): $(S00_NB_SRC) $(LIB_SRC)
	$(PY) $(S00_NB_SRC)

s01: $(S01_NB_OUT) $(S01_02_NB_OUT)
$(S01_NB_OUT): $(S01_NB_SRC) $(S01_OUT)
	$(PY) $(S01_NB_SRC)
$(S01_02_NB_OUT): $(S01_02_NB_SRC) $(LIB_SRC) $(S03_OUT)
	$(PY) $(S01_02_NB_SRC)

s02: $(S02_NB_OUT)
$(S02_NB_OUT): $(S02_NB_SRC) $(LIB_SRC) $(S02_OUT) $(S03_OUT)
	$(PY) $(S02_NB_SRC)

s03: $(S03_NB_OUT)
$(S03_NB_OUT): $(S03_NB_SRC) $(LIB_SRC) src/prophet_ab/features/compositional.py $(S02_OUT) $(S03_OUT)
	$(PY) $(S03_NB_SRC)

s07: $(S07_NB_OUT)
$(S07_NB_OUT): $(S07_NB_SRC) $(LIB_SRC) | $(S05_CV_CACHE)
	$(PY) $(S07_NB_SRC)

s08: $(S08_NB_OUT)
$(S08_NB_OUT): $(S08_NB_SRC) $(LIB_SRC) src/prophet_ab/features/format_effects.py $(S02_OUT) $(S03_OUT)
	$(PY) $(S08_NB_SRC)

s09: $(S09_NB_OUT)
$(S09_NB_OUT): $(S09_NB_SRC) $(LIB_SRC) $(S03_OUT)
	$(PY) $(S09_NB_SRC)

s13: $(S13_FIG2_OUT) $(S13_FIG2A_OUT) $(S13_FIG3_OUT) $(S13_FIG5_OUT) $(S13_SUPP_OUT)

$(S13_FIG2_OUT): $(S13_FIG2_SRC) $(LIB_SRC) src/prophet_ab/features/transforms.py $(S02_OUT) $(S03_OUT)
	$(PY) $(S13_FIG2_SRC)

$(S13_FIG2A_OUT): $(S13_FIG2A_SRC) $(LIB_SRC) src/prophet_ab/features/transforms.py $(S02_OUT) $(S03_OUT)
	$(PY) $(S13_FIG2A_SRC)

$(S13_FIG3_OUT): $(S13_FIG3_SRC) $(LIB_SRC) $(S02_OUT) $(S03_OUT)
	$(PY) $(S13_FIG3_SRC)

$(S13_FIG5_OUT): $(S13_FIG5_SRC) $(LIB_SRC) src/prophet_ab/features/naming.py $(S13_FIG5_DIAG) $(S03_NB_OUT) | $(S05_CV_CACHE)
	$(PY) $(S13_FIG5_SRC)

$(S13_SUPP_OUT): $(S13_SUPP_SRC) $(LIB_SRC) $(S02_OUT) $(S03_OUT)
	$(PY) $(S13_SUPP_SRC)

source-data: $(SOURCE_XLSX)
$(SOURCE_XLSX): $(EXTRACT_SCRIPTS) reports/figures/figure_data_supplement/scripts/assemble_source_data.py \
		$(S00_NB_OUT) $(S01_02_NB_OUT) $(S02_NB_OUT) $(S03_NB_OUT) $(S07_NB_OUT) $(S08_NB_OUT) $(S09_NB_OUT) $(S13_FIG2_OUT) $(S13_FIG3_OUT) $(S13_FIG5_OUT) $(S13_SUPP_OUT)
	@mkdir -p reports/figures/figure_data_supplement/extracted
	@for s in $(EXTRACT_SCRIPTS); do echo "extract: $$s"; $(PY) $$s; done
	$(PY) reports/figures/figure_data_supplement/scripts/assemble_source_data.py

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/processed/01_normalized data/processed/02_joined \
	       data/processed/03_aggregated data/processed/04_features \
	       data/processed/05_modeling/n3_features_wide.parquet \
	       data/processed/05_modeling/n3_labels_wide.parquet \
	       reports/figures/*.png reports/figures/main \
	       reports/tables \
	       reports/figures/figure_data_supplement/extracted \
	       reports/figures/figure_data_supplement/source_data.xlsx
