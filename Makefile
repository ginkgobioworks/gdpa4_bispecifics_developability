PY := .venv/bin/python
MPL_ENV := MPLCONFIGDIR=.mplconfig

MAIN_FIGURES := 01 02 03 04 05 06
SUPP_FIGURES := 01 02 04 05 07 08 09 11 13 14 15 16

.PHONY: all data figures source-data test clean fit-local \
        $(addprefix figure-,$(MAIN_FIGURES)) \
        $(addprefix figure-s,$(SUPP_FIGURES))

all: data figures source-data

data:
	$(PY) -m prophet_ab.pipelines.build_01_normalized
	$(PY) -m prophet_ab.pipelines.build_02_joined
	$(PY) -m prophet_ab.pipelines.build_03_aggregated
	$(PY) -m prophet_ab.pipelines.build_04_features
	$(PY) -m prophet_ab.pipelines.build_05_wide
	$(MPL_ENV) $(PY) -m prophet_ab.pipelines.build_compositional_baselines

figures: $(addprefix figure-,$(MAIN_FIGURES)) $(addprefix figure-s,$(SUPP_FIGURES))

# Static pattern rules, not implicit pattern rules. GNU Make 3.81, the make
# shipped with Xcode, does not apply an implicit rule to a .PHONY target.
$(addprefix figure-,$(MAIN_FIGURES)): figure-%: data
	$(MPL_ENV) $(PY) figures/main/figure_$*.py

$(addprefix figure-s,$(SUPP_FIGURES)): figure-s%: data
	$(MPL_ENV) $(PY) figures/supplementary/figure_s$*.py

source-data:
	$(PY) reports/figures/figure_data_supplement/scripts/assemble_source_data.py

# Optional and expensive. The committed model records are sufficient for all
# manuscript figures; this target is retained only for methodological replay.
fit-local: data
	$(PY) -m prophet_ab.pipelines.build_05_cv

test: data
	$(PY) -m pytest tests -q

clean:
	rm -rf data/processed/01_normalized \
	       data/processed/02_joined \
	       data/processed/03_aggregated \
	       data/processed/04_features \
	       data/processed/05_modeling/bispecific_features_wide.parquet \
	       data/processed/05_modeling/bispecific_labels_wide.parquet \
	       reports/figures/main \
	       reports/figures/supplementary \
	       reports/figures/figure_data_supplement/extracted \
	       reports/figures/figure_data_supplement/source_data.xlsx \
	       reports/tables
