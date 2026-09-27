"""Canonical names, enums, and cross-dataset mappings.

Legacy campaign labels are accepted only at the raw-data boundary. Processed
tables use biological format names so publication-facing analysis does not
depend on internal production campaign terminology.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Sample kinds
# ---------------------------------------------------------------------------
KIND_BISPECIFIC = "bispecific"
KIND_MONOSPECIFIC = "monospecific"
KIND_GDPA1 = "GDPa1"  # historical monospecific

# ---------------------------------------------------------------------------
# PROPHET-Ab assay panel (this campaign)
# ---------------------------------------------------------------------------
# (assay, condition, value_col) tuples actually present in the tall CSV.
ASSAY_PANEL: tuple[tuple[str, str, str], ...] = (
    ("AC-SINS",  "1X PBS",          "acsins_Lmax"),
    ("AC-SINS",  "1X PBS",          "acsins_delta_Lmax"),
    ("AC-SINS",  "His/Arg, pH 6",   "acsins_Lmax"),
    ("AC-SINS",  "His/Arg, pH 6",   "acsins_delta_Lmax"),
    ("AC-SINS",  "His/NaCl, pH 6",  "acsins_Lmax"),
    ("AC-SINS",  "His/NaCl, pH 6",  "acsins_delta_Lmax"),
    ("BVP",      "default",         "bvp_score_norm"),
    ("HPLC-HAC", "default",         "hachplc_retentiontime"),
    ("HPLC-HIC", "default",         "hihplc_normretentiontime"),
    ("HPLC-SEC", "default",         "sehplc_pct_mono"),
    ("HPLC-SMAC","default",         "smachplc_retentiontime"),
    ("IntactMS", "default",         "purity_pct"),
    ("PR",       "CHO",             "pr_score"),
    ("PR",       "CHO",             "pr_score_norm"),
    ("PR",       "Ovalbumin",       "pr_score"),
    ("PR",       "Ovalbumin",       "pr_score_norm"),
    ("PTS-IF",   "Tm1",             "thermostability_tm1"),
    ("PTS-IF",   "Tm2",             "thermostability_tm2"),
    ("PTS-IF",   "Tonset",          "thermostability_tonset"),
)

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
# All 71 monospecific monospecifics were expressed on a uniform IgG1 constant region
# (D-2026-04-27-IGG1-UNIFORM). Three monospecific names carry an explicit `_IgG1`
# suffix flagging the originally non-IgG1 parents; their bispecific components drop
# the suffix, so strip at join time (D-2026-04-27-ISOTYPE).
ISOTYPE_SUFFIXES_TO_STRIP: tuple[str, ...] = ("_IgG1",)

# Plate controls. These also occur as legitimate monospecific parents in some bispecifics; do
# not drop them by name — instead flag.
CONTROL_NAMES: frozenset[str] = frozenset({"atezolizumab", "trastuzumab", "adalimumab"})

LEGACY_BISPECIFIC_PREFIX = "N3-"
BISPECIFIC_PAIR_SEP = "__x__"

# ---------------------------------------------------------------------------
# Metrics deprecated for downstream analysis
# ---------------------------------------------------------------------------
# value_cols here are kept in raw + per-antibody parquets (so the measurement
# record is preserved) but are excluded from feature builders, label builders,
# and any analysis-layer reporting. Maintained as a single source of truth so
# that "switch the canonical metric" is a one-line edit.
#
# Per D-2026-04-27-PR-PRIMARY: pr_score is the canonical PR metric;
# pr_score_norm is excluded from baselines, features, and labels.
DEPRECATED_VALUE_COLS: frozenset[str] = frozenset({"pr_score_norm", "acsins_Lmax"})

# Production QC metrics — flow through gdpa4_long and aggregation for
# descriptive analysis but are excluded from feature/label builders.
PRODUCTION_VALUE_COLS: frozenset[str] = frozenset({
    "production_amount_mg",
    "production_concentration_mg_ml",
    "production_endotoxin_eu_mg",
    "production_purity_sdspage_pct",
    "production_purity_sechplc_pct",
})

EXCLUDED_FROM_MODELING: frozenset[str] = DEPRECATED_VALUE_COLS | PRODUCTION_VALUE_COLS

# ---------------------------------------------------------------------------
# Reported subset for key figures
# ---------------------------------------------------------------------------
# Display-only filter: figures tagged as "reported" show only these value_cols.
# Modeling and full-panel analysis are unaffected.
REPORTED_VALUE_COLS: frozenset[str] = frozenset({
    "hihplc_normretentiontime",
    "smachplc_retentiontime",
    "hachplc_retentiontime",
    "acsins_delta_Lmax",
    "pr_score",
    "bvp_score_norm",
    "thermostability_tm1",
    "thermostability_tm2",
})

# ---------------------------------------------------------------------------
# bispecific/monospecific  ↔  GDPa1 cross-platform mapping
# ---------------------------------------------------------------------------
# (this_value_col, this_condition_or_None) -> gdpa1_column
# `this_condition` of None means "any condition / not condition-keyed".
PROPHET_TO_GDPA1: dict[tuple[str, str | None], str] = {
    ("acsins_delta_Lmax",                    "1X PBS"):         "acsins_dLmax_ph7.4",
    ("acsins_delta_Lmax",                    "His/Arg, pH 6"):  "acsins_dLmax_ph6.0",
    ("hihplc_normretentiontime",             "default"):        "hic_rt",
    ("smachplc_retentiontime",               "default"):        "smac_rt",
    ("hachplc_retentiontime",                "default"):        "hac_rt",
    ("pr_score",                             "CHO"):            "polyreactivity_prscore_cho",
    ("pr_score",                             "Ovalbumin"):      "polyreactivity_prscore_ova",
}
# Decision D-2026-04-27-AC-SINS: His/Arg only is the like-for-like comparator
# to GDPa1's pH 6.0 condition; His/NaCl is intentionally not mapped.
# Decision D-2026-04-27-PR-V2: pr_score (not pr_score_norm) maps to GDPa1
# (supersedes the original D-2026-04-27-PR).
# Decision D-2026-05-08-DROP-SEC-THERMO: SEC %monomer and thermostability
# (Tm1, Tm2, Tonset) removed from cross-platform comparison.
# Decision D-2026-05-08-GDPA1-IGG1: filter GDPa1 to IgG1 (hc_subtype).

# ---------------------------------------------------------------------------
# Display names for figures
# ---------------------------------------------------------------------------
VALUE_COL_DISPLAY: dict[str, str] = {
    "acsins_Lmax":                     "AC-SINS Lmax",
    "acsins_delta_Lmax":               "AC-SINS ΔLmax",
    "bvp_score_norm":                  "BVP Score",
    "hachplc_retentiontime":           "HAC RT",
    "hihplc_normretentiontime":        "HIC RT (norm)",
    "sehplc_pct_mono":                 "SEC % Monomer",
    "smachplc_retentiontime":          "SMAC RT",
    "purity_pct":                      "Purity (%)",
    "pr_score":                        "PR Score",
    "pr_score_norm":                   "PR Score (norm)",
    "thermostability_tm1":             "Tm1",
    "thermostability_tm2":             "Tm2",
    "thermostability_tonset":          "Tonset",
    "production_amount_mg":            "Amount (mg)",
    "production_concentration_mg_ml":  "Concentration (mg/mL)",
    "production_endotoxin_eu_mg":      "Endotoxin (EU/mg)",
    "production_purity_sdspage_pct":   "SDS-PAGE Purity (%)",
    "production_purity_sechplc_pct":   "SEC-HPLC Purity (%)",
}

CONDITION_DISPLAY: dict[str, str] = {
    "default":         "",
    "1X PBS":          "PBS",
    "His/Arg, pH 6":   "His/Arg pH 6",
    "His/NaCl, pH 6":  "His/NaCl pH 6",
    "CHO":             "CHO",
    "Ovalbumin":       "Ovalbumin",
    "Tm1":             "Tm1",
    "Tm2":             "Tm2",
    "Tonset":          "Tonset",
}

CONFIG_DISPLAY: dict[str, str] = {
    "compositional_baseline":          "Parent Mean",
    "corresponding_experimental":      "Same Assay",
    "all_experimental":                "All Assays",
    "in_silico_only":                  "In Silico",
    "in_silico_plus_corresponding":    "In Silico + Same Assay",
    "in_silico_plus_all_experimental": "In Silico + All Assays",
    "arm_corresponding":                    "Arm Same Assay",
    "arm_all_experimental":                 "Arm All Assays",
    "arm_in_silico_only":                   "Arm In Silico",
    "arm_in_silico_plus_corresponding":     "Arm In Silico + Same Assay",
    "arm_in_silico_plus_all_experimental":  "Arm In Silico + All Assays",
}

OPERATOR_DISPLAY: dict[str, str] = {
    "mean":     "Mean",
    "min":      "Min",
    "max":      "Max",
    "geomean":  "GeoMean",
    "abs_diff": "|Diff|",
    "a": "Arm A",
    "b": "Arm B",
}
