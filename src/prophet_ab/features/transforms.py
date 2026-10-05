"""Arm-combination transforms and the three-tier inheritance metadata.

- four operators that predict a bispecific value from its two parental medians
- the operator used for each metric
- the short display label per metric
- the three inheritance tiers, with colors and captions

Metrics are keyed by ``(value_col, condition)`` as in
``data/processed/03_aggregated/gdpa4_per_antibody.parquet``.
"""
from __future__ import annotations

from datapoints_figures import DATAPOINTS_COLORS

# --- Arm-combination operators -------------------------------------------
# Each operator maps two aligned pandas Series (parent A, parent B medians)
# to a predicted bispecific value.
TRANSFORMS = {
    "mean": lambda a, b: (a + b) / 2,
    "min": lambda a, b: a.combine(b, min),
    "max": lambda a, b: a.combine(b, max),
    "geomean": lambda a, b: a.combine(b, lambda x, y: (
        float("nan") if x <= 0 or y <= 0 else (x * y) ** 0.5
    )),
}

# --- Operator used for each metric ---------------------------------------
BEST_TRANSFORM = {
    ("pr_score", "Ovalbumin"):                          "mean",
    ("pr_score", "CHO"):                                "mean",
    ("bvp_score_norm", "default"):                      "mean",
    ("hihplc_normretentiontime", "default"):            "mean",
    ("smachplc_retentiontime", "default"):              "mean",
    ("hachplc_retentiontime", "default"):               "max",
    ("thermostability_tm1", "Tm1"):                     "min",
    ("thermostability_tm2", "Tm2"):                     "min",
    ("thermostability_tonset", "Tonset"):               "min",
    ("acsins_delta_Lmax", "1X PBS"):                    "mean",
    ("acsins_delta_Lmax", "His/Arg, pH 6"):             "mean",
    ("acsins_delta_Lmax", "His/NaCl, pH 6"):            "mean",
    ("sehplc_pct_mono", "default"):                     "max",
}

# --- Short display label per metric --------------------------------------
SHORT_LABEL = {
    ("pr_score", "Ovalbumin"):                          "PR-OVA",
    ("pr_score", "CHO"):                                "PR-CHO",
    ("bvp_score_norm", "default"):                      "PR-BVP",
    ("hihplc_normretentiontime", "default"):            "HIC",
    ("smachplc_retentiontime", "default"):              "SMAC",
    ("hachplc_retentiontime", "default"):               "HAC",
    ("thermostability_tm1", "Tm1"):                     "Tm1",
    ("thermostability_tm2", "Tm2"):                     "Tm2",
    ("thermostability_tonset", "Tonset"):               "Tonset",
    ("acsins_delta_Lmax", "1X PBS"):                    "AC-SINS PBS pH 7.4",
    ("acsins_delta_Lmax", "His/Arg, pH 6"):             "AC-SINS His/Arg pH 6.0",
    ("acsins_delta_Lmax", "His/NaCl, pH 6"):            "AC-SINS His/NaCl pH 6.0",
    ("sehplc_pct_mono", "default"):                     "SEC %mono",
}

PRIMARY_METRICS = list(BEST_TRANSFORM.keys())

# --- Three-tier inheritance grouping (s13 combined figure) ---------------
# Tier 1 (inheritance): surface chromatography, ρ >= 0.8.
# Tier 2 (emergence): self-association + polyreactivity, 0.4 <= ρ < 0.8.
# Tier 3 (non-inheritance): thermostability + SEC, ρ < 0.4.
# Each tier is one row of three scatter panels in the combined figure; the
# two Tier-2 row-groups share tier number 2.
TIER_DEFS = [
    (1, [
        ("hihplc_normretentiontime", "default"),
        ("smachplc_retentiontime", "default"),
        ("hachplc_retentiontime", "default"),
    ]),
    (2, [
        ("acsins_delta_Lmax", "1X PBS"),
        ("acsins_delta_Lmax", "His/NaCl, pH 6"),
        ("acsins_delta_Lmax", "His/Arg, pH 6"),
    ]),
    (2, [
        ("pr_score", "CHO"),
        ("pr_score", "Ovalbumin"),
        ("bvp_score_norm", "default"),
    ]),
    (3, [
        ("thermostability_tm1", "Tm1"),
        ("thermostability_tm2", "Tm2"),
        ("sehplc_pct_mono", "default"),
    ]),
]

# Semantic tier colors from the house palette.
TIER_COLORS = {
    1: DATAPOINTS_COLORS["green"],
    2: DATAPOINTS_COLORS["amber"],
    3: DATAPOINTS_COLORS["red"],
}

# Tier captions for the figure's side annotations: (title, tagline).
TIER_LABELS = {
    1: ("Class I: Inherited Properties", "highly additive (ρ ≥ 0.8)"),
    2: ("Class II: Context-dependent Properties", "moderate (0.4 ≤ ρ < 0.8)"),
    3: ("Class III: Format-driven Properties", "non-additive (ρ < 0.4)"),
}
