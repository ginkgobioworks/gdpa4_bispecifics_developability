"""Per-antibody replicate aggregation.

Centralizes the policy used everywhere downstream. Change here, not in
notebooks.
"""
from __future__ import annotations

import pandas as pd

# Decision D-2026-04-27-AGG (see decisions.md): use median across replicates
# as the per-antibody summary. Mean and std are also retained for context
# but downstream models should default to median.
SUMMARY_STATS: tuple[str, ...] = ("median", "mean", "std", "min", "max", "count")


def per_antibody_long(
    df: pd.DataFrame,
    *,
    name_col: str = "antibody_name",
    value_col_field: str = "value_col",
    condition_col: str = "condition",
    value_col: str = "value",
    extra_keys: tuple[str, ...] = ("kind", "is_control"),
) -> pd.DataFrame:
    """Collapse a tall measurement frame to per-(antibody × value_col × condition) summaries.

    Returns long format: one row per (antibody, value_col, condition) with one
    column per summary statistic in SUMMARY_STATS.
    """
    keys = [name_col, *extra_keys, value_col_field, condition_col]
    g = df.groupby(keys, dropna=False)[value_col]
    out = g.agg(list(SUMMARY_STATS)).reset_index()
    out = out.rename(columns={"count": "n_replicates"})
    return out
