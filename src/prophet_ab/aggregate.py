"""Per-antibody replicate aggregation.

The per-antibody summary is the median across replicates. Mean and standard
deviation are retained; downstream models use the median.
"""
from __future__ import annotations

import pandas as pd

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
