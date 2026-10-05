"""Parental-mean thermal residuals for supplementary figure S13.

Each produced bispecific is compared with the arithmetic mean of its two
monospecific parents on the same nanoDSF metric::

    delta = observed - mean(parent_a, parent_b)

Tm1, Tm2, and Tonset stay side by side. A metric is missing when the
bispecific or either parent has no median. Orientations are not collapsed:
the figure asks what CrossMab assembly cost the molecules that were made,
so the table has one row per bispecific (160), not per unique parental pair.
Tm2 is unresolved in some thermograms, so that column is shorter (125).

Several quartiles land on a half-hundredth (the ΔTm2 upper quartile is
−0.625 °C). ``round_half_away`` rounds half away from zero so that value
prints as −0.63; Python's default half-to-even rounding prints −0.62.

Example::

    from prophet_ab.features.thermal import load_crossmab_deltas, summarize_delta

    deltas = load_crossmab_deltas()
    tm2 = summarize_delta(deltas["delta_tm2"])
    tm2["n"]  # 125
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import paths, schema
from ..normalize import strip_isotype_suffix

# (short key, value_col, condition). Condition matches the PTS-IF row in the
# tall assay table, not a buffer.
THERMAL_METRICS: tuple[tuple[str, str, str], ...] = (
    ("tm1", "thermostability_tm1", "Tm1"),
    ("tm2", "thermostability_tm2", "Tm2"),
    ("tonset", "thermostability_tonset", "Tonset"),
)

# Left-tail cuts used by the figure, in °C.
TAIL_CUTS_C: tuple[float, ...] = (3.0, 5.0, 7.0)


def round_half_away(value: float, ndigits: int = 2) -> float:
    """Round half away from zero.

    Example::

        round_half_away(-0.625)  # -0.63, matching the S13 caption
    """
    factor = 10 ** ndigits
    scaled = abs(float(value)) * factor + 0.5
    rounded = math.floor(scaled) / factor
    if value < 0:
        return -rounded
    return rounded


def _observed_medians(
    summaries: pd.DataFrame, value_col: str, condition: str
) -> pd.Series:
    block = summaries[
        (summaries["value_col"] == value_col)
        & (summaries["condition"] == condition)
        & (summaries["kind"] == schema.KIND_BISPECIFIC)
    ]
    if block["antibody_name"].duplicated().any():
        raise ValueError(f"duplicate bispecific medians for {value_col}")
    return block.set_index("antibody_name")["median"]


def _parent_medians(
    summaries: pd.DataFrame, value_col: str, condition: str
) -> pd.Series:
    block = summaries[
        (summaries["value_col"] == value_col)
        & (summaries["condition"] == condition)
        & (summaries["kind"] == schema.KIND_MONOSPECIFIC)
    ]
    if block["antibody_name"].duplicated().any():
        raise ValueError(f"duplicate monospecific medians for {value_col}")
    parents = block["antibody_name"].map(strip_isotype_suffix)
    medians = pd.Series(block["median"].to_numpy(), index=parents, name="median")
    if medians.index.duplicated().any():
        dupes = sorted(set(medians.index[medians.index.duplicated(keep=False)]))
        raise ValueError(f"multiple monospecific medians for {value_col}: {dupes}")
    return medians


def load_crossmab_deltas(
    summaries: pd.DataFrame | None = None,
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per produced bispecific, with all three residuals.

    Reads ``bispecific_components.parquet`` and ``gdpa4_per_antibody.parquet``
    when frames are not passed in. Parental names are already suffix-stripped
    on the component table; monospecific names lose an ``_IgG1`` suffix here
    before the join.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    if components is None:
        components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")

    out = (
        components[["antibody_name", "parent_a", "parent_b"]]
        .drop_duplicates("antibody_name")
        .sort_values("antibody_name")
        .reset_index(drop=True)
    )
    for key, value_col, condition in THERMAL_METRICS:
        observed = _observed_medians(summaries, value_col, condition)
        parents = _parent_medians(summaries, value_col, condition)
        out[key] = out["antibody_name"].map(observed)
        out[f"{key}_a"] = out["parent_a"].map(parents)
        out[f"{key}_b"] = out["parent_b"].map(parents)
        out[f"{key}_parent_mean"] = (out[f"{key}_a"] + out[f"{key}_b"]) / 2.0
        out[f"delta_{key}"] = out[key] - out[f"{key}_parent_mean"]
    return out


def summarize_delta(values: pd.Series) -> dict[str, float]:
    """n, quartiles, range, and counts below −3, −5, and −7 °C.

    Missing residuals are dropped. Percentages are 100 × count / n.
    Quartiles use the linear (inclusive) definition, matching
    ``numpy.quantile(..., method="linear")``.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    array = clean.to_numpy(dtype=float)
    if array.size == 0:
        raise ValueError("no finite residuals to summarize")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    n = int(array.size)
    summary: dict[str, float] = {
        "n": float(n),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
    for cut in TAIL_CUTS_C:
        n_tail = int((array < -cut).sum())
        label = f"{cut:g}"
        summary[f"n_below_{label}"] = float(n_tail)
        summary[f"pct_below_{label}"] = 100.0 * n_tail / n
    return summary
