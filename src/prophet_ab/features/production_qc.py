"""Parental-mean production and purity tiers for supplementary figure S15.

Reviewer 2 asked whether the three-tier inheritance framework also covers
expression titer and purity. For each attribute the parental expectation is
the arithmetic mean of the two monospecific medians. A bispecific is kept
only when both arms have a median, which is why SEC percent monomer has
158 molecules and the three production attributes have 160.

The tier rule is the one quoted for this figure::

    if rho >= 0.7 and abs(median residual) < sigma_arm:  Class I
    if rho < 0.3:                                        Class III
    otherwise:                                           Class II

``sigma_arm`` is the sample standard deviation (ddof = 1) of every
monospecific median for that attribute, including parents that were not
used in a bispecific. Residual is observed minus parental mean. Class III
does not depend on the sign of that residual.

On the repository medians every attribute is Class III. Titer and SDS-PAGE
purity sit below the parental mean; SEC-HPLC purity and SEC percent monomer
sit slightly above it. Panel D's Spearman ρ is 0.141. The revision note
prints 0.140; both are far below the 0.3 cutoff, so the class is the same.

Example::

    from prophet_ab.features.production_qc import (
        load_production_pairs,
        tier_assignments,
    )

    pairs = load_production_pairs()
    tiers = tier_assignments(pairs)
    tiers.loc[tiers["key"] == "titer", "tier"].item()  # "III"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import paths, schema
from ..normalize import strip_isotype_suffix

# (panel letter, short key, display name, value_col, condition).
# Conditions match the tall assay table, not the workbook's flattened headers.
PRODUCTION_ATTRIBUTES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "A",
        "titer",
        "Expression titer (mg/mL)",
        "production_concentration_mg_ml",
        "Concentration",
    ),
    (
        "B",
        "sds_page",
        "SDS-PAGE purity (%)",
        "production_purity_sdspage_pct",
        "SDS-PAGE Purity NR",
    ),
    (
        "C",
        "sec_hplc",
        "SEC-HPLC purity (%)",
        "production_purity_sechplc_pct",
        "SEC-HPLC Purity",
    ),
    (
        "D",
        "sec_mono",
        "SEC % monomer",
        "sehplc_pct_mono",
        "default",
    ),
)

# Class I needs both a strong rank correlation and a typical offset smaller
# than the spread of the monospecifics. Class III is rank correlation alone.
RHO_INHERITED = 0.7
RHO_FORMAT_DRIVEN = 0.3

PAIR_COLUMNS: tuple[str, ...] = (
    "antibody_name",
    "parent_a",
    "parent_b",
    "panel",
    "key",
    "attribute",
    "value_col",
    "condition",
    "arm_a_value",
    "arm_b_value",
    "parental_mean",
    "observed",
    "residual",
)

ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "panel",
    "key",
    "attribute",
    "value_col",
    "condition",
    "n",
    "n_mab",
    "spearman_rho",
    "pearson_r",
    "median_residual",
    "sigma_residual",
    "sigma_arm",
    "tier",
)


def assign_tier(rho: float, median_residual: float, sigma_arm: float) -> str:
    """Return ``"I"``, ``"II"``, or ``"III"`` for one attribute.

    Class I is inherited: Spearman ρ at least 0.7 and a median residual
    whose absolute value is strictly smaller than the monospecific standard
    deviation. Class III is format-driven: ρ below 0.3, whatever the sign
    or the size of the residual. Everything else is Class II.

    Example::

        assign_tier(0.90, 0.1, 0.5)   # "I"
        assign_tier(0.90, 0.5, 0.5)   # "II"  offset is not smaller than σ arm
        assign_tier(0.024, -3.02, 0.67)  # "III"
    """
    rho = float(rho)
    median_residual = float(median_residual)
    sigma_arm = float(sigma_arm)
    if not (
        np.isfinite(rho)
        and np.isfinite(median_residual)
        and np.isfinite(sigma_arm)
    ):
        raise ValueError("rho, median residual, and sigma_arm must be finite")
    if sigma_arm < 0:
        raise ValueError("sigma_arm must be non-negative")
    if rho >= RHO_INHERITED and abs(median_residual) < sigma_arm:
        return "I"
    if rho < RHO_FORMAT_DRIVEN:
        return "III"
    return "II"


def _medians(
    summaries: pd.DataFrame, value_col: str, condition: str, kind: str
) -> pd.Series:
    """Per-antibody median. Monospecific names lose an ``_IgG1`` suffix."""
    block = summaries[
        (summaries["value_col"] == value_col)
        & (summaries["condition"] == condition)
        & (summaries["kind"] == kind)
    ]
    if block["antibody_name"].duplicated().any():
        raise ValueError(f"duplicate {kind} medians for {value_col} / {condition}")
    names = block["antibody_name"]
    if kind == schema.KIND_MONOSPECIFIC:
        names = names.map(strip_isotype_suffix)
        if names.duplicated().any():
            dupes = sorted(set(names[names.duplicated(keep=False)]))
            raise ValueError(f"multiple monospecific medians for {value_col}: {dupes}")
    return pd.Series(
        block["median"].to_numpy(dtype=float),
        index=pd.Index(names.to_numpy()),
        name="median",
    )


def _components(components: pd.DataFrame | None) -> pd.DataFrame:
    if components is None:
        components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")
    return (
        components[["antibody_name", "parent_a", "parent_b"]]
        .drop_duplicates("antibody_name")
        .sort_values("antibody_name", kind="mergesort")
        .reset_index(drop=True)
    )


def load_production_pairs(
    summaries: pd.DataFrame | None = None,
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per bispecific per attribute, complete parental pairs only.

    Reads the stage-02 component table and the stage-03 per-antibody medians
    when frames are not passed in. Rows missing the bispecific median or
    either parental median are dropped, so attributes can differ in length.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    base = _components(components)
    pieces: list[pd.DataFrame] = []
    for panel, key, attribute, value_col, condition in PRODUCTION_ATTRIBUTES:
        observed = _medians(summaries, value_col, condition, schema.KIND_BISPECIFIC)
        parents = _medians(summaries, value_col, condition, schema.KIND_MONOSPECIFIC)
        part = base.copy()
        part["panel"] = panel
        part["key"] = key
        part["attribute"] = attribute
        part["value_col"] = value_col
        part["condition"] = condition
        part["observed"] = part["antibody_name"].map(observed)
        part["arm_a_value"] = part["parent_a"].map(parents)
        part["arm_b_value"] = part["parent_b"].map(parents)
        part["parental_mean"] = (part["arm_a_value"] + part["arm_b_value"]) / 2.0
        part["residual"] = part["observed"] - part["parental_mean"]
        part = part.dropna(subset=["observed", "arm_a_value", "arm_b_value"])
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True)
    order = {key: i for i, (_, key, *_) in enumerate(PRODUCTION_ATTRIBUTES)}
    out["_ord"] = out["key"].map(order)
    out = (
        out.sort_values(["_ord", "antibody_name"], kind="mergesort")
        .drop(columns="_ord")
        .reset_index(drop=True)
    )
    return out.loc[:, list(PAIR_COLUMNS)]


def _arm_sigma(
    summaries: pd.DataFrame, value_col: str, condition: str
) -> tuple[float, int]:
    """Sample standard deviation of the monospecific medians, and how many."""
    parents = _medians(summaries, value_col, condition, schema.KIND_MONOSPECIFIC)
    values = parents.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(
            f"need at least two monospecific medians for {value_col} / {condition}"
        )
    return float(np.std(values, ddof=1)), int(values.size)


def tier_assignments(
    pairs: pd.DataFrame,
    summaries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per attribute: correlations, residual spread, and tier.

    ``sigma_arm`` is computed from every monospecific median, not only from
    parents that appear in ``pairs``. Pass ``pairs`` from
    :func:`load_production_pairs`.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    rows: list[dict[str, object]] = []
    for panel, key, attribute, value_col, condition in PRODUCTION_ATTRIBUTES:
        block = pairs.loc[pairs["key"] == key]
        if len(block) < 3:
            raise ValueError(f"{key}: need at least three complete bispecifics")
        predicted = block["parental_mean"].to_numpy(dtype=float)
        observed = block["observed"].to_numpy(dtype=float)
        residual = observed - predicted
        rho = float(spearmanr(predicted, observed).statistic)
        pearson = float(pearsonr(predicted, observed).statistic)
        median_residual = float(np.median(residual))
        sigma_residual = float(np.std(residual, ddof=1))
        sigma_arm, n_mab = _arm_sigma(summaries, value_col, condition)
        if not (np.isfinite(rho) and np.isfinite(pearson)):
            raise ValueError(f"{key}: correlation is not finite (n={len(block)})")
        rows.append(
            {
                "panel": panel,
                "key": key,
                "attribute": attribute,
                "value_col": value_col,
                "condition": condition,
                "n": int(len(block)),
                "n_mab": n_mab,
                "spearman_rho": rho,
                "pearson_r": pearson,
                "median_residual": median_residual,
                "sigma_residual": sigma_residual,
                "sigma_arm": sigma_arm,
                "tier": assign_tier(rho, median_residual, sigma_arm),
            }
        )
    return pd.DataFrame(rows).loc[:, list(ASSIGNMENT_COLUMNS)]


def source_scatter_table(
    pairs: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Scatter points with the panel's tier callout repeated on each row.

    One sheet can then carry both the plotted coordinates and the ρ, n, and
    class printed on the figure.
    """
    callout = [
        "key",
        "n",
        "spearman_rho",
        "pearson_r",
        "median_residual",
        "sigma_residual",
        "sigma_arm",
        "tier",
    ]
    merged = pairs.merge(assignments.loc[:, callout], on="key", how="left", validate="many_to_one")
    if merged["tier"].isna().any():
        missing = sorted(set(pairs["key"]) - set(assignments["key"]))
        raise ValueError(f"assignments missing keys: {missing}")
    return merged.loc[:, list(PAIR_COLUMNS) + callout[1:]]
