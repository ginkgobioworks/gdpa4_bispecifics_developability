"""Parental-mean HIC, HAC, and PR-CHO for supplementary figure S14.

For a bispecific built from arms *a* and *b*, the parental expectation is the
arithmetic mean of the two monospecific medians in the same assay. A
bispecific is kept for an assay only when both arms and the bispecific
itself have a median. Orientations stay as separate molecules.

The figure highlights every bispecific that contains brazikumab or
ligelizumab. ``is_brazi_lige_pair`` is true only for a molecule built from
both of those arms. This library has no such molecule.

Example::

    from prophet_ab.features.parental_surface import (
        highlight_table,
        summarize_parental_frontier,
        load_surface_long,
    )

    surface = load_surface_long()
    summary = summarize_parental_frontier(surface)
    summary["n_brazi_bsabs"]  # 16
    len(highlight_table(surface))  # 99
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import paths, schema
from ..normalize import strip_isotype_suffix

# (short key, value_col, condition). Condition is the tall-table buffer or
# PR antigen, not a display name.
SURFACE_ASSAYS: tuple[tuple[str, str, str], ...] = (
    ("hic", "hihplc_normretentiontime", "default"),
    ("hac", "hachplc_retentiontime", "default"),
    ("pr_cho", "pr_score", "CHO"),
)

BRAZIKUMAB = "brazikumab"
LIGELIZUMAB = "ligelizumab"
EXAMPLE_ARMS: tuple[str, ...] = (BRAZIKUMAB, LIGELIZUMAB)

# Hyndman–Fan type 5 (Hazen). The linear quartile does not match the
# three-decimal thresholds used in the figure (3.136 and 4.313).
QUARTILE_METHOD = "hazen"

HIGHLIGHT_COLUMNS: tuple[str, ...] = (
    "antibody_name",
    "parent_a",
    "parent_b",
    "example_arm",
    "assay",
    "value_col",
    "condition",
    "arm_a_value",
    "arm_b_value",
    "parental_mean",
    "observed",
    "residual",
)

SURFACE_COLUMNS: tuple[str, ...] = (
    "antibody_name",
    "parent_a",
    "parent_b",
    "assay",
    "value_col",
    "condition",
    "arm_a_value",
    "arm_b_value",
    "parental_mean",
    "observed",
    "residual",
    "contains_brazikumab",
    "contains_ligelizumab",
    "is_brazi_lige_pair",
)


def flag_example_arms(frame: pd.DataFrame) -> pd.DataFrame:
    """Add brazikumab / ligelizumab membership columns.

    ``is_brazi_lige_pair`` is true only when the two parents are exactly
    those two arms, in either orientation.

    Example::

        flagged = flag_example_arms(pd.DataFrame({
            "parent_a": ["brazikumab"],
            "parent_b": ["ligelizumab"],
        }))
        bool(flagged.loc[0, "is_brazi_lige_pair"])  # True
    """
    if "parent_a" not in frame.columns or "parent_b" not in frame.columns:
        raise ValueError("frame must include parent_a and parent_b")
    out = frame.copy()
    out["contains_brazikumab"] = (out["parent_a"] == BRAZIKUMAB) | (
        out["parent_b"] == BRAZIKUMAB
    )
    out["contains_ligelizumab"] = (out["parent_a"] == LIGELIZUMAB) | (
        out["parent_b"] == LIGELIZUMAB
    )
    out["is_brazi_lige_pair"] = (
        out["contains_brazikumab"] & out["contains_ligelizumab"]
    )
    return out


def example_membership(components: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per produced bispecific, with example-arm flags.

    Reads ``bispecific_components.parquet`` when ``components`` is not passed.
    """
    if components is None:
        components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")
    out = (
        components[["antibody_name", "parent_a", "parent_b"]]
        .drop_duplicates("antibody_name")
        .sort_values("antibody_name")
        .reset_index(drop=True)
    )
    return flag_example_arms(out)


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


def load_surface_long(
    summaries: pd.DataFrame | None = None,
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Complete bispecific rows for HIC, HAC, and PR-CHO.

    One row per bispecific per assay. Rows missing the bispecific median
    or either parental median are dropped, so the three assays can differ
    in length.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    base = example_membership(components)
    pieces: list[pd.DataFrame] = []
    for assay, value_col, condition in SURFACE_ASSAYS:
        observed = _medians(summaries, value_col, condition, schema.KIND_BISPECIFIC)
        parents = _medians(summaries, value_col, condition, schema.KIND_MONOSPECIFIC)
        part = base.copy()
        part["assay"] = assay
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
    order = {assay: i for i, (assay, _, _) in enumerate(SURFACE_ASSAYS)}
    out["_ord"] = out["assay"].map(order)
    out = (
        out.sort_values(["_ord", "antibody_name"], kind="mergesort")
        .drop(columns="_ord")
        .reset_index(drop=True)
    )
    return out.loc[:, list(SURFACE_COLUMNS)]


def load_example_parents(summaries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monospecific medians for brazikumab and ligelizumab on each assay.

    These are the diagonal diamonds. A monospecific's parental mean is its
    own value, so the plot places each parent at ``(value, value)``.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    rows: list[dict[str, object]] = []
    for assay, value_col, condition in SURFACE_ASSAYS:
        parents = _medians(summaries, value_col, condition, schema.KIND_MONOSPECIFIC)
        for arm in EXAMPLE_ARMS:
            if arm not in parents.index or pd.isna(parents.loc[arm]):
                raise ValueError(f"{arm} has no {assay} median")
            rows.append(
                {
                    "arm": arm,
                    "assay": assay,
                    "value_col": value_col,
                    "condition": condition,
                    "value": float(parents.loc[arm]),
                }
            )
    return pd.DataFrame(rows)


def highlight_table(surface: pd.DataFrame) -> pd.DataFrame:
    """Brazikumab- and ligelizumab-containing rows, one per assay.

    A true brazikumab × ligelizumab bispecific would appear once per assay,
    labeled ``brazikumab x ligelizumab``, not twice.
    """
    mask = surface["contains_brazikumab"] | surface["contains_ligelizumab"]
    out = surface.loc[mask].copy()
    out["example_arm"] = np.where(
        out["is_brazi_lige_pair"],
        "brazikumab x ligelizumab",
        np.where(out["contains_brazikumab"], BRAZIKUMAB, LIGELIZUMAB),
    )
    return out.loc[:, list(HIGHLIGHT_COLUMNS)].reset_index(drop=True)


def summarize_parental_frontier(
    surface: pd.DataFrame,
    components: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Spearman of parental-mean HIC vs HAC, and the double-quartile count.

    Quartiles use the Hazen definition on the bispecifics that have both
    parental means. ``n_double_frontier_all`` counts molecules strictly
    above both 75th percentiles. Bispecifics missing a parental mean cannot
    clear both cuts, so the library denominator is every produced molecule.
    """
    membership = example_membership(components)
    hic = surface.loc[
        surface["assay"] == "hic", ["antibody_name", "parental_mean"]
    ].rename(columns={"parental_mean": "hic"})
    hac = surface.loc[
        surface["assay"] == "hac", ["antibody_name", "parental_mean"]
    ].rename(columns={"parental_mean": "hac"})
    paired = hic.merge(hac, on="antibody_name", how="inner")
    if paired.empty:
        raise ValueError("no bispecifics with both parental-mean HIC and HAC")
    rho = float(spearmanr(paired["hic"], paired["hac"]).statistic)
    hic_q75 = float(np.quantile(paired["hic"], 0.75, method=QUARTILE_METHOD))
    hac_q75 = float(np.quantile(paired["hac"], 0.75, method=QUARTILE_METHOD))
    n_double = int(((paired["hic"] > hic_q75) & (paired["hac"] > hac_q75)).sum())
    return {
        "n_library": float(len(membership)),
        "n_brazi_bsabs": float(membership["contains_brazikumab"].sum()),
        "n_lige_bsabs": float(membership["contains_ligelizumab"].sum()),
        "n_brazi_lige": float(membership["is_brazi_lige_pair"].sum()),
        "spearman_phic_phac": rho,
        "spearman_n": float(len(paired)),
        "hic_q75": hic_q75,
        "hac_q75": hac_q75,
        "n_double_frontier_all": float(n_double),
    }


def source_scatter_table(
    surface: pd.DataFrame, parents: pd.DataFrame
) -> pd.DataFrame:
    """Plotted coordinates: every complete bispecific, then the two parents.

    Parent rows sit on the diagonal (``parental_mean == observed == value``).
    """
    bis = surface.copy()
    bis.insert(0, "point", "bispecific")
    parent_rows = []
    for rec in parents.itertuples(index=False):
        parent_rows.append(
            {
                "point": "parent_mab",
                "antibody_name": rec.arm,
                "parent_a": pd.NA,
                "parent_b": pd.NA,
                "assay": rec.assay,
                "value_col": rec.value_col,
                "condition": rec.condition,
                "arm_a_value": np.nan,
                "arm_b_value": np.nan,
                "parental_mean": float(rec.value),
                "observed": float(rec.value),
                "residual": 0.0,
                "contains_brazikumab": rec.arm == BRAZIKUMAB,
                "contains_ligelizumab": rec.arm == LIGELIZUMAB,
                "is_brazi_lige_pair": False,
            }
        )
    cols = ["point", *SURFACE_COLUMNS]
    stacked = pd.concat([bis, pd.DataFrame(parent_rows)], ignore_index=True)
    return stacked.loc[:, cols]
