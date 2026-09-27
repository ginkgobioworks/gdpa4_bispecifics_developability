"""A-B versus B-A orientation pairs for supplementary figure S16.

Reviewer 1 asked for the direct comparison that Figure 2 summarizes with a
minimum and a maximum. A bispecific is orientation-paired when the library
contains both ``A__x__B`` and ``B__x__A`` as separate molecules. Arms are
sorted lexicographically so the assignment is reproducible: the first arm
is A, the second is B, and the A-B configuration is the molecule whose
``parent_a`` sorts first. Self-pairs (``parent_a == parent_b``) are
excluded, and a sorted arm pair is kept only when it has exactly two
molecules.

That rule yields 21 unordered pairs. A pair is plotted for an assay only
when both orientations have a median. Tm2 is unresolved in seven of those
pairs, so that panel has 14 points and the table has 98 rows
(21 + 21 + 21 + 14 + 21). ``difference`` is B-A minus A-B.

Panel order is AC-SINS, HIC-HPLC norm RT, SE-HPLC percent monomer, Tm2, then
PR-CHO. Spearman ρ on the repository medians, rounded to the two decimals
quoted in the note: AC-SINS PBS 0.94, HIC-HPLC norm RT 0.91, PR-CHO 0.82, Tm2
0.59, SEC percent monomer 0.14.

Example::

    from prophet_ab.features.orientation import (
        load_orientation_pairs,
        orientation_correlations,
        orientation_pair_index,
    )

    len(orientation_pair_index())          # 21
    pairs = load_orientation_pairs()
    pairs.shape[0]                         # 98
    orientation_correlations(pairs).loc[
        lambda d: d["key"] == "sec_mono", "spearman_rho"
    ].item()                               # 0.139...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import paths, schema

# (panel letter, short key, display name, value_col, condition).
# Display names are the assay labels in the S16 note. Conditions match the
# tall assay table, not the workbook's flattened headers.
ORIENTATION_ASSAYS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "A",
        "acsins",
        "AC-SINS ΔLmax, PBS",
        "acsins_delta_Lmax",
        "1X PBS",
    ),
    (
        "B",
        "hic",
        "HIC-HPLC norm RT",
        "hihplc_normretentiontime",
        "default",
    ),
    (
        "C",
        "sec_mono",
        "SE-HPLC % monomer",
        "sehplc_pct_mono",
        "default",
    ),
    (
        "D",
        "tm2",
        "Tm2",
        "thermostability_tm2",
        "Tm2",
    ),
    (
        "E",
        "pr_cho",
        "PR-CHO score",
        "pr_score",
        "CHO",
    ),
)

INDEX_COLUMNS: tuple[str, ...] = ("arm_a", "arm_b", "ab_name", "ba_name")

PAIR_COLUMNS: tuple[str, ...] = (
    "panel",
    "key",
    "assay",
    "value_col",
    "condition",
    "arm_a",
    "arm_b",
    "ab_name",
    "ba_name",
    "A_B",
    "B_A",
    "difference",
)

CORRELATION_COLUMNS: tuple[str, ...] = (
    "panel",
    "key",
    "assay",
    "value_col",
    "condition",
    "n",
    "spearman_rho",
)


def _components(components: pd.DataFrame | None) -> pd.DataFrame:
    if components is None:
        components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")
    needed = ["antibody_name", "parent_a", "parent_b"]
    missing = [col for col in needed if col not in components.columns]
    if missing:
        raise ValueError(f"components missing columns: {missing}")
    frame = components.loc[:, needed].copy()
    if frame["antibody_name"].duplicated().any():
        dupes = sorted(set(frame.loc[frame["antibody_name"].duplicated(), "antibody_name"]))
        raise ValueError(f"duplicate antibody_name in components: {dupes}")
    return frame.dropna(subset=["parent_a", "parent_b"]).reset_index(drop=True)


def orientation_pair_index(components: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per unordered orientation pair, with lex-sorted axis assignment.

    ``arm_a`` sorts before ``arm_b``. ``ab_name`` is the A-B molecule
    (``parent_a == arm_a``) and ``ba_name`` is the B-A molecule. Groups
    that are not exactly one of each orientation are rejected when they
    have two members, and ignored otherwise.
    """
    frame = _components(components)
    frame = frame.loc[frame["parent_a"] != frame["parent_b"]].copy()
    arms = frame.loc[:, ["parent_a", "parent_b"]]
    frame["arm_a"] = arms.min(axis=1)
    frame["arm_b"] = arms.max(axis=1)
    if not (frame["arm_a"] < frame["arm_b"]).all():
        raise ValueError("lexicographic arm sort did not order arm_a before arm_b")

    sizes = frame.groupby(["arm_a", "arm_b"], sort=True).size()
    paired = sizes[sizes == 2]
    if paired.empty:
        return pd.DataFrame(columns=list(INDEX_COLUMNS))

    grouped = frame.set_index(["arm_a", "arm_b"]).loc[paired.index].reset_index()
    is_ab = (grouped["parent_a"] == grouped["arm_a"]) & (
        grouped["parent_b"] == grouped["arm_b"]
    )
    is_ba = (grouped["parent_a"] == grouped["arm_b"]) & (
        grouped["parent_b"] == grouped["arm_a"]
    )
    n_ab = grouped.loc[is_ab].groupby(["arm_a", "arm_b"], sort=False).size()
    n_ba = grouped.loc[is_ba].groupby(["arm_a", "arm_b"], sort=False).size()
    one_each = n_ab.reindex(paired.index).eq(1) & n_ba.reindex(paired.index).eq(1)
    if not bool(one_each.all()):
        raise ValueError(
            "an orientation pair must contain one A-B molecule and one B-A molecule"
        )

    ab = grouped.loc[is_ab, ["arm_a", "arm_b", "antibody_name"]].rename(
        columns={"antibody_name": "ab_name"}
    )
    ba = grouped.loc[is_ba, ["arm_a", "arm_b", "antibody_name"]].rename(
        columns={"antibody_name": "ba_name"}
    )
    out = ab.merge(ba, on=["arm_a", "arm_b"], how="inner", validate="one_to_one")
    return (
        out.sort_values(["arm_a", "arm_b"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, list(INDEX_COLUMNS)]
    )


def _bispecific_medians(summaries: pd.DataFrame, value_col: str, condition: str) -> pd.Series:
    """Per-bispecific median for one assay. Duplicate names are an error."""
    block = summaries[
        (summaries["value_col"] == value_col)
        & (summaries["condition"] == condition)
        & (summaries["kind"] == schema.KIND_BISPECIFIC)
    ]
    if block["antibody_name"].duplicated().any():
        raise ValueError(f"duplicate bispecific medians for {value_col} / {condition}")
    return pd.Series(
        block["median"].to_numpy(dtype=float),
        index=pd.Index(block["antibody_name"].to_numpy()),
        name="median",
    )


def load_orientation_pairs(
    summaries: pd.DataFrame | None = None,
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per orientation pair per assay, both medians required.

    Reads the stage-02 component table and the stage-03 per-antibody medians
    when frames are not passed in. A library pair that lacks either
    orientation on an assay is omitted from that assay only.
    """
    if summaries is None:
        summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    index = orientation_pair_index(components)
    pieces: list[pd.DataFrame] = []
    for panel, key, assay, value_col, condition in ORIENTATION_ASSAYS:
        observed = _bispecific_medians(summaries, value_col, condition)
        part = index.copy()
        part["panel"] = panel
        part["key"] = key
        part["assay"] = assay
        part["value_col"] = value_col
        part["condition"] = condition
        part["A_B"] = part["ab_name"].map(observed)
        part["B_A"] = part["ba_name"].map(observed)
        part["difference"] = part["B_A"] - part["A_B"]
        finite = np.isfinite(part["A_B"].to_numpy(dtype=float)) & np.isfinite(
            part["B_A"].to_numpy(dtype=float)
        )
        pieces.append(part.loc[finite])
    out = pd.concat(pieces, ignore_index=True)
    order = {key: i for i, (_, key, *_) in enumerate(ORIENTATION_ASSAYS)}
    out["_ord"] = out["key"].map(order)
    out = (
        out.sort_values(["_ord", "arm_a", "arm_b"], kind="mergesort")
        .drop(columns="_ord")
        .reset_index(drop=True)
    )
    return out.loc[:, list(PAIR_COLUMNS)]


def orientation_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between A-B and B-A, one row per assay.

    Pass ``pairs`` from :func:`load_orientation_pairs`. Pairs with a missing
    orientation are already absent, which is why Tm2 has a smaller n.
    """
    rows: list[dict[str, object]] = []
    for panel, key, assay, value_col, condition in ORIENTATION_ASSAYS:
        block = pairs.loc[pairs["key"] == key]
        if len(block) < 3:
            raise ValueError(f"{key}: need at least three complete orientation pairs")
        x = block["A_B"].to_numpy(dtype=float)
        y = block["B_A"].to_numpy(dtype=float)
        rho = float(spearmanr(x, y).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"{key}: Spearman ρ is not finite (n={len(block)})")
        rows.append(
            {
                "panel": panel,
                "key": key,
                "assay": assay,
                "value_col": value_col,
                "condition": condition,
                "n": int(len(block)),
                "spearman_rho": rho,
            }
        )
    return pd.DataFrame(rows).loc[:, list(CORRELATION_COLUMNS)]


def source_pair_table(pairs: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    """Pair rows with that assay's n and Spearman ρ repeated on each row.

    One sheet then carries the plotted coordinates and the numbers printed
    on the figure.
    """
    callout = ["key", "n", "spearman_rho"]
    merged = pairs.merge(
        correlations.loc[:, callout],
        on="key",
        how="left",
        validate="many_to_one",
    )
    if merged["spearman_rho"].isna().any():
        missing = sorted(set(pairs["key"]) - set(correlations["key"]))
        raise ValueError(f"correlations missing keys: {missing}")
    return merged.loc[:, list(PAIR_COLUMNS) + ["n", "spearman_rho"]]
