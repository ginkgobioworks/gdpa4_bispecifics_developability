"""Per-N3 wide feature matrices and label tables.

Three builders, each returning a DataFrame indexed by N3 antibody_name:

- `build_measured_wide` — measured N4 features applied through every operator
  in `compositional.OPERATORS`. Columns: `meas__{value_col}__{condition}__{op}`.
- `build_in_silico_wide` — pivots stage-04 in-silico per-N3 long form.
  Columns: `is__{source}__{feature}__{op}`.
- `build_labels_wide` — per-N3 measured medians (the prediction targets).
  Columns: `label__{value_col}__{condition}`.

Column naming convention is flat with `__` delimiter so configs can subset
by string prefix or pattern match.
"""
from __future__ import annotations

import pandas as pd

from .. import normalize as nz
from .. import schema
from . import compositional


MEAS_PREFIX = "meas"
IS_PREFIX = "is"
ARM_MEAS_PREFIX = "arm_meas"
ARM_IS_PREFIX = "arm_is"
LABEL_PREFIX = "label"
SEP = "__"


def _meas_col(value_col: str, condition: str, op: str) -> str:
    return SEP.join([MEAS_PREFIX, value_col, condition, op])


def _is_col(source: str, feature: str, op: str) -> str:
    return SEP.join([IS_PREFIX, source, feature, op])


def _arm_meas_col(value_col: str, condition: str, arm: str) -> str:
    return SEP.join([ARM_MEAS_PREFIX, value_col, condition, arm])


def _arm_is_col(source: str, feature: str, arm: str) -> str:
    return SEP.join([ARM_IS_PREFIX, source, feature, arm])


def _label_col(value_col: str, condition: str) -> str:
    return SEP.join([LABEL_PREFIX, value_col, condition])


def build_measured_wide(per_antibody: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Apply every compositional operator to parent N4 medians, per (value_col, condition).

    Filters out value_cols listed in `schema.EXCLUDED_FROM_MODELING`.
    """
    n4 = per_antibody[
        (per_antibody["kind"] == schema.KIND_N4)
        & (~per_antibody["value_col"].isin(schema.EXCLUDED_FROM_MODELING))
    ][["antibody_name", "value_col", "condition", "median"]].copy()
    n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)
    n4_lookup = n4[["parent", "value_col", "condition", "median"]]

    base = components[["antibody_name", "parent_a", "parent_b"]]
    merged = (
        base.merge(
            n4_lookup.rename(columns={"parent": "parent_a", "median": "a"}),
            on="parent_a", how="left",
        )
        .merge(
            n4_lookup.rename(columns={"parent": "parent_b", "median": "b"}),
            on=["parent_b", "value_col", "condition"], how="left",
        )
    )

    parts = []
    for op_name, op_fn in compositional.OPERATORS.items():
        v = op_fn(merged["a"], merged["b"])
        parts.append(merged.assign(operator=op_name, value=v))
    long = pd.concat(parts, ignore_index=True)
    long["col"] = long.apply(
        lambda r: _meas_col(r["value_col"], r["condition"], r["operator"]), axis=1
    )
    wide = long.pivot_table(index="antibody_name", columns="col", values="value", aggfunc="first")
    return wide


def build_arm_measured_wide(per_antibody: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Raw parent arm values per (value_col, condition) — no compositional operators.

    Produces two columns per measurement: arm_meas__{vc}__{cond}__a and __b,
    using the structural ordering from the N3 name.
    """
    n4 = per_antibody[
        (per_antibody["kind"] == schema.KIND_N4)
        & (~per_antibody["value_col"].isin(schema.EXCLUDED_FROM_MODELING))
    ][["antibody_name", "value_col", "condition", "median"]].copy()
    n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)
    n4_lookup = n4[["parent", "value_col", "condition", "median"]]

    base = components[["antibody_name", "parent_a", "parent_b"]]
    merged = (
        base.merge(
            n4_lookup.rename(columns={"parent": "parent_a", "median": "a"}),
            on="parent_a", how="left",
        )
        .merge(
            n4_lookup.rename(columns={"parent": "parent_b", "median": "b"}),
            on=["parent_b", "value_col", "condition"], how="left",
        )
    )

    parts = []
    for arm in ("a", "b"):
        part = merged[["antibody_name", "value_col", "condition", arm]].rename(columns={arm: "value"})
        part = part.assign(arm=arm)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["col"] = long.apply(
        lambda r: _arm_meas_col(r["value_col"], r["condition"], r["arm"]), axis=1
    )
    return long.pivot_table(index="antibody_name", columns="col", values="value", aggfunc="first")


def build_in_silico_wide(per_n3_long: pd.DataFrame) -> pd.DataFrame:
    df = per_n3_long.copy()
    df["col"] = df.apply(
        lambda r: _is_col(r["source"], r["feature"], r["operator"]), axis=1
    )
    wide = df.pivot_table(index="antibody_name", columns="col", values="value", aggfunc="first")
    return wide


def build_arm_is_wide(per_n4_long: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Raw parent arm values for in-silico features — no compositional operators.

    Takes per-N4 in-silico long form (parent, source, feature, value) and the
    N3 components table. Produces two columns per (source, feature): __a and __b.
    """
    lookup = per_n4_long[["parent", "source", "feature", "value"]]
    base = components[["antibody_name", "parent_a", "parent_b"]]
    merged = (
        base.merge(
            lookup.rename(columns={"parent": "parent_a", "value": "a"}),
            on="parent_a", how="left",
        )
        .merge(
            lookup.rename(columns={"parent": "parent_b", "value": "b"}),
            on=["parent_b", "source", "feature"], how="left",
        )
    )

    parts = []
    for arm in ("a", "b"):
        part = merged[["antibody_name", "source", "feature", arm]].rename(columns={arm: "value"})
        part = part.assign(arm=arm)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["col"] = long.apply(
        lambda r: _arm_is_col(r["source"], r["feature"], r["arm"]), axis=1
    )
    return long.pivot_table(index="antibody_name", columns="col", values="value", aggfunc="first")


def build_labels_wide(per_antibody: pd.DataFrame) -> pd.DataFrame:
    """Per-N3 labels (medians). Filters out `schema.EXCLUDED_FROM_MODELING`."""
    n3 = per_antibody[
        (per_antibody["kind"] == schema.KIND_N3)
        & (~per_antibody["value_col"].isin(schema.EXCLUDED_FROM_MODELING))
    ][["antibody_name", "value_col", "condition", "median"]].copy()
    n3["col"] = n3.apply(lambda r: _label_col(r["value_col"], r["condition"]), axis=1)
    wide = n3.pivot_table(index="antibody_name", columns="col", values="median", aggfunc="first")
    return wide


def label_to_meas_prefix(label_col: str) -> str:
    """For a label like 'label__hic_rt__default', return its meas prefix
    'meas__hic_rt__default__' (used by the corresponding-experimental config)."""
    assert label_col.startswith(LABEL_PREFIX + SEP)
    body = label_col[len(LABEL_PREFIX) + len(SEP):]
    return MEAS_PREFIX + SEP + body + SEP


def label_to_arm_meas_prefix(label_col: str) -> str:
    """For a label like 'label__hic_rt__default', return its arm_meas prefix
    'arm_meas__hic_rt__default__' (used by the arm-corresponding config)."""
    assert label_col.startswith(LABEL_PREFIX + SEP)
    body = label_col[len(LABEL_PREFIX) + len(SEP):]
    return ARM_MEAS_PREFIX + SEP + body + SEP
