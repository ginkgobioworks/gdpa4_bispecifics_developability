"""Pure functions for normalizing antibody names and parsing N3 components."""
from __future__ import annotations

import pandas as pd

from .schema import (
    CONTROL_NAMES,
    ISOTYPE_SUFFIXES_TO_STRIP,
    KIND_N3,
    KIND_N4,
    N3_PAIR_SEP,
    N3_PREFIX,
)


def strip_isotype_suffix(name: str) -> str:
    for s in ISOTYPE_SUFFIXES_TO_STRIP:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def canonical_name(name: str) -> str:
    """Canonical antibody name: drop a leading `N3-` prefix if present.

    Raw sources are inconsistent: the tall CSV names bispecifics
    `parent_a__x__parent_b` (no prefix) while `n3_production.xlsx` keeps the
    historical `N3-parent_a__x__parent_b` form. Canonicalize to the prefix-less
    convention so every processed table shares one naming scheme.
    """
    return name.removeprefix(N3_PREFIX)


def classify_kind(name: str) -> str:
    # Bispecifics (N3) are the only names containing the pair separator; the
    # `N3-` prefix is no longer emitted by the tall export, so key on `__x__`.
    return KIND_N3 if N3_PAIR_SEP in name else KIND_N4


def parse_n3_components(name: str) -> tuple[str, str] | tuple[None, None]:
    """Return (parent_a, parent_b) for an N3 name, else (None, None).

    Tolerates an optional leading `N3-` prefix. Parents are returned with
    isotype suffix stripped so they match N4 names.
    """
    body = canonical_name(name)
    if N3_PAIR_SEP not in body:
        return (None, None)
    parts = body.split(N3_PAIR_SEP)
    if len(parts) != 2:
        return (None, None)
    return (strip_isotype_suffix(parts[0]), strip_isotype_suffix(parts[1]))


def annotate(df: pd.DataFrame, name_col: str = "antibody_name") -> pd.DataFrame:
    """Add `kind`, `parent_a`, `parent_b`, `is_control` columns. Pure.

    Also canonicalizes `name_col` (strips a leading `N3-`) so downstream joins
    across sources use one naming convention.
    """
    out = df.copy()
    out[name_col] = out[name_col].map(canonical_name)
    out["kind"] = out[name_col].map(classify_kind)
    parents = out[name_col].map(parse_n3_components)
    out["parent_a"] = parents.map(lambda t: t[0])
    out["parent_b"] = parents.map(lambda t: t[1])
    out["is_control"] = out[name_col].map(
        lambda n: strip_isotype_suffix(n) in CONTROL_NAMES
    )
    return out
