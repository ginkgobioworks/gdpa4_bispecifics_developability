"""Pure functions for normalizing antibody names and parsing bispecific components."""
from __future__ import annotations

import pandas as pd

from .schema import (
    CONTROL_NAMES,
    ISOTYPE_SUFFIXES_TO_STRIP,
    KIND_BISPECIFIC,
    KIND_MONOSPECIFIC,
    BISPECIFIC_PAIR_SEP,
    BISPECIFIC_NAME_PREFIX,
)


def strip_isotype_suffix(name: str) -> str:
    for s in ISOTYPE_SUFFIXES_TO_STRIP:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def canonical_name(name: str) -> str:
    """Drop a leading ``N3-`` prefix when present.

    Bispecific names use ``parent_a__x__parent_b``. Some production records
    prefix that name with ``N3-``. Processed tables use the prefix-less form.
    """
    return name.removeprefix(BISPECIFIC_NAME_PREFIX)


def classify_kind(name: str) -> str:
    # A bispecific name contains the pair separator `__x__`.
    return KIND_BISPECIFIC if BISPECIFIC_PAIR_SEP in name else KIND_MONOSPECIFIC


def parse_bispecific_components(name: str) -> tuple[str, str] | tuple[None, None]:
    """Return (parent_a, parent_b) for an bispecific name, else (None, None).

    Tolerates an optional leading `N3-` prefix. Parents are returned with
    isotype suffix stripped so they match monospecific names.
    """
    body = canonical_name(name)
    if BISPECIFIC_PAIR_SEP not in body:
        return (None, None)
    parts = body.split(BISPECIFIC_PAIR_SEP)
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
    parents = out[name_col].map(parse_bispecific_components)
    out["parent_a"] = parents.map(lambda t: t[0])
    out["parent_b"] = parents.map(lambda t: t[1])
    out["is_control"] = out[name_col].map(
        lambda n: strip_isotype_suffix(n) in CONTROL_NAMES
    )
    return out
