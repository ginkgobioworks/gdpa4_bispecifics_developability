"""Registry of in-silico predictors and a uniform reader.

Each source is a CSV keyed by `antibody_name` (matching the GDPa1 naming
convention, i.e. bare INN without isotype suffix). The loader returns a
long-form DataFrame with columns (antibody_name, source, feature, value).

Numeric features only — non-numeric columns listed in `id_cols` are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import paths


@dataclass(frozen=True)
class InSilicoSource:
    name: str
    filename: str
    id_cols: tuple[str, ...]   # columns that identify rows, not features


SOURCES: tuple[InSilicoSource, ...] = (
    InSilicoSource("Aggrescan3D", "Aggrescan3D.csv",
                   id_cols=("antibody_name",)),
    InSilicoSource("AntiFold", "AntiFold.csv",
                   id_cols=("antibody_name",)),
    InSilicoSource("DeepViscosity", "DeepViscosity.csv",
                   id_cols=("antibody_name",)),
    InSilicoSource("MOE", "MOE_properties.csv",
                   id_cols=("antibody_id", "antibody_name", "mseq")),
    InSilicoSource("Saprot_VH", "Saprot_VH.csv",
                   id_cols=("antibody_name",)),
    InSilicoSource("TAP", "TAP.csv",
                   id_cols=("antibody_name",)),
)


def read_source(source: InSilicoSource) -> pd.DataFrame:
    """Read one CSV and return as-is (preserves id_cols)."""
    return pd.read_csv(paths.RAW_IN_SILICO_DIR / source.filename)


def to_long(source: InSilicoSource, df: pd.DataFrame) -> pd.DataFrame:
    """Melt one source's wide table to (antibody_name, source, feature, value)."""
    feat_cols = [c for c in df.columns if c not in source.id_cols]
    long = df.melt(
        id_vars=["antibody_name"],
        value_vars=feat_cols,
        var_name="feature",
        value_name="value",
    )
    long["source"] = source.name
    return long[["antibody_name", "source", "feature", "value"]]


def load_all_long() -> pd.DataFrame:
    """Concatenate every source into a single long-form table."""
    parts = [to_long(s, read_source(s)) for s in SOURCES]
    return pd.concat(parts, ignore_index=True)
