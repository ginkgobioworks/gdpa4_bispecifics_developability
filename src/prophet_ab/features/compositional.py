"""Operators that combine two parent values into a bispecific baseline.

Pure functions. Each takes two pandas Series (parent_a, parent_b) of equal
length and returns a Series of the same length. NaN-propagating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPERATORS: dict[str, callable] = {}


def _register(name: str):
    def deco(fn):
        OPERATORS[name] = fn
        return fn
    return deco


@_register("mean")
def mean(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a + b) / 2.0


@_register("min")
def minimum(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).min(axis=1, skipna=False)


@_register("max")
def maximum(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).max(axis=1, skipna=False)


@_register("geomean")
def geomean(a: pd.Series, b: pd.Series) -> pd.Series:
    """Geometric mean. Returns NaN where either input is non-positive."""
    safe = (a > 0) & (b > 0)
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    out[safe] = np.sqrt(a[safe] * b[safe])
    return out


@_register("abs_diff")
def abs_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Asymmetry: |parent_a - parent_b|. Order-invariant."""
    return (a - b).abs()


def apply_all(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with one column per registered operator."""
    return pd.DataFrame({name: fn(a, b) for name, fn in OPERATORS.items()})
