"""Per-Fv bispecific format-effect model over the connected parent graph.

Single source of truth for the OLS that s08 (D-2026-05-05-FORMAT-EFFECTS,
D-2026-05-05-MIN-DEGREE) fits: for each assay, the bispecific-vs-mean(parents)
residual is decomposed into a sum of per-Fv format effects

    r_ij = bs_i + bs_j + eps        (OLS, no intercept)

restricted to Fvs that appear in at least ``min_degree`` bispecific
combinations (degree >= 3 by default). Per-Fv t-tests with Benjamini-Hochberg
FDR correction identify Fvs whose variable region behaves systematically
differently in bispecific format ("translators").

The s08 notebook implements this inline; this module lifts it into importable
functions so any downstream stage (e.g. the stage-07 translation filter) reuses
the exact same fit rather than re-deriving it or reading the notebook's CSV
output. ``fit_format_effects`` reproduces
``reports/tables/s08_format_effect_coefficients.csv`` column-for-column.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .. import normalize as nz
from .. import schema

MIN_DEGREE = 3


def _components_from_names(n3_names) -> pd.DataFrame:
    """(antibody_name, parent_a, parent_b) parsed from N3 names.

    Robust to an optional ``N3-`` prefix (``normalize.parse_n3_components``),
    so it does not depend on the possibly-stale stage-02 parquet.
    """
    parsed = [nz.parse_n3_components(n) for n in n3_names]
    return pd.DataFrame(
        {
            "antibody_name": list(n3_names),
            "parent_a": [p[0] for p in parsed],
            "parent_b": [p[1] for p in parsed],
        }
    ).dropna(subset=["parent_a", "parent_b"])


def build_residuals_long(
    per_antibody: pd.DataFrame, components: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Bispecific residuals off the compositional parent mean, in long form.

    For every (N3, value_col, condition): ``residual = n3_median -
    mean(parent_a_median, parent_b_median)``. Excludes
    ``schema.EXCLUDED_FROM_MODELING`` value_cols. Mirrors the s08 residuals
    cell. Returns rows with ``parent_a``, ``parent_b``, ``residual`` populated.
    """
    n4 = per_antibody[per_antibody["kind"] == schema.KIND_N4][
        ["antibody_name", "value_col", "condition", "median"]
    ].copy()
    n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)
    n4_lookup = n4[["parent", "value_col", "condition", "median"]]

    n3 = per_antibody[per_antibody["kind"] == schema.KIND_N3][
        ["antibody_name", "value_col", "condition", "median"]
    ].rename(columns={"median": "n3_median"})
    n3 = n3[~n3["value_col"].isin(schema.EXCLUDED_FROM_MODELING)]

    if components is None:
        components = _components_from_names(n3["antibody_name"].unique())

    merged = (
        n3.merge(components[["antibody_name", "parent_a", "parent_b"]], on="antibody_name")
        .merge(
            n4_lookup.rename(columns={"parent": "parent_a", "median": "pa_median"}),
            on=["parent_a", "value_col", "condition"], how="left",
        )
        .merge(
            n4_lookup.rename(columns={"parent": "parent_b", "median": "pb_median"}),
            on=["parent_b", "value_col", "condition"], how="left",
        )
    )
    merged["parent_mean"] = (merged["pa_median"] + merged["pb_median"]) / 2
    merged["residual"] = merged["n3_median"] - merged["parent_mean"]
    return merged.dropna(subset=["residual"])


def passing_fvs(components: pd.DataFrame, min_degree: int = MIN_DEGREE) -> list[str]:
    """Fvs with degree >= ``min_degree`` in the bispecific parent graph."""
    g = nx.Graph()
    for _, row in components.iterrows():
        g.add_edge(row["parent_a"], row["parent_b"])
    return sorted([n for n, d in g.degree() if d >= min_degree])


def fit_format_effects(
    per_antibody: pd.DataFrame,
    components: pd.DataFrame | None = None,
    min_degree: int = MIN_DEGREE,
) -> pd.DataFrame:
    """Per-Fv format-effect coefficients per assay (reproduces s08).

    Fits ``residual = bs_a + bs_b`` (OLS, no intercept) per (value_col,
    condition) over N3s whose both parents pass the degree threshold. Returns
    one row per (value_col, condition, fv) with the same columns as
    ``reports/tables/s08_format_effect_coefficients.csv``.
    """
    resid = build_residuals_long(per_antibody, components)
    if components is None:
        components = _components_from_names(resid["antibody_name"].unique())

    fvs = passing_fvs(components, min_degree)
    fvs_set = set(fvs)
    fv_to_idx = {fv: i for i, fv in enumerate(fvs)}
    n_fv = len(fvs)

    assays = resid.groupby(["value_col", "condition"]).size().reset_index(name="n")

    rows = []
    for _, arow in assays.iterrows():
        vc, cond = arow["value_col"], arow["condition"]
        g = resid[
            (resid["value_col"] == vc)
            & (resid["condition"] == cond)
            & resid["parent_a"].isin(fvs_set)
            & resid["parent_b"].isin(fvs_set)
        ].reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue

        y = g["residual"].values
        X = np.zeros((n, n_fv))
        X[np.arange(n), g["parent_a"].map(fv_to_idx).values] = 1.0
        X[np.arange(n), g["parent_b"].map(fv_to_idx).values] = 1.0

        used = X.sum(axis=0) > 0
        X_sub = X[:, used]
        fvs_sub = [fv for fv, u in zip(fvs, used) if u]
        p = X_sub.shape[1]
        if n <= p or p == 0:
            continue

        beta, _, rank, _ = np.linalg.lstsq(X_sub, y, rcond=None)
        y_hat = X_sub @ beta
        rss = float(np.sum((y - y_hat) ** 2))
        dof = n - int(rank)
        if dof <= 0:
            continue
        sigma2 = rss / dof

        try:
            xtx_inv = np.linalg.inv(X_sub.T @ X_sub)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(X_sub.T @ X_sub)

        se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
        t_stat = np.where(se > 0, beta / se, 0.0)
        p_val = 2 * (1 - sp_stats.t.cdf(np.abs(t_stat), dof))
        p_adj = sp_stats.false_discovery_control(p_val, method="bh")

        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - rss / tss if tss > 0 else np.nan
        f_stat = ((tss - rss) / p) / (rss / dof) if dof > 0 else np.nan
        f_pval = (
            1 - sp_stats.f.cdf(f_stat, p, dof) if np.isfinite(f_stat) else np.nan
        )

        for j, fv in enumerate(fvs_sub):
            rows.append({
                "value_col": vc,
                "condition": cond,
                "fv": fv,
                "degree": int(X_sub[:, j].sum()),
                "coef": float(beta[j]),
                "se": float(se[j]),
                "t_stat": float(t_stat[j]),
                "p_value": float(p_val[j]),
                "p_adj": float(p_adj[j]),
                "significant": bool(p_adj[j] < 0.05),
                "n_obs": n,
                "n_fv": p,
                "model_r2": r2,
                "model_f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
                "model_f_pval": float(f_pval) if np.isfinite(f_pval) else np.nan,
            })

    return pd.DataFrame(rows)


def significant_assay_counts(coef: pd.DataFrame) -> pd.Series:
    """Number of FDR-significant assays per Fv (index=fv)."""
    return (
        coef[coef["significant"]]
        .groupby("fv")
        .size()
        .rename("n_sig_assays")
        .sort_values(ascending=False)
    )


def flagged_parents(coef: pd.DataFrame, min_sig_assays: int = 3) -> set[str]:
    """Parents ("translators") FDR-significant in >= ``min_sig_assays`` assays.

    Global-union rule (D-2026-07-22-TRANSLATION-FILTER): a single set of
    parents flagged across all assays.
    """
    counts = significant_assay_counts(coef)
    return set(counts[counts >= min_sig_assays].index)


def clean_panel(n3_names, bad_parents: set[str]) -> list[str]:
    """N3 names whose neither parent is in ``bad_parents`` (order preserved)."""
    bad = set(bad_parents)
    kept = []
    for name in n3_names:
        pa, pb = nz.parse_n3_components(name)
        if pa is None:
            continue
        if pa not in bad and pb not in bad:
            kept.append(name)
    return kept
