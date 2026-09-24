"""Extract source data for Supplemental Figure 8."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from prophet_ab import normalize as nz, paths, schema

OUT_DIR = Path(__file__).resolve().parent.parent / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DEGREE = 3
EXCLUDED = schema.DEPRECATED_VALUE_COLS | schema.PRODUCTION_VALUE_COLS

# ── Load and compute residuals ───────────────────────────────────────────
summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
components = pd.read_parquet(paths.S02 / "n3_components.parquet")

n4 = summaries[summaries["kind"] == schema.KIND_N4][
    ["antibody_name", "value_col", "condition", "median"]
].copy()
n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)

n3 = summaries[summaries["kind"] == schema.KIND_N3][
    ["antibody_name", "value_col", "condition", "median"]
].rename(columns={"median": "n3_median"})
n3 = n3[~n3["value_col"].isin(EXCLUDED)]

n3p = n3.merge(
    components[["antibody_name", "parent_a", "parent_b"]],
    on="antibody_name",
)
n3p = (
    n3p.merge(
        n4[["parent", "value_col", "condition", "median"]].rename(
            columns={"parent": "parent_a", "median": "pa_median"}
        ),
        on=["parent_a", "value_col", "condition"],
        how="left",
    ).merge(
        n4[["parent", "value_col", "condition", "median"]].rename(
            columns={"parent": "parent_b", "median": "pb_median"}
        ),
        on=["parent_b", "value_col", "condition"],
        how="left",
    )
)
n3p["parent_mean"] = (n3p["pa_median"] + n3p["pb_median"]) / 2
n3p["residual"] = n3p["n3_median"] - n3p["parent_mean"]
residuals_long = n3p.dropna(subset=["residual"])

# ── Build parent graph and filter by degree ──────────────────────────────
G_full = nx.Graph()
for _, row in components.iterrows():
    G_full.add_edge(row["parent_a"], row["parent_b"], n3=row["antibody_name"])

fvs_passing = sorted([n for n, d in G_full.degree() if d >= MIN_DEGREE])
fvs_set = set(fvs_passing)
fv_to_idx = {fv: i for i, fv in enumerate(fvs_passing)}
n_fv = len(fvs_passing)

# ── OLS per assay ────────────────────────────────────────────────────────
assays = (
    residuals_long.groupby(["value_col", "condition"])
    .size()
    .reset_index(name="n")
)

all_rows = []
for _, arow in assays.iterrows():
    vc, cond = arow["value_col"], arow["condition"]
    g = residuals_long[
        (residuals_long["value_col"] == vc)
        & (residuals_long["condition"] == cond)
        & residuals_long["parent_a"].isin(fvs_set)
        & residuals_long["parent_b"].isin(fvs_set)
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
    fvs_sub = [fv for fv, u in zip(fvs_passing, used) if u]
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

    for j, fv in enumerate(fvs_sub):
        all_rows.append(
            {
                "value_col": vc,
                "condition": cond,
                "fv": fv,
                "coef": float(beta[j]),
                "se": float(se[j]),
                "p_value": float(p_val[j]),
                "p_adjusted": float(p_adj[j]),
                "significant": bool(p_adj[j] < 0.05),
            }
        )

coefficients = pd.DataFrame(all_rows)

# ── Filter to reported value_cols ────────────────────────────────────────
coefficients = coefficients[
    coefficients["value_col"].isin(schema.REPORTED_VALUE_COLS)
].copy()

# ── Z-score within each assay ────────────────────────────────────────────
z_parts = []
for (vc, cond), grp in coefficients.groupby(["value_col", "condition"]):
    if grp.empty:
        continue
    sd = grp["coef"].std()
    if sd == 0 or np.isnan(sd):
        continue
    gc = grp.copy()
    gc["z_coef"] = (gc["coef"] - gc["coef"].mean()) / sd
    gc["z_se"] = gc["se"] / sd
    z_parts.append(gc)

out = pd.concat(z_parts, ignore_index=True)

# Keep requested columns
out = out[
    [
        "fv",
        "value_col",
        "condition",
        "coef",
        "se",
        "z_coef",
        "z_se",
        "p_value",
        "p_adjusted",
        "significant",
    ]
]

out_path = OUT_DIR / "supp_fig8_coefficients.csv"
out.to_csv(out_path, index=False)
print(f"wrote {out_path} ({len(out)} rows)")
