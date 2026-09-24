"""Extract source data for Supplemental Figure 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from prophet_ab import paths, schema
from prophet_ab import normalize as nz

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to pivot into wide format
_METRICS = [
    ("hihplc_normretentiontime", "default", "hic"),
    ("hachplc_retentiontime", "default", "hac"),
    ("pr_score", "Ovalbumin", "pr_ova"),
    ("pr_score", "CHO", "pr_cho"),
    ("bvp_score_norm", "default", "bvp"),
]

PR_ASSAYS = [
    ("pr_ova", "PR-OVA"),
    ("pr_cho", "PR-CHO"),
    ("bvp", "PR-BVP"),
]


def _wide(summaries: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pivot summaries to one-row-per-antibody wide format."""
    df = None
    for vc, cond, alias in _METRICS:
        sub = summaries[
            (summaries["kind"] == kind)
            & (summaries["value_col"] == vc)
            & (summaries["condition"] == cond)
        ][["antibody_name", "median"]].rename(columns={"median": alias})
        if df is None:
            df = sub
        else:
            df = df.merge(sub, on="antibody_name", how="outer")
    return df


def main() -> None:
    summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")[
        ["antibody_name", "parent_a", "parent_b"]
    ]

    wide_n4 = _wide(summaries, schema.KIND_N4)
    wide_n4["parent"] = wide_n4["antibody_name"].map(nz.strip_isotype_suffix)
    wide_n3 = _wide(summaries, schema.KIND_N3)

    # ---- Pareto MAX-MAX frontier on N4 HIC x HAC ----
    core = wide_n4.dropna(subset=["hic", "hac"]).copy().reset_index(drop=True)
    pts = core[["hic", "hac"]].values
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (
                pts[j, 0] >= pts[i, 0]
                and pts[j, 1] >= pts[i, 1]
                and (pts[j, 0] > pts[i, 0] or pts[j, 1] > pts[i, 1])
            ):
                is_pareto[i] = False
                break

    frontier_core = core[is_pareto].sort_values("hic")
    frontier_pts = frontier_core[["hic", "hac"]].values
    frontier_parent_set = set(frontier_core["parent"])

    # ---- N3 pair averaging across orientations ----
    n3 = wide_n3.merge(components, on="antibody_name", how="left")
    n3 = n3.dropna(subset=["hic", "hac", "parent_a", "parent_b"]).copy()
    n3["pair_key"] = n3.apply(
        lambda r: tuple(sorted([r["parent_a"], r["parent_b"]])), axis=1
    )

    crosser_df = n3.groupby("pair_key", as_index=False).agg(
        parent_a=("parent_a", "first"),
        parent_b=("parent_b", "first"),
        antibody_name=("antibody_name", "first"),
        hic=("hic", "mean"),
        hac=("hac", "mean"),
        pr_ova=("pr_ova", "mean"),
        pr_cho=("pr_cho", "mean"),
        bvp=("bvp", "mean"),
    )

    # ---- frontier_dist ----
    fx, fy = frontier_pts[:, 0], frontier_pts[:, 1]
    crosser_df["frontier_dist"] = crosser_df["hac"] - np.interp(
        crosser_df["hic"], fx, fy, left=fy[0], right=fy[-1]
    )
    crosser_df["crossed"] = crosser_df["frontier_dist"] > 0

    # ---- Classification ----
    def _categorize(row: pd.Series) -> str:
        if not row["crossed"]:
            return "Within"
        a_on = row["parent_a"] in frontier_parent_set
        b_on = row["parent_b"] in frontier_parent_set
        if a_on and b_on:
            return "Additive"
        if a_on or b_on:
            return "Partial"
        return "Emergent"

    crosser_df["category"] = crosser_df.apply(_categorize, axis=1)

    # ---- Panel A: scatter data + frontier ----
    # N3 rows
    scatter_n3 = crosser_df[
        ["pair_key", "hic", "hac", "pr_cho", "pr_ova", "bvp", "category", "frontier_dist"]
    ].copy()
    scatter_n3["pair_key"] = scatter_n3["pair_key"].apply(
        lambda t: f"{t[0]}__{t[1]}"
    )
    scatter_n3["kind"] = "N3"

    # N4 rows
    scatter_n4 = wide_n4.dropna(subset=["hic", "hac"]).copy()
    scatter_n4["pair_key"] = scatter_n4["parent"]
    scatter_n4["kind"] = "N4"
    scatter_n4["category"] = ""
    scatter_n4["frontier_dist"] = np.nan
    scatter_n4 = scatter_n4[
        ["pair_key", "hic", "hac", "pr_cho", "pr_ova", "bvp", "category", "frontier_dist", "kind"]
    ]

    scatter = pd.concat(
        [scatter_n3[scatter_n4.columns], scatter_n4], ignore_index=True
    )
    out_scatter = OUT_DIR / "supp_fig4a_scatter.csv"
    scatter.to_csv(out_scatter, index=False)
    print(f"Panel A scatter: wrote {len(scatter)} rows to {out_scatter}")

    frontier_df = pd.DataFrame(frontier_pts, columns=["hic", "hac"])
    out_frontier = OUT_DIR / "supp_fig4a_frontier.csv"
    frontier_df.to_csv(out_frontier, index=False)
    print(f"Panel A frontier: wrote {len(frontier_df)} rows to {out_frontier}")

    # ---- Panel B: ROC curves ----
    roc_rows = []
    for col, label in PR_ASSAYS:
        valid = crosser_df[["frontier_dist", col]].dropna()
        x = valid["frontier_dist"].values
        y = valid[col].values
        poly = (y > np.percentile(y, 80)).astype(int)
        fpr, tpr, _ = roc_curve(poly, x)
        auc = roc_auc_score(poly, x)

        # 2000-iteration bootstrap for 95% CI
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(poly), len(poly))
            if len(np.unique(poly[idx])) < 2:
                continue
            boot.append(roc_auc_score(poly[idx], x[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        for i in range(len(fpr)):
            roc_rows.append(
                {
                    "pr_assay": label,
                    "fpr": fpr[i],
                    "tpr": tpr[i],
                    "auc": auc,
                    "auc_ci_lo": lo,
                    "auc_ci_hi": hi,
                }
            )

    roc_df = pd.DataFrame(roc_rows)
    out_roc = OUT_DIR / "supp_fig4b_roc.csv"
    roc_df.to_csv(out_roc, index=False)
    print(f"Panel B ROC: wrote {len(roc_df)} rows to {out_roc}")

    # ---- Panel C: confusion matrix ----
    valid = crosser_df[["frontier_dist", "pr_cho"]].dropna()
    x = valid["frontier_dist"].values
    y = valid["pr_cho"].values
    pred = (x > 0).astype(int)
    true = (y > np.percentile(y, 80)).astype(int)

    confusion_rows = []
    for crossed_val in [0, 1]:
        for top20_val in [0, 1]:
            count = int(((pred == crossed_val) & (true == top20_val)).sum())
            confusion_rows.append(
                {
                    "crossed": bool(crossed_val),
                    "top20_pr_cho": bool(top20_val),
                    "count": count,
                }
            )

    confusion_df = pd.DataFrame(confusion_rows)
    out_conf = OUT_DIR / "supp_fig4c_confusion.csv"
    confusion_df.to_csv(out_conf, index=False)
    print(f"Panel C confusion: wrote {len(confusion_df)} rows to {out_conf}")


if __name__ == "__main__":
    main()
