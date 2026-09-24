"""Extract source data for Figure 3."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prophet_ab import paths, schema
from prophet_ab import normalize as nz
from prophet_ab.features.transforms import (
    TRANSFORMS,
    BEST_TRANSFORM,
    SHORT_LABEL,
    PRIMARY_METRICS,
    TIER_DEFS,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"

# Metrics excluded from Figure 3 (SEC %monomer and Tonset)
_EXCLUDE = {
    ("sehplc_pct_mono", "default"),
    ("thermostability_tonset", "Tonset"),
}

# Build tier lookup: (value_col, condition) -> tier number
_METRIC_TO_TIER = {}
for _tier_num, _members in TIER_DEFS:
    for m in _members:
        _METRIC_TO_TIER[m] = _tier_num


def _build_n3_with_parents():
    """Join N3 observed medians with parent N4 medians."""
    summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")

    n4 = summaries[summaries["kind"] == schema.KIND_N4][
        ["antibody_name", "value_col", "condition", "median"]
    ].copy()
    n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)

    n3 = summaries[summaries["kind"] == schema.KIND_N3][
        ["antibody_name", "value_col", "condition", "median"]
    ].rename(columns={"median": "observed"})

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
        )
        .merge(
            n4[["parent", "value_col", "condition", "median"]].rename(
                columns={"parent": "parent_b", "median": "pb_median"}
            ),
            on=["parent_b", "value_col", "condition"],
            how="left",
        )
    )
    return n3p.dropna(subset=["pa_median", "pb_median"])


def main():
    n3_with_parents = _build_n3_with_parents()

    # ---- Panel A scatter data: per-metric, per-operator --------------------
    # Only include the 11 metrics shown in the figure (exclude SEC %mono, Tonset)
    figure_metrics = [m for m in PRIMARY_METRICS if m not in _EXCLUDE]

    scatter_rows = []
    for vc, cond in figure_metrics:
        g = n3_with_parents[
            (n3_with_parents["value_col"] == vc)
            & (n3_with_parents["condition"] == cond)
        ]
        chosen_op = BEST_TRANSFORM[(vc, cond)]
        tier = _METRIC_TO_TIER.get((vc, cond), 0)
        pred = TRANSFORMS[chosen_op](g["pa_median"], g["pb_median"]).values
        obs = g["observed"].values
        mask = ~(np.isnan(pred) | np.isnan(obs))

        for idx in np.where(mask)[0]:
            scatter_rows.append({
                "antibody_name": g.iloc[idx]["antibody_name"],
                "value_col": vc,
                "condition": cond,
                "observed": obs[idx],
                "predicted": pred[idx],
                "operator": chosen_op,
                "tier": tier,
            })

    df_scatter = pd.DataFrame(scatter_rows)
    out_scatter = OUT_DIR / "fig3a_scatter.csv"
    out_scatter.parent.mkdir(parents=True, exist_ok=True)
    df_scatter.to_csv(out_scatter, index=False)
    print(f"Wrote {out_scatter}  ({len(df_scatter)} rows)")

    # ---- Panel B heatmap data: Spearman rho per (metric, operator) ---------
    heatmap_rows = []
    for vc, cond in figure_metrics:
        g = n3_with_parents[
            (n3_with_parents["value_col"] == vc)
            & (n3_with_parents["condition"] == cond)
        ]
        obs = g["observed"].values
        a = g["pa_median"]
        b = g["pb_median"]
        chosen_op = BEST_TRANSFORM[(vc, cond)]
        tier = _METRIC_TO_TIER.get((vc, cond), 0)

        for t_name, t_fn in TRANSFORMS.items():
            pred = t_fn(a, b).values
            mask = ~(np.isnan(pred) | np.isnan(obs))
            n = int(mask.sum())
            if n < 5:
                rho = np.nan
            else:
                rho, _ = spearmanr(pred[mask], obs[mask])
                rho = float(rho)
            heatmap_rows.append({
                "value_col": vc,
                "condition": cond,
                "label": SHORT_LABEL[(vc, cond)],
                "operator": t_name,
                "spearman_rho": rho,
                "n": n,
                "tier": tier,
                "is_chosen": t_name == chosen_op,
            })

    df_heatmap = pd.DataFrame(heatmap_rows)
    out_heatmap = OUT_DIR / "fig3b_heatmap.csv"
    df_heatmap.to_csv(out_heatmap, index=False)
    print(f"Wrote {out_heatmap}  ({len(df_heatmap)} rows)")


if __name__ == "__main__":
    main()
