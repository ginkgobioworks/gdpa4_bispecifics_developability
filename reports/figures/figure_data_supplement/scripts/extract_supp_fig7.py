"""Extract source data for Supplemental Figure 6."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import pandas as pd

from prophet_ab import paths, schema
from prophet_ab.features.naming import parse_label_name

OUT_DIR = Path(__file__).resolve().parent.parent / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────
m = pd.read_parquet(paths.S05 / "cv_metrics_loo.parquet")
imp = pd.read_parquet(paths.S05 / "feature_importance_long.parquet")

# Add abs_perm column used for sorting
imp = imp.assign(abs_perm=imp["perm_importance_mean"].abs())

# ── Filter to reported labels ────────────────────────────────────────────
def _is_reported(label_col: str) -> bool:
    vc, _ = parse_label_name(label_col)
    return vc in schema.REPORTED_VALUE_COLS

m_reported = m[m["label"].map(_is_reported)].copy()
imp_reported = imp[imp["label"].map(_is_reported)].copy()

# ── Non-arm in-silico configs ────────────────────────────────────────────
NONARM_IS_CONFIGS = (
    "in_silico_only",
    "in_silico_plus_corresponding",
    "in_silico_plus_all_experimental",
)

ms = m_reported[m_reported["config"].isin(NONARM_IS_CONFIGS)].dropna(
    subset=["spearman_rho"]
)

# Best (config, model) per label by Spearman rho
best = (
    ms.sort_values("spearman_rho", ascending=False)
    .groupby("label")
    .head(1)
    .sort_values("spearman_rho", ascending=False)
)

# ── Collect top-8 features per best combination ──────────────────────────
rows = []
for _, brow in best.iterrows():
    sl = imp_reported[
        (imp_reported["label"] == brow["label"])
        & (imp_reported["config"] == brow["config"])
        & (imp_reported["model"] == brow["model"])
    ]
    sl = sl.sort_values("abs_perm", ascending=False).head(8)
    for rank, (_, frow) in enumerate(sl.iterrows(), 1):
        rows.append(
            {
                "label": brow["label"],
                "config": brow["config"],
                "model": brow["model"],
                "spearman_rho": brow["spearman_rho"],
                "feature": frow["feature"],
                "perm_importance_mean": frow["perm_importance_mean"],
                "perm_importance_std": frow["perm_importance_std"],
                "rank": rank,
            }
        )

out = pd.DataFrame(rows)
out_path = OUT_DIR / "supp_fig7_top_features.csv"
out.to_csv(out_path, index=False)
print(f"wrote {out_path} ({len(out)} rows)")
