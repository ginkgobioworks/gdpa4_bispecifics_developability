"""Extract source data for Figure 5."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd

from prophet_ab import paths, schema

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load per-antibody summaries and pivot to wide
# ---------------------------------------------------------------------------
per_ab = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")

metrics = [
    ("hihplc_normretentiontime", "default", "hic"),
    ("hachplc_retentiontime", "default", "hac"),
    ("pr_score", "CHO", "pr_cho"),
    ("pr_score", "Ovalbumin", "pr_ova"),
    ("bvp_score_norm", "default", "bvp"),
]

frames = []
for vc, cond, alias in metrics:
    sub = (
        per_ab[(per_ab["value_col"] == vc) & (per_ab["condition"] == cond)]
        [["antibody_name", "kind", "is_control", "median"]]
        .rename(columns={"median": alias})
    )
    frames.append((alias, sub))

wide = frames[0][1]
for alias, f in frames[1:]:
    wide = wide.merge(f[["antibody_name", alias]], on="antibody_name", how="outer")

# ---------------------------------------------------------------------------
# Save scatter data (all antibodies, split by kind)
# ---------------------------------------------------------------------------
out_scatter = OUT_DIR / "fig5_scatter.csv"
wide[["antibody_name", "kind", "hic", "hac", "pr_cho", "pr_ova", "bvp"]].to_csv(
    out_scatter, index=False
)
print(f"Scatter: {len(wide)} antibodies -> {out_scatter}")

# ---------------------------------------------------------------------------
# Pareto frontier on monospecific HIC x HAC (maximize both)
# ---------------------------------------------------------------------------
monospecific = (
    wide[wide["kind"] == schema.KIND_MONOSPECIFIC]
    .dropna(subset=["hic", "hac"])
    .copy()
    .reset_index(drop=True)
)
pts = monospecific[["hic", "hac"]].values
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

fpts = pts[is_pareto]
frontier_pts = fpts[np.argsort(fpts[:, 0])]

out_pareto = OUT_DIR / "fig5_pareto_frontier.csv"
pd.DataFrame(frontier_pts, columns=["hic", "hac"]).to_csv(out_pareto, index=False)
print(f"Pareto frontier: {len(frontier_pts)} points -> {out_pareto}")
