"""Extract source data for Figure 6."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import pandas as pd

from prophet_ab import paths, schema
from prophet_ab.features.naming import display_config, display_label, parse_label_name

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load LOO metrics; filter deprecated and purity labels, drop NaN rho
# ---------------------------------------------------------------------------
DROP_VALUE_COLS = set(schema.DEPRECATED_VALUE_COLS) | {"purity_pct"}


def keep_label(label_col: str) -> bool:
    vc, _ = parse_label_name(label_col)
    return vc not in DROP_VALUE_COLS


m = pd.read_parquet(paths.S05 / "cv_metrics_loo.parquet")
m = m[m["label"].map(keep_label)].copy()
m = m.dropna(subset=["spearman_rho"])

# ---------------------------------------------------------------------------
# Map config and model to display names
# ---------------------------------------------------------------------------
m["config_display"] = m["config"].map(display_config)
m["model_upper"] = m["model"].str.upper()

out = OUT_DIR / "fig6b_loo_boxenplot.csv"
m[["model_upper", "config", "config_display", "label", "spearman_rho"]].rename(
    columns={"model_upper": "model"}
).to_csv(out, index=False)
print(f"LOO boxenplot: {len(m)} rows -> {out}")
