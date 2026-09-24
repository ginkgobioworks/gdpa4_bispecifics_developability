"""Extract source data for Supplemental Figure 5."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import pandas as pd

from prophet_ab import paths, schema

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    per_ab = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")

    # Filter to Tm1 and Tm2 (PTS-IF, pooled), exclude deprecated
    tm_cols = {"thermostability_tm1", "thermostability_tm2"}
    pts = per_ab[
        per_ab["value_col"].isin(tm_cols)
        & ~per_ab["value_col"].isin(schema.DEPRECATED_VALUE_COLS)
    ].copy()

    result = pts[["antibody_name", "kind", "value_col", "condition", "median"]].dropna(
        subset=["median"]
    )

    out_path = OUT_DIR / "supp_fig5_violin_tm.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
