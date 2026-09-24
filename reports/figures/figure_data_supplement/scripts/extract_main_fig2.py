"""Extract source data for Figure 2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd

from prophet_ab import paths, schema
from prophet_ab.normalize import parse_n3_components

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"


def main():
    # ---- Load per-antibody summaries (stage 03) ----------------------------
    per_ab = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")

    # Filter to assay data (exclude deprecated + production)
    assay_data = per_ab[
        ~per_ab["value_col"].isin(schema.DEPRECATED_VALUE_COLS)
        & ~per_ab["value_col"].isin(schema.PRODUCTION_VALUE_COLS)
    ].copy()

    # ---- Restrict to N3 + reported value_cols ------------------------------
    n3 = assay_data[
        (assay_data["kind"] == "N3")
        & assay_data["value_col"].isin(schema.REPORTED_VALUE_COLS)
    ].copy()

    # Parse N3 components -> sorted pair key
    comps = n3["antibody_name"].map(parse_n3_components)
    n3["pair"] = comps.map(lambda t: tuple(sorted(t)))

    # Keep only pairs with exactly 2 orientations
    orient_counts = (
        n3.groupby(["pair", "value_col", "condition"])["antibody_name"]
        .nunique()
        .reset_index(name="_n_orient")
    )
    swap_pair_keys = frozenset(
        orient_counts.loc[orient_counts["_n_orient"] == 2, "pair"]
    )
    n3 = n3[n3["pair"].isin(swap_pair_keys)]

    # ---- Build swap pair table: sort medians into lo/hi --------------------
    rows = []
    for (pair, vc, cond), grp in n3.groupby(["pair", "value_col", "condition"]):
        if grp["antibody_name"].nunique() != 2:
            continue
        sorted_grp = grp.sort_values("median")
        r_lo, r_hi = sorted_grp.iloc[0], sorted_grp.iloc[1]
        rows.append({
            "value_col": vc,
            "condition": cond,
            "pair": f"{pair[0]}__{pair[1]}",
            "lo": float(r_lo["median"]),
            "hi": float(r_hi["median"]),
        })

    df_swap = pd.DataFrame(rows)

    out_path = OUT_DIR / "fig2_swap_pairs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_swap.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df_swap)} rows)")


if __name__ == "__main__":
    main()
