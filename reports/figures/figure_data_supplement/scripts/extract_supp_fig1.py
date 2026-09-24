"""Extract source data for Supplemental Figure 1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from prophet_ab import paths, schema
from prophet_ab import normalize as nz

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    n3n4 = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    _gdpa1_all = pd.read_parquet(paths.S03 / "gdpa1_per_antibody.parquet")
    gdpa1 = _gdpa1_all[_gdpa1_all["hc_subtype"] == "IgG1"]
    n4_map = pd.read_parquet(paths.S02 / "n4_gdpa1_map.parquet")

    # Filter to N4, attach stripped name for cross-platform join
    n4 = n3n4[n3n4["kind"] == schema.KIND_N4].merge(
        n4_map[["n4_antibody_name", "n4_stripped"]],
        left_on="antibody_name",
        right_on="n4_antibody_name",
        how="left",
    )

    # Build display label helper (mirrors notebook logic)
    def _panel_label(vc: str, cond: str) -> str:
        name = schema.VALUE_COL_DISPLAY.get(vc, vc)
        cond_disp = schema.CONDITION_DISPLAY.get(cond, cond)
        if cond_disp in (None, "", "default"):
            return name
        return f"{name} @ {cond_disp}"

    all_rows: list[pd.DataFrame] = []

    for (this_vc, this_cond), gdpa1_col in schema.PROPHET_TO_GDPA1.items():
        ours = n4[(n4["value_col"] == this_vc) & (n4["condition"] == this_cond)][
            ["n4_stripped", "median"]
        ].rename(columns={"median": "this_campaign_median"})

        theirs = gdpa1[gdpa1["value_col"] == gdpa1_col][
            ["antibody_name", "median"]
        ].rename(
            columns={"antibody_name": "n4_stripped", "median": "gdpa1_median"}
        )

        merged = ours.merge(theirs, on="n4_stripped", how="inner").dropna(
            subset=["this_campaign_median", "gdpa1_median"]
        )

        if len(merged) < 3:
            continue

        panel = _panel_label(this_vc, this_cond)

        # Z-score for AC-SINS deltaLmax panels
        if this_vc == "acsins_delta_Lmax":
            for col, zcol in [
                ("this_campaign_median", "this_campaign_zscore"),
                ("gdpa1_median", "gdpa1_zscore"),
            ]:
                vals = merged[col]
                merged[zcol] = (vals - vals.mean()) / vals.std(ddof=0)
        else:
            merged["this_campaign_zscore"] = np.nan
            merged["gdpa1_zscore"] = np.nan

        # Pearson r on raw medians
        r, _ = stats.pearsonr(
            merged["this_campaign_median"], merged["gdpa1_median"]
        )
        n = len(merged)

        merged["antibody_name"] = merged["n4_stripped"]
        merged["value_col"] = this_vc
        merged["condition"] = this_cond
        merged["panel_label"] = panel
        merged["pearson_r"] = r
        merged["n"] = n

        all_rows.append(
            merged[
                [
                    "antibody_name",
                    "value_col",
                    "condition",
                    "panel_label",
                    "this_campaign_median",
                    "gdpa1_median",
                    "this_campaign_zscore",
                    "gdpa1_zscore",
                    "pearson_r",
                    "n",
                ]
            ]
        )

    result = pd.concat(all_rows, ignore_index=True)
    out_path = OUT_DIR / "supp_fig1_cross_platform.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
