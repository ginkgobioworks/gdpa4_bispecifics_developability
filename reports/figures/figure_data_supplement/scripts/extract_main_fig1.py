"""Extract source data for Figure 1."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import pandas as pd

from prophet_ab import paths

DESIGN_DATA = paths.RAW / "bsab_design" / "bsabs_dec_2025" / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_exported = pd.read_csv(DESIGN_DATA / "exported_bsabs.csv")
    print(f"Exported bsAbs: {len(df_exported)}")

    out_1b = OUT_DIR / "fig1b_umap.csv"
    shutil.copyfile(paths.RAW_UMAP_COORDS, out_1b)
    n_1b = sum(1 for _ in open(out_1b)) - 1
    print(f"Wrote {out_1b}  ({n_1b} rows, frozen UMAP coordinates)")

    all_arms = pd.concat(
        [df_exported["antibody_name-1"], df_exported["antibody_name-2"]]
    )
    counts = all_arms.value_counts().sort_values(ascending=False)
    df_1c = pd.DataFrame({
        "parent_mab": counts.index,
        "count": counts.values,
    })
    out_1c = OUT_DIR / "fig1c_arm_distribution.csv"
    df_1c.to_csv(out_1c, index=False)
    print(f"Wrote {out_1c}  ({len(df_1c)} rows)")


if __name__ == "__main__":
    main()
