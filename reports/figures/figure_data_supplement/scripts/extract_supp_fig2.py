"""Extract source data for Supplemental Figure 2 (revised, ungated S2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd

from prophet_ab.features.charge import horserace, load_acsins_pbs_pairs

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _signed_gmean(q1: float, q2: float) -> float:
    return float(-np.sign(q1 * q2) * np.sqrt(np.abs(q1 * q2)))


def main() -> None:
    pairs = load_acsins_pbs_pairs().dropna(subset=["q1", "q2"]).copy()
    pairs["category"] = pairs["category_ungated"]

    table = horserace(pairs, "category")
    panel_a = table.rename(columns={"side_label": "side"})[
        ["transform", "side", "coef", "wald_p", "aic", "delta_aic", "auc"]
    ]
    out_a = OUT_DIR / "supp_fig2a_delta_aic.csv"
    panel_a.to_csv(out_a, index=False)
    print(f"Panel A: wrote {len(panel_a)} rows to {out_a}")

    enh = pairs[pairs["category"] == "Enhancer"]
    sup = pairs[pairs["category"] == "Suppressor"]
    rest = pairs[pairs["category"] == ""]

    panel_b_rows = []
    for frame, label in ((enh, "Enhancer"), (rest, "Not classified")):
        for _, row in frame.iterrows():
            panel_b_rows.append(
                {
                    "pair": f"{row['parent_a']}__{row['parent_b']}",
                    "abs_sum_charge": abs(row["q1"] + row["q2"]),
                    "category": label,
                }
            )
    panel_b = pd.DataFrame(panel_b_rows)
    out_b = OUT_DIR / "supp_fig2b_enhancer_violin.csv"
    panel_b.to_csv(out_b, index=False)
    print(f"Panel B: wrote {len(panel_b)} rows to {out_b}")

    panel_c_rows = []
    for frame, label in ((sup, "Suppressor"), (rest, "Not classified")):
        for _, row in frame.iterrows():
            panel_c_rows.append(
                {
                    "pair": f"{row['parent_a']}__{row['parent_b']}",
                    "signed_geomean_charge": _signed_gmean(row["q1"], row["q2"]),
                    "category": label,
                }
            )
    panel_c = pd.DataFrame(panel_c_rows)
    out_c = OUT_DIR / "supp_fig2c_suppressor_violin.csv"
    panel_c.to_csv(out_c, index=False)
    print(f"Panel C: wrote {len(panel_c)} rows to {out_c}")


if __name__ == "__main__":
    main()
