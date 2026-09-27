"""Extract source data for Supplemental Figure 16 (orientation A-B vs B-A)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from prophet_ab.features.orientation import (
    load_orientation_pairs,
    orientation_correlations,
    source_pair_table,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pairs = load_orientation_pairs()
    table = source_pair_table(pairs, orientation_correlations(pairs))
    out_path = OUT_DIR / "supp_fig16_orientation_pairs.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table)} rows to {out_path}")


if __name__ == "__main__":
    main()
