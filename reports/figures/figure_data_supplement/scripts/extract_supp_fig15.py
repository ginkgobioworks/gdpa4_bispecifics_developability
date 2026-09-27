"""Extract source data for Supplemental Figure 15 (production and purity tiers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from prophet_ab.features.production_qc import (
    load_production_pairs,
    source_scatter_table,
    tier_assignments,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pairs = load_production_pairs()
    table = source_scatter_table(pairs, tier_assignments(pairs))
    out_path = OUT_DIR / "supp_fig15_production_scatter.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table)} rows to {out_path}")


if __name__ == "__main__":
    main()
