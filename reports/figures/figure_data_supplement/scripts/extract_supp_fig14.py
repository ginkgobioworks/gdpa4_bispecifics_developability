"""Extract source data for Supplemental Figure 14 (brazikumab / ligelizumab)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from prophet_ab.features.parental_surface import (
    load_example_parents,
    load_surface_long,
    source_scatter_table,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    surface = load_surface_long()
    parents = load_example_parents()
    table = source_scatter_table(surface, parents)
    out_path = OUT_DIR / "supp_fig14_surface_scatter.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote {len(table)} rows to {out_path}")


if __name__ == "__main__":
    main()
