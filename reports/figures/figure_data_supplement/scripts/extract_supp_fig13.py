"""Extract source data for Supplemental Figure 13 (CrossMab thermal penalty)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from prophet_ab.features.thermal import load_crossmab_deltas

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    deltas = load_crossmab_deltas()
    out_path = OUT_DIR / "supp_fig13_crossmab_deltas.csv"
    deltas.to_csv(out_path, index=False)
    print(f"Wrote {len(deltas)} rows to {out_path}")


if __name__ == "__main__":
    main()
