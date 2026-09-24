"""Stage 03 — per-antibody summary statistics.

Outputs (data/processed/03_aggregated/):
    n3n4_per_antibody.parquet   — per-antibody summaries from this campaign
                                   (one row per antibody × value_col × condition).
    gdpa1_per_antibody.parquet  — per-antibody summaries from the GDPa1 wide
                                   tidy table, melted to long form to match
                                   our schema.

Run: `python -m prophet_ab.pipelines.build_03_aggregated`
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import aggregate, paths


# Columns in GDPa1 tidy that are measurement values (not metadata).
# Anything else is an identity / replicate key.
GDPA1_VALUE_COLS: tuple[str, ...] = (
    "titer", "normalized_titer", "purity_%lc+hc",
    "tonset_nanodsf", "tm1_nanodsf", "tm2_nanodsf", "tm3_nanodsf",
    "sec_%monomer", "smac_rt", "hic_rt", "hac_rt",
    "acsins_dLmax_ph7.4", "acsins_dLmax_ph6.0",
    "polyreactivity_prscore_cho", "polyreactivity_prscore_ova",
    "dlskd_ph7.4", "dlskd_ph6.0",
)


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def _melt_gdpa1(tidy: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["antibody_id", "antibody_name", "hc_subtype", "lc_subtype"]
    value_cols = [c for c in GDPA1_VALUE_COLS if c in tidy.columns]
    long = tidy.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="value_col",
        value_name="value",
    ).dropna(subset=["value"])
    long["condition"] = "default"  # GDPa1 tidy is one-condition-per-column
    return long


def main() -> None:
    out = paths.S03

    print("[1/2] n3n4_per_antibody")
    long = pd.read_parquet(paths.S01 / "n3n4_long.parquet")
    _write(aggregate.per_antibody_long(long), out / "n3n4_per_antibody.parquet")

    print("[2/2] gdpa1_per_antibody")
    tidy = pd.read_parquet(paths.S01 / "gdpa1_tidy.parquet")
    long_g = _melt_gdpa1(tidy)
    summary = aggregate.per_antibody_long(
        long_g,
        extra_keys=("antibody_id", "hc_subtype", "lc_subtype"),
    )
    _write(summary, out / "gdpa1_per_antibody.parquet")

    print("done.")


if __name__ == "__main__":
    main()
