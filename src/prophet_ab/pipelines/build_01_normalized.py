"""Stage 01 — read raw inputs, normalize names, write parquets.

Outputs (data/processed/01_normalized/):
    gdpa4_long.parquet         — primary tall measurement table, annotated
    gdpa1_tidy.parquet        — GDPa1 tidy assay table
    gdpa1_sequences.parquet   — GDPa1 sequences sheet
    gdpa1_prior_lit.parquet   — Jain/Makowski/Shehata/Kraft published values
    bispecific_production.parquet     — bispecific sequences + vendor QC
    monospecific_production.parquet     — monospecific yields

Run: `python -m prophet_ab.pipelines.build_01_normalized`
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import in_silico, io, normalize, paths


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def main() -> None:
    out = paths.S01

    print("[1/6] gdpa4_long")
    df = normalize.annotate(io.read_gdpa4_tall())
    n_na = df["value"].isna().sum()
    if n_na:
        print(f"  dropping {n_na} rows with NaN value")
        df = df.dropna(subset=["value"])
    _write(df, out / "gdpa4_long.parquet")

    print("[2/6] gdpa1_tidy")
    _write(io.read_gdpa1_tidy(), out / "gdpa1_tidy.parquet")

    print("[3/6] gdpa1_sequences")
    _write(io.read_gdpa1_sequences(), out / "gdpa1_sequences.parquet")

    print("[4/6] gdpa1_prior_lit")
    _write(io.read_gdpa1_prior_lit(), out / "gdpa1_prior_lit.parquet")

    print("[5/6] bispecific_production")
    production = normalize.annotate(io.read_bispecific_production())
    _write(production, out / "bispecific_production.parquet")

    print("[6/6] monospecific_production")
    n4p = normalize.annotate(io.read_monospecific_production())
    _write(n4p, out / "monospecific_production.parquet")

    print("[in-silico] one parquet per source")
    for src in in_silico.SOURCES:
        df = in_silico.read_source(src)
        _write(df, out / "in_silico" / f"{src.name}.parquet")

    print("done.")


if __name__ == "__main__":
    main()
