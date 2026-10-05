"""Stage 02 — pure joins and lookup tables (no aggregation, no statistics).

Outputs (data/processed/02_joined/):
    bispecific_components.parquet    — one row per bispecific, with parent_a/parent_b
                                and a flag for whether both parents are in monospecific.
    monospecific_gdpa1_map.parquet     — one row per monospecific antibody, with the matched
                                GDPa1 antibody_id (or NaN), suffix-stripped.
    bispecific_chains_long.parquet   — bispecific chains pivoted to long form
                                (one row per chain, columns: antibody_name,
                                 chain_role, chain_name, chain_sequence).

Run: `python -m prophet_ab.pipelines.build_02_joined`
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import io, normalize, paths, schema


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def _build_bispecific_components(long: pd.DataFrame, monospecific_names: set[str]) -> pd.DataFrame:
    bispecific = long[long["kind"] == schema.KIND_BISPECIFIC][["antibody_name", "parent_a", "parent_b"]].drop_duplicates()
    bispecific["parent_a_in_monospecific"] = bispecific["parent_a"].isin(monospecific_names)
    bispecific["parent_b_in_monospecific"] = bispecific["parent_b"].isin(monospecific_names)
    bispecific["both_parents_in_monospecific"] = bispecific["parent_a_in_monospecific"] & bispecific["parent_b_in_monospecific"]
    return bispecific.sort_values("antibody_name").reset_index(drop=True)


def _build_monospecific_gdpa1_map(monospecific_names: list[str], gdpa1_seq: pd.DataFrame) -> pd.DataFrame:
    # Strip isotype suffix on monospecific to match GDPa1 antibody_name (which is bare).
    rows = []
    gdpa1_lookup = gdpa1_seq.set_index("antibody_name")["antibody_id"].to_dict()
    for name in monospecific_names:
        stripped = normalize.strip_isotype_suffix(name)
        rows.append({
            "monospecific_antibody_name": name,
            "monospecific_stripped": stripped,
            "gdpa1_antibody_id": gdpa1_lookup.get(stripped),
            "in_gdpa1": stripped in gdpa1_lookup,
        })
    return pd.DataFrame(rows).sort_values("monospecific_antibody_name").reset_index(drop=True)


def _build_bispecific_chains_long(bispecific_prod: pd.DataFrame) -> pd.DataFrame:
    # bispecific_production has wide columns seq{1..4}_name and sequence_{1..4}.
    parts = []
    for i in (1, 2, 3, 4):
        block = bispecific_prod[["antibody_name", f"seq{i}_name", f"sequence_{i}"]].copy()
        block.columns = ["antibody_name", "chain_role", "chain_sequence"]
        block["chain_index"] = i
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    out["chain_sequence"] = out["chain_sequence"].str.rstrip("*")
    return out.sort_values(["antibody_name", "chain_index"]).reset_index(drop=True)


def main() -> None:
    out = paths.S02

    long = pd.read_parquet(paths.S01 / "gdpa4_long.parquet")
    monospecific_long = long[long["kind"] == schema.KIND_MONOSPECIFIC]
    monospecific_names = sorted(monospecific_long["antibody_name"].unique())
    # Compare parents, already suffix-stripped, with suffix-stripped monospecific names.
    monospecific_stripped_set = {normalize.strip_isotype_suffix(n) for n in monospecific_names}

    print("[1/3] bispecific_components")
    _write(_build_bispecific_components(long, monospecific_stripped_set), out / "bispecific_components.parquet")

    print("[2/3] monospecific_gdpa1_map")
    gdpa1_seq = pd.read_parquet(paths.S01 / "gdpa1_sequences.parquet")
    _write(_build_monospecific_gdpa1_map(monospecific_names, gdpa1_seq), out / "monospecific_gdpa1_map.parquet")

    print("[3/3] bispecific_chains_long")
    bispecific_prod = pd.read_parquet(paths.S01 / "bispecific_production.parquet")
    _write(_build_bispecific_chains_long(bispecific_prod), out / "bispecific_chains_long.parquet")

    print("done.")


if __name__ == "__main__":
    main()
