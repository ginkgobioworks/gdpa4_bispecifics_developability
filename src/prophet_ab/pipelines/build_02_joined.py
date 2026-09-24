"""Stage 02 — pure joins and lookup tables (no aggregation, no statistics).

Outputs (data/processed/02_joined/):
    n3_components.parquet    — one row per N3, with parent_a/parent_b
                                and a flag for whether both parents are in N4.
    n4_gdpa1_map.parquet     — one row per N4 antibody, with the matched
                                GDPa1 antibody_id (or NaN), suffix-stripped.
    n3_chains_long.parquet   — N3 chains pivoted to long form
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


def _build_n3_components(long: pd.DataFrame, n4_names: set[str]) -> pd.DataFrame:
    n3 = long[long["kind"] == schema.KIND_N3][["antibody_name", "parent_a", "parent_b"]].drop_duplicates()
    n3["parent_a_in_n4"] = n3["parent_a"].isin(n4_names)
    n3["parent_b_in_n4"] = n3["parent_b"].isin(n4_names)
    n3["both_parents_in_n4"] = n3["parent_a_in_n4"] & n3["parent_b_in_n4"]
    return n3.sort_values("antibody_name").reset_index(drop=True)


def _build_n4_gdpa1_map(n4_names: list[str], gdpa1_seq: pd.DataFrame) -> pd.DataFrame:
    # Strip isotype suffix on N4 to match GDPa1 antibody_name (which is bare).
    rows = []
    gdpa1_lookup = gdpa1_seq.set_index("antibody_name")["antibody_id"].to_dict()
    for name in n4_names:
        stripped = normalize.strip_isotype_suffix(name)
        rows.append({
            "n4_antibody_name": name,
            "n4_stripped": stripped,
            "gdpa1_antibody_id": gdpa1_lookup.get(stripped),
            "in_gdpa1": stripped in gdpa1_lookup,
        })
    return pd.DataFrame(rows).sort_values("n4_antibody_name").reset_index(drop=True)


def _build_n3_chains_long(n3_prod: pd.DataFrame) -> pd.DataFrame:
    # n3_production has wide columns seq{1..4}_name and sequence_{1..4}.
    parts = []
    for i in (1, 2, 3, 4):
        block = n3_prod[["antibody_name", f"seq{i}_name", f"sequence_{i}"]].copy()
        block.columns = ["antibody_name", "chain_role", "chain_sequence"]
        block["chain_index"] = i
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    out["chain_sequence"] = out["chain_sequence"].str.rstrip("*")
    return out.sort_values(["antibody_name", "chain_index"]).reset_index(drop=True)


def main() -> None:
    out = paths.S02

    long = pd.read_parquet(paths.S01 / "n3n4_long.parquet")
    n4_long = long[long["kind"] == schema.KIND_N4]
    n4_names = sorted(n4_long["antibody_name"].unique())
    # Compare parents (already suffix-stripped by `parse_n3_components`) to
    # suffix-stripped N4 names. Per D-2026-04-27-ISOTYPE.
    n4_stripped_set = {normalize.strip_isotype_suffix(n) for n in n4_names}

    print("[1/3] n3_components")
    _write(_build_n3_components(long, n4_stripped_set), out / "n3_components.parquet")

    print("[2/3] n4_gdpa1_map")
    gdpa1_seq = pd.read_parquet(paths.S01 / "gdpa1_sequences.parquet")
    _write(_build_n4_gdpa1_map(n4_names, gdpa1_seq), out / "n4_gdpa1_map.parquet")

    print("[3/3] n3_chains_long")
    n3_prod = pd.read_parquet(paths.S01 / "n3_production.parquet")
    _write(_build_n3_chains_long(n3_prod), out / "n3_chains_long.parquet")

    print("done.")


if __name__ == "__main__":
    main()
