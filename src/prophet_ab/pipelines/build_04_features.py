"""Stage 04 — feature tables.

Outputs (data/processed/04_features/):
    in_silico_per_n4_long.parquet
        Per-N4 in-silico features in long form.
        Cols: n4_antibody_name, parent (stripped), source, feature, value
        One row per (N4 antibody × in-silico feature). NaN values dropped.

    in_silico_per_n3_long.parquet
        Per-N3 in-silico features computed by applying every operator in
        compositional.OPERATORS to the two parents' values.
        Cols: antibody_name, parent_a, parent_b, source, feature, operator, value

Note: stage 04 also receives `n3_compositional_predictions.parquet` from the
s03 notebook (compositional baseline labels + predictions).

Run: `python -m prophet_ab.pipelines.build_04_features`
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import in_silico, paths
from ..features import compositional


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def _build_per_n4(in_silico_long: pd.DataFrame, n4_map: pd.DataFrame) -> pd.DataFrame:
    # in_silico_long.antibody_name uses the bare GDPa1 name; join via n4_stripped.
    merged = in_silico_long.merge(
        n4_map[["n4_antibody_name", "n4_stripped"]],
        left_on="antibody_name", right_on="n4_stripped", how="inner",
    )
    out = merged[["n4_antibody_name", "n4_stripped", "source", "feature", "value"]].rename(
        columns={"n4_stripped": "parent"}
    )
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _build_per_n3(per_n4: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    # Per-(parent, source, feature) lookup → join twice for parent_a and parent_b.
    lookup = per_n4[["parent", "source", "feature", "value"]]
    base = components[["antibody_name", "parent_a", "parent_b"]]

    merged = (
        base.merge(lookup.rename(columns={"parent": "parent_a", "value": "a"}),
                   on="parent_a", how="left")
            .merge(lookup.rename(columns={"parent": "parent_b", "value": "b"}),
                   on=["parent_b", "source", "feature"], how="left")
    )
    # apply each operator
    parts = []
    for op_name, op_fn in compositional.OPERATORS.items():
        v = op_fn(merged["a"], merged["b"])
        parts.append(merged.assign(operator=op_name, value=v)[
            ["antibody_name", "parent_a", "parent_b", "source", "feature", "operator", "value"]
        ])
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["value"]).reset_index(drop=True)


def main() -> None:
    out = paths.S04

    print("[1/2] in_silico_per_n4_long")
    insilico_long = in_silico.load_all_long().dropna(subset=["value"])
    n4_map = pd.read_parquet(paths.S02 / "n4_gdpa1_map.parquet")
    per_n4 = _build_per_n4(insilico_long, n4_map)
    _write(per_n4, out / "in_silico_per_n4_long.parquet")

    print("[2/2] in_silico_per_n3_long")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")
    per_n3 = _build_per_n3(per_n4, components)
    _write(per_n3, out / "in_silico_per_n3_long.parquet")

    print("done.")


if __name__ == "__main__":
    main()
