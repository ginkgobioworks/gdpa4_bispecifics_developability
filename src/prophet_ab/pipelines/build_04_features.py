"""Stage 04 — feature tables.

Outputs (data/processed/04_features/):
    in_silico_per_monospecific_long.parquet
        Per-monospecific in-silico features in long form.
        Cols: monospecific_antibody_name, parent (stripped), source, feature, value
        One row per (monospecific antibody × in-silico feature). NaN values dropped.

    in_silico_per_bispecific_long.parquet
        Per-bispecific in-silico features computed by applying every operator in
        compositional.OPERATORS to the two parents' values.
        Cols: antibody_name, parent_a, parent_b, source, feature, operator, value

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


def _build_per_monospecific(in_silico_long: pd.DataFrame, monospecific_map: pd.DataFrame) -> pd.DataFrame:
    # in_silico_long.antibody_name uses the bare GDPa1 name; join via monospecific_stripped.
    merged = in_silico_long.merge(
        monospecific_map[["monospecific_antibody_name", "monospecific_stripped"]],
        left_on="antibody_name", right_on="monospecific_stripped", how="inner",
    )
    out = merged[["monospecific_antibody_name", "monospecific_stripped", "source", "feature", "value"]].rename(
        columns={"monospecific_stripped": "parent"}
    )
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _build_per_bispecific(per_monospecific: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    # Per-(parent, source, feature) lookup → join twice for parent_a and parent_b.
    lookup = per_monospecific[["parent", "source", "feature", "value"]]
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

    print("[1/2] in_silico_per_monospecific_long")
    insilico_long = in_silico.load_all_long().dropna(subset=["value"])
    monospecific_map = pd.read_parquet(paths.S02 / "monospecific_gdpa1_map.parquet")
    per_monospecific = _build_per_monospecific(insilico_long, monospecific_map)
    _write(per_monospecific, out / "in_silico_per_monospecific_long.parquet")

    print("[2/2] in_silico_per_bispecific_long")
    components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")
    per_bispecific = _build_per_bispecific(per_monospecific, components)
    _write(per_bispecific, out / "in_silico_per_bispecific_long.parquet")

    print("done.")


if __name__ == "__main__":
    main()
