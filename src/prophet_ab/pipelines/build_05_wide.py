"""Stage 05a — wide feature and label matrices.

Outputs (data/processed/05_modeling/):
    n3_features_wide.parquet           — combined wide feature matrix per N3
                                          (measured + in-silico, all operators).
    n3_labels_wide.parquet             — per-N3 measured medians (prediction targets).

Run: `python -m prophet_ab.pipelines.build_05_wide`
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import paths
from ..features import wide


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def main() -> None:
    out = paths.S05

    print("[1/1] building wide feature matrix")
    per_antibody = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")
    is_long = pd.read_parquet(paths.S04 / "in_silico_per_n3_long.parquet")
    is_n4_long = pd.read_parquet(paths.S04 / "in_silico_per_n4_long.parquet")

    meas_w = wide.build_measured_wide(per_antibody, components)
    is_w = wide.build_in_silico_wide(is_long)
    arm_meas_w = wide.build_arm_measured_wide(per_antibody, components)
    arm_is_w = wide.build_arm_is_wide(is_n4_long, components)
    labels_w = wide.build_labels_wide(per_antibody)

    feats = meas_w.join(is_w, how="outer").join(arm_meas_w, how="outer").join(arm_is_w, how="outer")
    print(f"  features: {feats.shape}, labels: {labels_w.shape}")

    _write(feats.reset_index(), out / "n3_features_wide.parquet")
    _write(labels_w.reset_index(), out / "n3_labels_wide.parquet")

    print("done.")


if __name__ == "__main__":
    main()
