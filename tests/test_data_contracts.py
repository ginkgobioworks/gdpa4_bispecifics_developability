"""Minimal publication-data contracts.

Rendering every standalone figure is the integration test. These assertions
only protect inputs that would be costly or impossible to reconstruct.
"""
from __future__ import annotations

import pandas as pd

from prophet_ab import io, normalize, paths, schema


def test_design_and_assayed_cohort_sizes() -> None:
    design = pd.read_csv(
        paths.RAW
        / "bsab_design"
        / "bsabs_dec_2025"
        / "data"
        / "exported_bsabs.csv"
    )
    assert len(design) == 160
    parents = set(design["antibody_name-1"]) | set(design["antibody_name-2"])
    assert len(parents) == 65

    assayed = normalize.annotate(io.read_gdpa4_tall())
    counts = (
        assayed[["antibody_name", "kind"]]
        .drop_duplicates()
        .groupby("kind")
        .size()
        .to_dict()
    )
    assert counts[schema.KIND_BISPECIFIC] == 160
    assert counts[schema.KIND_MONOSPECIFIC] == 71


def test_preserved_model_records_are_readable() -> None:
    expected = {
        "cv_metrics.parquet",
        "cv_metrics_loo.parquet",
        "cv_oof_predictions_loo.parquet",
        "feature_importance_long.parquet",
    }
    for filename in expected:
        artifact = paths.S05 / filename
        assert artifact.is_file(), filename
        assert not pd.read_parquet(artifact).empty, filename
