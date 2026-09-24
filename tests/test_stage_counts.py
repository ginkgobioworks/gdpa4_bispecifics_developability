"""Smoke checks on processed parquets produced by stages 01–03."""
import pandas as pd
import pytest

from prophet_ab import paths, schema


def test_n3_n4_counts():
    p = paths.S02 / "n3_components.parquet"
    if not p.is_file():
        pytest.skip("run `make stages` first")
    components = pd.read_parquet(p)
    assert components["antibody_name"].nunique() == 160

    long = pd.read_parquet(paths.S01 / "n3n4_long.parquet")
    n4 = long[long["kind"] == schema.KIND_N4]["antibody_name"].nunique()
    n3 = long[long["kind"] == schema.KIND_N3]["antibody_name"].nunique()
    assert n3 == 160
    assert n4 == 71
