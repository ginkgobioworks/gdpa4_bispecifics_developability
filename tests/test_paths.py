"""Input layout the README describes."""
from pathlib import Path

from prophet_ab import paths


def test_gdpa4_tall_table_exists_and_is_named_gdpa4():
    assert paths.RAW_N3N4_TALL_CSV.name == "GDPa4_N3_N4_Summary_tall.csv"
    assert paths.RAW_N3N4_TALL_CSV.is_file()


def test_moe_and_umap_inputs_exist():
    assert (paths.RAW_IN_SILICO_DIR / "MOE_properties.csv").is_file()
    assert paths.RAW_UMAP_COORDS.is_file()


def test_committed_model_caches_exist():
    for name in (
        "cv_metrics.parquet",
        "cv_metrics_loo.parquet",
        "cv_oof_predictions_loo.parquet",
        "feature_importance_long.parquet",
    ):
        assert (paths.S05 / name).is_file(), name


def test_missing_gdpa4_table_is_a_file_not_found(tmp_path, monkeypatch):
    missing = tmp_path / "GDPa4_N3_N4_Summary_tall.csv"
    monkeypatch.setattr(paths, "RAW_N3N4_TALL_CSV", missing)
    from prophet_ab import io
    import pandas as pd

    try:
        io.read_n3n4_tall()
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
    except pd.errors.EmptyDataError:
        raise
    except Exception as exc:  # pandas may wrap the missing path
        assert "GDPa4_N3_N4_Summary_tall.csv" in str(exc) or isinstance(
            exc, FileNotFoundError
        )
