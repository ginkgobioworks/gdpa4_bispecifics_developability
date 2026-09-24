"""Compare rebuilt source-data extracts and key tables to committed goldens."""
from pathlib import Path

import pandas as pd
import pytest

from prophet_ab import paths

GOLDEN = Path(__file__).resolve().parent / "golden"
EXTRACTED = paths.FIGURES / "figure_data_supplement" / "extracted"

CSV_PAIRS = [
    ("fig1c_arm_distribution.csv", ["parent_mab", "count"]),
    ("fig2_swap_pairs.csv", None),
    ("fig3a_scatter.csv", None),
    ("fig3b_heatmap.csv", None),
    ("fig4a_acsins_classification.csv", None),
    ("fig4b_charge_quadrant.csv", None),
    ("fig6b_loo_boxenplot.csv", None),
    ("supp_fig1_cross_platform.csv", None),
    ("supp_fig6_top_features.csv", None),
    ("supp_fig7_top_features_arm.csv", None),
    ("supp_fig8_coefficients.csv", None),
]


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.skip(f"missing {path}; run `make all` first")
    return path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(_require(path))


@pytest.mark.parametrize("name,sort_cols", CSV_PAIRS)
def test_extracted_matches_golden(name, sort_cols):
    got = _read(EXTRACTED / name)
    exp = _read(GOLDEN / name)
    if sort_cols:
        got = got.sort_values(sort_cols).reset_index(drop=True)
        exp = exp.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        exp.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-5,
        atol=1e-5,
    )


def test_umap_coords_match_frozen_table():
    frozen = _read(paths.RAW_UMAP_COORDS)
    extracted = _read(EXTRACTED / "fig1b_umap.csv")
    pd.testing.assert_frame_equal(frozen, extracted, check_dtype=False)


def test_s02_correlations_table():
    got = _read(paths.TABLES / "s02_cross_platform_correlations.csv")
    exp = _read(GOLDEN / "s02_cross_platform_correlations.csv")
    pd.testing.assert_frame_equal(got, exp, check_dtype=False, rtol=1e-6, atol=1e-6)


def test_s03_baseline_spearman():
    got = _read(paths.TABLES / "s03_baseline_metrics.csv")
    exp = _read(GOLDEN / "s03_baseline_metrics.csv")
    cols = ["value_col", "condition", "operator", "n", "spearman_rho"]
    pd.testing.assert_frame_equal(
        got[cols].sort_values(cols[:3]).reset_index(drop=True),
        exp[cols].sort_values(cols[:3]).reset_index(drop=True),
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_acsins_enhancer_suppressor_counts():
    df = _read(EXTRACTED / "fig4a_acsins_classification.csv")
    counts = df["category"].value_counts()
    assert int(counts.get("Enhancer", 0)) == 13
    assert int(counts.get("Suppressor", 0)) == 20
    assert len(df) == 139


def test_source_workbook_has_expected_sheets():
    xlsx = paths.FIGURES / "figure_data_supplement" / "source_data.xlsx"
    _require(xlsx)
    xl = pd.ExcelFile(xlsx)
    expected = {
        "Figure 1b",
        "Figure 1c",
        "Figure 2",
        "Figure 3a",
        "Figure 3b",
        "Figure 4a",
        "Figure 4b",
        "Figure 5",
        "Figure 6b",
        "Supp Figure 1",
        "Supp Figure 2a",
        "Supp Figure 4a",
        "Supp Figure 5",
        "Supp Figure 6",
        "Supp Figure 7",
        "Supp Figure 8",
        "Supp Figure 10 mAbs",
    }
    missing = expected - set(xl.sheet_names)
    assert not missing, missing
