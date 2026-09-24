"""Assert the listed figure PNGs exist after `make all`."""
from pathlib import Path

from prophet_ab import paths

REQUIRED = [
    "s00_umap_selected.png",
    "s00_arm_distribution.png",
    "s00_coverage_consolidated.png",
    "s01_swap_pair_scatters_pearson.png",
    "s01_violin_pts_if_tm_pooled.png",
    "s02_cross_platform_scatter.png",
    "s09_polyreactivity_vs_chromatography.png",
    "s07_top_features_all_groups_reported_subset.png",
    "s07_top_features_all_arm_groups_reported_subset.png",
    "s08_coefficient_overlay_reported_subset.png",
    "main/figure_2_tiers.png",
    "main/figure_2a_wide.png",
    "main/figure_3.png",
    "main/figure_3_version_2.png",
    "main/figure_4.png",
    "main/figure_5_loo.png",
    "main/figure_5_version_2.png",
    "main/figure_5_version_3.png",
    "main/figure_6_loo.png",
    "main/supplemental_figure_ceiling.png",
]


def test_required_figures_exist_and_are_nonempty():
    missing = []
    tiny = []
    for rel in REQUIRED:
        p = paths.FIGURES / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size < 1000:
            tiny.append(rel)
    assert not missing, f"run `make all` first; missing {missing}"
    assert not tiny, f"suspiciously small PNGs: {tiny}"
