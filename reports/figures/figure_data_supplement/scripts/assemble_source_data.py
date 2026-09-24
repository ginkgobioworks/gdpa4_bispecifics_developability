"""Assemble all extracted figure data into a single source data Excel workbook.

Reads CSVs from reports/figures/figure_data_supplement/extracted/ and writes
one sheet per figure/panel to reports/figures/figure_data_supplement/source_data.xlsx,
following the Nature Communications example format.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import pandas as pd

EXTRACTED = Path(__file__).resolve().parent.parent / "extracted"
OUT = Path(__file__).resolve().parent.parent / "source_data.xlsx"

SHEETS: list[tuple[str, str, str | None]] = [
    # (sheet_name, csv_filename, description_for_header)
    # --- Main Figures ---
    ("Figure 1b", "fig1b_umap.csv",
     "UMAP coordinates for all possible mAb pairs (background) and selected 160 bispecifics"),
    ("Figure 1c", "fig1c_arm_distribution.csv",
     "Parent mAb arm usage counts across 160 selected bispecifics"),
    ("Figure 2", "fig2_swap_pairs.csv",
     "Swap pair orientation comparison: lo/hi medians per assay for pairs built in both orientations"),
    ("Figure 3a", "fig3a_scatter.csv",
     "Observed bispecific value vs parental operator prediction, per metric and tier"),
    ("Figure 3b", "fig3b_heatmap.csv",
     "Spearman rho for each metric x operator combination; is_chosen marks the selected operator"),
    ("Figure 4a", "fig4a_acsins_classification.csv",
     "AC-SINS observed vs expected with enhancer/suppressor classification (PBS pH 7.4)"),
    ("Figure 4b", "fig4b_charge_quadrant.csv",
     "Arm charge quadrant data: MOE ensemble charge per arm with classification"),
    ("Figure 5", "fig5_scatter.csv",
     "HIC vs HAC scatter colored by polyreactivity scores (N3 and N4)"),
    ("Figure 5 frontier", "fig5_pareto_frontier.csv",
     "N4 Pareto frontier points (maximize HIC and HAC)"),
    ("Figure 6b", "fig6b_loo_boxenplot.csv",
     "Leave-one-out cross-validation Spearman rho per model, config, and label"),
    # --- Supplemental Figures ---
    ("Supp Figure 1", "supp_fig1_cross_platform.csv",
     "Cross-platform (this campaign vs GDPa1) scatter data per assay"),
    ("Supp Figure 2a", "supp_fig2a_delta_aic.csv",
     "Delta-AIC for six charge transforms predicting enhancer/suppressor status"),
    ("Supp Figure 2b", "supp_fig2b_enhancer_violin.csv",
     "Absolute sum of arm charges for enhancers vs not-classified"),
    ("Supp Figure 2c", "supp_fig2c_suppressor_violin.csv",
     "Signed geometric mean of arm charges for suppressors vs not-classified"),
    ("Supp Figure 4a", "supp_fig4a_scatter.csv",
     "HIC x HAC scatter with Pareto frontier crosser categories"),
    ("Supp Figure 4a frontier", "supp_fig4a_frontier.csv",
     "N4 Pareto frontier points for supplemental ceiling figure"),
    ("Supp Figure 4b", "supp_fig4b_roc.csv",
     "ROC curves: frontier distance predicting top-20% polyreactivity"),
    ("Supp Figure 4c", "supp_fig4c_confusion.csv",
     "Confusion matrix: frontier crossing vs top-20% PR-CHO"),
    ("Supp Figure 5", "supp_fig5_violin_tm.csv",
     "Pooled Tm1+Tm2 values for bispecific vs monospecific violins"),
    ("Supp Figure 6", "supp_fig6_top_features.csv",
     "Top-8 permutation importance features per label (operator configs)"),
    ("Supp Figure 7", "supp_fig7_top_features_arm.csv",
     "Top-8 permutation importance features per label (arm configs)"),
    ("Supp Figure 8", "supp_fig8_coefficients.csv",
     "Format-effect OLS coefficients per Fv per assay (z-scored, FDR-corrected)"),
    ("Supp Figure 10 mAbs", "supp_fig10_mab_values.csv",
     "Per-mAb biophysical property values (13 properties, 246 GDPa1 mAbs)"),
]

WIDE_PIVOT_SHEETS: list[tuple[str, str, str]] = [
    ("Supp Figure 10 pairs", "supp_fig10_pair_values.csv",
     "Per-pair average biophysical property values (13 properties, all C(246,2) pairs)"),
]


def write_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    description: str,
) -> None:
    """Write a single sheet with a header row containing the figure label."""
    df.to_excel(writer, sheet_name=sheet_name, startrow=2, index=False)
    ws = writer.sheets[sheet_name]
    ws.cell(row=1, column=1, value=sheet_name)
    ws.cell(row=2, column=1, value=description)


def main() -> None:
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for sheet_name, csv_name, desc in SHEETS:
            csv_path = EXTRACTED / csv_name
            if not csv_path.exists():
                print(f"SKIP {csv_name}: not found")
                continue
            df = pd.read_csv(csv_path)
            write_sheet(writer, sheet_name, df, desc)
            print(f"  {sheet_name}: {len(df)} rows")

        for sheet_name, csv_name, desc in WIDE_PIVOT_SHEETS:
            csv_path = EXTRACTED / csv_name
            if not csv_path.exists():
                print(f"SKIP {csv_name}: not found")
                continue
            long = pd.read_csv(csv_path)
            wide = long.pivot_table(
                index=["pair", "is_selected_pair"],
                columns="property",
                values="value_avg",
                aggfunc="first",
            ).reset_index()
            wide.columns.name = None
            write_sheet(writer, sheet_name, wide, desc)
            print(f"  {sheet_name}: {len(wide)} rows (pivoted from {len(long)} long)")

    print(f"\nWrote {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
