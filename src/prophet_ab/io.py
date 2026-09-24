"""Readers for raw inputs. Each returns a typed DataFrame; no joins, no aggregation."""
from __future__ import annotations

import pandas as pd

from . import paths


def read_n3n4_tall() -> pd.DataFrame:
    df = pd.read_csv(paths.RAW_N3N4_TALL_CSV)
    df["plateid"] = df["plateid"].astype("string")
    return df


def read_gdpa1_tidy() -> pd.DataFrame:
    return pd.read_excel(paths.RAW_GDPA1_XLSX, sheet_name="Assay Data - tidy format")


def read_gdpa1_sequences() -> pd.DataFrame:
    return pd.read_excel(paths.RAW_GDPA1_XLSX, sheet_name="Sequences")


def read_gdpa1_prior_lit() -> pd.DataFrame:
    return pd.read_excel(paths.RAW_GDPA1_XLSX, sheet_name="Prior literature Data")


def read_n3_production() -> pd.DataFrame:
    """N3 production summary with sequences, yields, vendor QC."""
    df = pd.read_excel(
        paths.RAW_N3_PRODUCTION_XLSX,
        sheet_name="Customized service_Main item",
    )
    rename = {
        "Order ID": "order_id",
        "Order Item Id": "order_item_id",
        "Protein Name*": "antibody_name",
        "Plasmid Transfection Ratio": "plasmid_ratio",
        "seq1 Name": "seq1_name",
        "sequence 1": "sequence_1",
        "seq2 Name": "seq2_name",
        "sequence 2": "sequence_2",
        "seq3 Name": "seq3_name",
        "sequence 3": "sequence_3",
        "seq4 Name": "seq4_name",
        "sequence 4": "sequence_4",
        "Expression Volume (mL)": "expression_volume_ml",
        "Concentration\n(mg/ml)": "concentration_mg_per_ml",
        "Purity by SDS-PAGE under NR(%)": "sds_page_nr_purity_pct",
        "Purity by SEC-HPLC(%)": "sec_hplc_purity_pct",
        "Amount \n(mg)": "amount_mg",
        "Endotoxin Level\n(EU/mg)": "endotoxin_eu_per_mg",
        "Lot No": "lot_no",
        "Volume Size\n(ml)": "volume_size_ml",
        "Unit (Tubes)": "unit_tubes",
        "Purification Strategy": "purification_strategy",
        "Buffer": "storage_buffer",
    }
    return df.rename(columns=rename)


def read_n4_production() -> pd.DataFrame:
    df = pd.read_excel(paths.RAW_N4_PRODUCTION_XLSX)
    return df.rename(columns={
        "Protein Name": "antibody_name",
        "Expression Volume (mL)": "expression_volume_ml",
        "Amount (mg)": "amount_mg",
        "Concentration (mg/mL)": "concentration_mg_per_ml",
    })
