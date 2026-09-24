from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"

S01 = PROCESSED / "01_normalized"
S02 = PROCESSED / "02_joined"
S03 = PROCESSED / "03_aggregated"
S04 = PROCESSED / "04_features"
S05 = PROCESSED / "05_modeling"

REPORTS = REPO_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# Raw file locations
RAW_N3N4_TALL_CSV = RAW / "GDPa4_N3_N4_Summary_tall.csv"
RAW_GDPA1_XLSX = RAW / "[External] AbDev peer-review 246 IgGs_Master data file_GDPa1.xlsx"
RAW_N3_PRODUCTION_XLSX = RAW / "production" / "Data_Summary_U594PPMRG0_03032026 (1).xlsx"
RAW_N4_PRODUCTION_XLSX = RAW / "production" / "N4_U126M421G0_AntibodyList_reformatted.xlsx"
RAW_IN_SILICO_DIR = RAW / "in_silico_gpa1" / "GDPa1"
RAW_UMAP_COORDS = RAW / "bsab_design" / "umap_coords.csv"
