"""Extract source data for Supplemental Figure 10."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from itertools import combinations

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from prophet_ab import paths

OUT_DIR = Path(__file__).resolve().parent.parent / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESIGN_DATA = paths.RAW / "bsab_design" / "bsabs_dec_2025" / "data"

# ── Sequence helpers ─────────────────────────────────────────────────────

def _seq(row):
    vh = row.get("vh_protein_sequence")
    vl = row.get("vl_protein_sequence")
    parts = []
    if pd.notna(vh):
        parts.append(str(vh).strip())
    if pd.notna(vl):
        parts.append(str(vl).strip())
    return "".join(parts).replace(" ", "").upper()


def _net_charge_at_pH(seq, pH=7.4):
    if not seq:
        return 0.0
    nK = seq.count("K")
    nR = seq.count("R")
    nH = seq.count("H")
    nD = seq.count("D")
    nE = seq.count("E")
    q = 0.0
    q += nK / (1 + 10 ** (pH - 10.53))
    q += nR / (1 + 10 ** (pH - 12.48))
    q += nH / (1 + 10 ** (pH - 6.0))
    q += 1.0 / (1 + 10 ** (pH - 8.0))
    q -= nD / (1 + 10 ** (3.65 - pH))
    q -= nE / (1 + 10 ** (4.25 - pH))
    q -= 1.0 / (1 + 10 ** (3.1 - pH))
    return q


def _props(row):
    s = _seq(row)
    vh = row.get("vh_protein_sequence")
    vl = row.get("vl_protein_sequence")
    vh_str = str(vh).strip().upper() if pd.notna(vh) else ""
    vl_str = str(vl).strip().upper() if pd.notna(vl) else ""
    nans = {
        "pI": np.nan,
        "hydrophobicity": np.nan,
        "aromaticity": np.nan,
        "instability_index": np.nan,
        "length": np.nan,
    }
    if not s:
        return pd.Series(nans)
    try:
        pa = ProteinAnalysis(s)
        return pd.Series(
            {
                "pI": pa.isoelectric_point(),
                "hydrophobicity": pa.gravy(),
                "aromaticity": pa.aromaticity(),
                "instability_index": pa.instability_index(),
                "length": float(len(s)),
            }
        )
    except Exception:
        return pd.Series(nans)


# ── Load GDPa1 + computed + MOE ──────────────────────────────────────────
raw = pd.read_csv(DESIGN_DATA / "GDPa1_v1.2_20250814.csv")
computed = raw.apply(_props, axis=1)
raw = pd.concat([raw, computed], axis=1)

moe = pd.read_csv(DESIGN_DATA / "p739_moe_properties.csv")
df_mabs = raw.merge(moe, on="antibody_name", how="left", suffixes=("", "_moe"))

# ── Load exported bsAbs for selection flags ──────────────────────────────
df_exported = pd.read_csv(DESIGN_DATA / "exported_bsabs.csv")
all_parents = sorted(
    set(
        df_exported["antibody_name-1"].tolist()
        + df_exported["antibody_name-2"].tolist()
    )
)
selected_parent_set = set(all_parents)

# ── Define properties ────────────────────────────────────────────────────
assays = ["HIC", "PR_CHO", "AC-SINS_pH7.4", "Tm2"]
computed_props = ["pI", "hydrophobicity", "aromaticity", "instability_index", "length"]
moe_props = ["patch_cdr_hyd", "ens_charge", "dipole_moment", "affinity_VL_VH"]
cols_of_interest = assays + computed_props + moe_props

# ── Build mAb-level output ───────────────────────────────────────────────
mab_rows = []
for col in cols_of_interest:
    for _, row in df_mabs.iterrows():
        val = row.get(col)
        if pd.isna(val):
            continue
        mab_rows.append(
            {
                "antibody_name": row["antibody_name"],
                "property": col,
                "value": float(val),
                "is_selected_parent": row["antibody_name"] in selected_parent_set,
            }
        )

df_mab_out = pd.DataFrame(mab_rows)

# ── Build all C(N,2) pair averages ───────────────────────────────────────
n_mabs = len(df_mabs)
idx = np.array(list(combinations(range(n_mabs), 2)))
names = df_mabs["antibody_name"].values

# Build selected pair set from exported bsAbs
selected_pairs = set()
for _, erow in df_exported.iterrows():
    a, b = erow["antibody_name-1"], erow["antibody_name-2"]
    selected_pairs.add((min(a, b), max(a, b)))

pair_rows = []
for col in cols_of_interest:
    v = df_mabs[col].values.astype(float)
    avg = (v[idx[:, 0]] + v[idx[:, 1]]) / 2
    for k in range(len(idx)):
        val = avg[k]
        if np.isnan(val):
            continue
        n1, n2 = names[idx[k, 0]], names[idx[k, 1]]
        pair_key = (min(n1, n2), max(n1, n2))
        pair_rows.append(
            {
                "pair": f"{n1}__{n2}",
                "property": col,
                "value_avg": float(val),
                "is_selected_pair": pair_key in selected_pairs,
            }
        )

df_pair_out = pd.DataFrame(pair_rows)

# ── Write outputs ────────────────────────────────────────────────────────
mab_path = OUT_DIR / "supp_fig11_mab_values.csv"
df_mab_out.to_csv(mab_path, index=False)
print(f"wrote {mab_path} ({len(df_mab_out)} rows)")

pair_path = OUT_DIR / "supp_fig11_pair_values.csv"
df_pair_out.to_csv(pair_path, index=False)
print(f"wrote {pair_path} ({len(df_pair_out)} rows)")
