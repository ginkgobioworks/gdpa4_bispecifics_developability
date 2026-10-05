"""Extract source data for Figure 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd

from prophet_ab import paths, schema
from prophet_ab import normalize as nz

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load per-antibody summaries and bispecific component map
# ---------------------------------------------------------------------------
summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")

acsins = summaries[summaries["value_col"] == "acsins_delta_Lmax"].copy()

# monospecific parent medians and SDs (strip isotype suffix for join)
monospecific = acsins[acsins["kind"] == schema.KIND_MONOSPECIFIC][
    ["antibody_name", "condition", "median", "std"]
].copy()
monospecific["parent"] = monospecific["antibody_name"].map(nz.strip_isotype_suffix)

# bispecific observed medians
bispecific = acsins[acsins["kind"] == schema.KIND_BISPECIFIC][
    ["antibody_name", "condition", "median"]
].rename(columns={"median": "observed"})

# Join bispecific with parent medians
paired = bispecific.merge(
    components[["antibody_name", "parent_a", "parent_b"]],
    on="antibody_name",
)
paired = (
    paired.merge(
        monospecific[["parent", "condition", "median"]].rename(
            columns={"parent": "parent_a", "median": "pa_median"}
        ),
        on=["parent_a", "condition"],
        how="left",
    )
    .merge(
        monospecific[["parent", "condition", "median"]].rename(
            columns={"parent": "parent_b", "median": "pb_median"}
        ),
        on=["parent_b", "condition"],
        how="left",
    )
)
paired["expected"] = (paired["pa_median"] + paired["pb_median"]) / 2
paired["residual"] = paired["observed"] - paired["expected"]

# ---------------------------------------------------------------------------
# Error-propagation noise band
# ---------------------------------------------------------------------------
pbs = acsins[acsins["condition"] == "1X PBS"]
sigma_arm = float(np.nanmedian(pbs[pbs["kind"] == schema.KIND_MONOSPECIFIC]["std"]))
sigma_bsab = float(np.nanmedian(pbs[pbs["kind"] == schema.KIND_BISPECIFIC]["std"]))
noise_pbs = float(np.sqrt(sigma_bsab**2 + sigma_arm**2 / 2))

bispecific_acsins_pbs = paired[paired["condition"] == "1X PBS"].dropna(subset=["residual"]).copy()

# ---------------------------------------------------------------------------
# Panel A: unique-pair classification (average both orientations, then classify)
# ---------------------------------------------------------------------------
NOISE_MULTIPLIER = 5
ENHANCER_THRESHOLD = 17.51
SUPPRESSOR_CEILING = 5.0
band = NOISE_MULTIPLIER * noise_pbs

g = bispecific_acsins_pbs.copy()
g["pair"] = g.apply(
    lambda r: tuple(sorted([r["parent_a"], r["parent_b"]])), axis=1
)

rows_a = []
for pair, grp in g.groupby("pair"):
    med = {}
    for _, rr in grp.iterrows():
        med[rr["parent_a"]] = rr["pa_median"]
        med[rr["parent_b"]] = rr["pb_median"]
    observed = float(grp["observed"].mean())
    expected = float(grp["expected"].mean())
    residual = observed - expected
    if residual >= band and observed >= ENHANCER_THRESHOLD:
        cat = "Enhancer"
    elif residual <= -band and observed <= SUPPRESSOR_CEILING:
        cat = "Suppressor"
    else:
        cat = ""
    rows_a.append({
        "pair": f"{pair[0]}__{pair[1]}",
        "parent_a": pair[0],
        "parent_b": pair[1],
        "observed": round(observed, 6),
        "expected": round(expected, 6),
        "residual": round(residual, 6),
        "category": cat,
        "band_half_width": round(band, 6),
    })

df_a = pd.DataFrame(rows_a)
out_a = OUT_DIR / "fig4a_acsins_classification.csv"
df_a.to_csv(out_a, index=False)
print(
    f"Panel A: {(df_a['category'] == 'Enhancer').sum()} enhancers, "
    f"{(df_a['category'] == 'Suppressor').sum()} suppressors, "
    f"{len(df_a)} pairs -> {out_a}"
)

# ---------------------------------------------------------------------------
# Panel B: arm charge quadrant scatter
# ---------------------------------------------------------------------------
n4f = pd.read_csv(paths.RAW_IN_SILICO_DIR / "MOE_properties.csv")
n4f["parent"] = n4f["antibody_name"].map(nz.strip_isotype_suffix)
charge_map = n4f.set_index("parent")["ens_charge"].to_dict()

pairs = df_a.copy()
pairs["q1"] = pairs["parent_a"].map(charge_map)
pairs["q2"] = pairs["parent_b"].map(charge_map)
pairs["xc"] = pairs[["q1", "q2"]].max(axis=1)
pairs["yc"] = pairs[["q1", "q2"]].min(axis=1)
pairs_charged = pairs.dropna(subset=["q1", "q2"]).copy()

out_b = OUT_DIR / "fig4b_charge_quadrant.csv"
pairs_charged[
    ["pair", "parent_a", "parent_b", "q1", "q2", "xc", "yc", "category"]
].to_csv(out_b, index=False)
print(
    f"Panel B: {len(pairs_charged)} pairs with charge data -> {out_b}"
)
