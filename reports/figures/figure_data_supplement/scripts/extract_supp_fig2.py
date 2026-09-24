"""Extract source data for Supplemental Figure 2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from prophet_ab import paths, schema
from prophet_ab import normalize as nz

OUT_DIR = Path(__file__).resolve().parents[1] / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Classification thresholds (Ammar pattern)
NOISE_MULTIPLIER = 5
ENHANCER_THRESHOLD = 17.51  # nm, from Arsiwala
SUPPRESSOR_CEILING = 5.0  # nm


def _compute_transforms(q1: np.ndarray, q2: np.ndarray) -> dict[str, np.ndarray]:
    """Six charge transforms from per-arm MOE ens_charge."""
    aq1 = np.abs(q1)
    aq2 = np.abs(q2)
    return {
        "B1_abs_sum_q": np.abs(q1 + q2),
        "B2_q_prod": q1 * q2,
        "T1_cancel_eff": np.where(
            (aq1 + aq2) > 0, np.abs(q1 + q2) / (aq1 + aq2), 0.0
        ),
        "T2_cancel_mass": np.minimum(aq1, aq2)
        * np.where(q1 * q2 < 0, 1.0, -1.0),
        "T3_signed_gmean": -np.sign(q1 * q2) * np.sqrt(np.abs(q1 * q2)),
        "T4_mono_dipole": np.where(
            np.abs(q1 - q2) > 1e-9, np.abs(q1 + q2) / np.abs(q1 - q2), 0.0
        ),
    }


def main() -> None:
    # ---- Load and classify AC-SINS PBS pairs ----
    summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")

    acsins = summaries[summaries["value_col"] == "acsins_delta_Lmax"].copy()

    n4 = acsins[acsins["kind"] == schema.KIND_N4][
        ["antibody_name", "condition", "median", "std"]
    ].copy()
    n4["parent"] = n4["antibody_name"].map(nz.strip_isotype_suffix)

    n3 = acsins[acsins["kind"] == schema.KIND_N3][
        ["antibody_name", "condition", "median"]
    ].rename(columns={"median": "observed"})

    n3p = n3.merge(
        components[["antibody_name", "parent_a", "parent_b"]],
        on="antibody_name",
    )
    n3p = n3p.merge(
        n4[["parent", "condition", "median"]].rename(
            columns={"parent": "parent_a", "median": "pa_median"}
        ),
        on=["parent_a", "condition"],
        how="left",
    ).merge(
        n4[["parent", "condition", "median"]].rename(
            columns={"parent": "parent_b", "median": "pb_median"}
        ),
        on=["parent_b", "condition"],
        how="left",
    )
    n3p["expected"] = (n3p["pa_median"] + n3p["pb_median"]) / 2
    n3p["residual"] = n3p["observed"] - n3p["expected"]

    # Noise band (PBS only)
    pbs = acsins[acsins["condition"] == "1X PBS"]
    sigma_arm = float(np.nanmedian(pbs[pbs["kind"] == schema.KIND_N4]["std"]))
    sigma_bsab = float(np.nanmedian(pbs[pbs["kind"] == schema.KIND_N3]["std"]))
    noise_pbs = float(np.sqrt(sigma_bsab**2 + sigma_arm**2 / 2))
    band = NOISE_MULTIPLIER * noise_pbs

    # PBS pairs, average-then-classify
    n3_pbs = n3p[n3p["condition"] == "1X PBS"].dropna(subset=["residual"]).copy()
    n3_pbs["pair"] = n3_pbs.apply(
        lambda r: tuple(sorted([r["parent_a"], r["parent_b"]])), axis=1
    )
    classified_rows = []
    for pair, grp in n3_pbs.groupby("pair"):
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
        classified_rows.append(
            {
                "pair": pair,
                "parent_a": pair[0],
                "parent_b": pair[1],
                "pa_median": med[pair[0]],
                "pb_median": med[pair[1]],
                "observed": observed,
                "expected": expected,
                "residual": residual,
                "category": cat,
            }
        )
    classified = pd.DataFrame(classified_rows)

    # ---- Attach charge data ----
    n4f = pd.read_csv(paths.RAW_IN_SILICO_DIR / "MOE_properties.csv")
    n4f["parent"] = n4f["antibody_name"].map(nz.strip_isotype_suffix)
    charge_map = n4f.set_index("parent")["ens_charge"].to_dict()

    classified["q1"] = classified["parent_a"].map(charge_map)
    classified["q2"] = classified["parent_b"].map(charge_map)
    pairs_charged = classified.dropna(subset=["q1", "q2"]).copy()

    # ---- Panel A: delta-AIC bar chart ----
    nc = pairs_charged[pairs_charged["category"] == ""]
    enh = pairs_charged[pairs_charged["category"] == "Enhancer"]
    sup = pairs_charged[pairs_charged["category"] == "Suppressor"]

    aic_rows = []
    for side, pos, side_label in [
        ("enh", enh, "Enhancer"),
        ("suppress", sup, "Suppressor"),
    ]:
        combined = pd.concat([pos, nc], ignore_index=True)
        y = np.array([1] * len(pos) + [0] * len(nc))
        transforms = _compute_transforms(
            combined["q1"].values, combined["q2"].values
        )
        for tname, xvals in transforms.items():
            X = xvals.reshape(-1, 1)
            valid = ~(np.isnan(X).ravel() | np.isinf(X).ravel())
            Xv, yv = X[valid], y[valid]
            if len(np.unique(yv)) < 2 or len(Xv) < 5:
                continue
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf.fit(Xv, yv)
            proba = clf.predict_proba(Xv)
            ll = float(
                np.sum(
                    yv * np.log(proba[:, 1] + 1e-15)
                    + (1 - yv) * np.log(proba[:, 0] + 1e-15)
                )
            )
            aic = 2 * 2 - 2 * ll  # k=2 (intercept + coefficient)
            aic_rows.append(
                {
                    "transform": tname,
                    "side": side_label,
                    "aic": aic,
                }
            )

    horserace = pd.DataFrame(aic_rows)
    for side in horserace["side"].unique():
        mask = horserace["side"] == side
        horserace.loc[mask, "delta_aic"] = (
            horserace.loc[mask, "aic"] - horserace.loc[mask, "aic"].min()
        )

    out_a = OUT_DIR / "supp_fig2a_delta_aic.csv"
    horserace[["transform", "side", "aic", "delta_aic"]].to_csv(out_a, index=False)
    print(f"Panel A: wrote {len(horserace)} rows to {out_a}")

    # ---- Panel B: enhancer violin - |q1+q2| ----
    enh_charged = pairs_charged[pairs_charged["category"] == "Enhancer"]
    nc_charged = pairs_charged[pairs_charged["category"] == ""]

    panel_b_rows = []
    for df_sub, cat_label in [(enh_charged, "Enhancer"), (nc_charged, "Not classified")]:
        for _, row in df_sub.iterrows():
            panel_b_rows.append(
                {
                    "pair": f"{row['parent_a']}__{row['parent_b']}",
                    "abs_sum_charge": abs(row["q1"] + row["q2"]),
                    "category": cat_label,
                }
            )
    panel_b = pd.DataFrame(panel_b_rows)
    out_b = OUT_DIR / "supp_fig2b_enhancer_violin.csv"
    panel_b.to_csv(out_b, index=False)
    print(f"Panel B: wrote {len(panel_b)} rows to {out_b}")

    # ---- Panel C: suppressor violin - signed geometric mean ----
    sup_charged = pairs_charged[pairs_charged["category"] == "Suppressor"]

    panel_c_rows = []
    for df_sub, cat_label in [
        (sup_charged, "Suppressor"),
        (nc_charged, "Not classified"),
    ]:
        for _, row in df_sub.iterrows():
            q1, q2 = row["q1"], row["q2"]
            sgm = -np.sign(q1 * q2) * np.sqrt(np.abs(q1 * q2))
            panel_c_rows.append(
                {
                    "pair": f"{row['parent_a']}__{row['parent_b']}",
                    "signed_geomean_charge": sgm,
                    "category": cat_label,
                }
            )
    panel_c = pd.DataFrame(panel_c_rows)
    out_c = OUT_DIR / "supp_fig2c_suppressor_violin.csv"
    panel_c.to_csv(out_c, index=False)
    print(f"Panel C: wrote {len(panel_c)} rows to {out_c}")


if __name__ == "__main__":
    main()
