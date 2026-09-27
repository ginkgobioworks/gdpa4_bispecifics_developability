"""Charge-transform horserace for supplementary figure S2.

The revised figure classifies each unique parental pair from the AC-SINS
replicate-noise band alone. The submitted absolute gates (observed ΔLmax
≥ 17.51 nm for an enhancer, ≤ 5 nm for a suppressor) are retained only as
``category_gated``, so the published gated horserace can be checked before
the ungated one is trusted.

For residual = observed − parental mean and half-width = 5 · σ_noise:

- enhancer: residual ≥ +half-width
- suppressor: residual ≤ −half-width
- otherwise: on-parental (empty category)

σ_noise propagates the median replicate standard deviations,

    σ_noise = sqrt(σ_bispecific² + σ_arm² / 2),

with both medians taken at AC-SINS ΔLmax in 1X PBS. Orientations of the same
parental pair are averaged before classification.

Each class is a single-predictor logistic regression against the on-parental
pairs. The predictor is standardized (mean 0, population sd 1) and fit with
``LogisticRegression(C=1e6)``. The default penalty (``C=1``) does not
reproduce the revision coefficients. Wald p-values come from the observed
information matrix of that fit. ΔAIC is within class, so the winner has
ΔAIC = 0.

Example::

    from prophet_ab.features.charge import horserace, load_acsins_pbs_pairs

    pairs = load_acsins_pbs_pairs()
    table = horserace(pairs, category_col="category_ungated")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .. import paths, schema
from ..normalize import strip_isotype_suffix

TRANSFORM_ORDER = (
    "B1_abs_sum_q",
    "B2_q_prod",
    "T1_cancel_eff",
    "T2_cancel_mass",
    "T3_signed_gmean",
    "T4_mono_dipole",
)

TRANSFORM_LABELS = {
    "B1_abs_sum_q": "|q1+q2|\nsum",
    "B2_q_prod": "q1*q2\nproduct",
    "T1_cancel_eff": "|Sq|/(|q1|+|q2|)\ncancel. eff.",
    "T2_cancel_mass": "min*opp\ncancel. mass",
    "T3_signed_gmean": "-sgn(q1q2)*sqrt|q1q2|\nsigned geom.",
    "T4_mono_dipole": "|Sq|/|Dq|\nmono/dipole",
}

# Submitted absolute gates, used only to reproduce the gated horserace.
ENHANCER_THRESHOLD = 17.51
SUPPRESSOR_CEILING = 5.0
NOISE_MULTIPLIER = 5
# Effectively unpenalized. sklearn's default C=1 will not match the revision.
LOGISTIC_C = 1e6


def charge_transforms(q1: np.ndarray, q2: np.ndarray) -> dict[str, np.ndarray]:
    """Six arm-charge summaries. ``q1`` and ``q2`` are MOE ``ens_charge``.

    ``T2_cancel_mass`` is ``min(|q1|, |q2|)`` when the arms have opposite
    sign and 0 otherwise. ``T4_mono_dipole`` is non-finite when the two
    charges are equal; callers drop those rows for that transform only.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    aq1 = np.abs(q1)
    aq2 = np.abs(q2)
    denom_sum = aq1 + aq2
    denom_diff = np.abs(q1 - q2)
    opposite = np.sign(q1) != np.sign(q2)
    return {
        "B1_abs_sum_q": np.abs(q1 + q2),
        "B2_q_prod": q1 * q2,
        "T1_cancel_eff": np.divide(
            np.abs(q1 + q2), denom_sum, out=np.zeros_like(denom_sum), where=denom_sum > 0
        ),
        "T2_cancel_mass": np.minimum(aq1, aq2) * opposite.astype(float),
        "T3_signed_gmean": -np.sign(q1 * q2) * np.sqrt(np.abs(q1 * q2)),
        "T4_mono_dipole": np.divide(
            np.abs(q1 + q2),
            denom_diff,
            out=np.full_like(denom_diff, np.nan),
            where=denom_diff > 1e-9,
        ),
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = float(x.std(ddof=0))
    if sd == 0:
        return np.zeros_like(x)
    return (x - float(x.mean())) / sd


def _wald_slope_p(x_std: np.ndarray, intercept: float, coef: float) -> float:
    """Two-sided Wald p for the slope, from the observed information."""
    design = np.column_stack([np.ones(len(x_std)), x_std])
    beta = np.array([intercept, coef], dtype=float)
    eta = np.clip(design @ beta, -30, 30)
    prob = 1 / (1 + np.exp(-eta))
    weight = np.clip(prob * (1 - prob), 1e-12, None)
    info = (design.T * weight) @ design
    se = float(np.sqrt(np.linalg.inv(info)[1, 1]))
    z = coef / se
    return float(2 * (1 - norm.cdf(abs(z))))


def _fit_one(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    x_std = _standardize(x)
    clf = LogisticRegression(C=LOGISTIC_C, solver="lbfgs", max_iter=5000)
    clf.fit(x_std.reshape(-1, 1), y)
    coef = float(clf.coef_[0, 0])
    intercept = float(clf.intercept_[0])
    proba = np.clip(clf.predict_proba(x_std.reshape(-1, 1))[:, 1], 1e-15, 1 - 1e-15)
    loglik = float(np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba)))
    return {
        "coef": coef,
        "wald_p": _wald_slope_p(x_std, intercept, coef),
        "aic": 4 - 2 * loglik,  # intercept + slope
        "auc": float(roc_auc_score(y, proba)),
        "n_pos": float(y.sum()),
        "n_neg": float(len(y) - y.sum()),
    }


def horserace(pairs: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Logistic horserace of each class against the on-parental pairs.

    ``category_col`` uses ``Enhancer``, ``Suppressor``, and ``""``.
    Rows missing ``q1`` or ``q2`` are dropped.
    """
    frame = pairs.dropna(subset=["q1", "q2"]).copy()
    on_parental = frame[frame[category_col] == ""]
    rows: list[dict] = []
    for side, label in (("enh", "Enhancer"), ("suppress", "Suppressor")):
        positive = frame[frame[category_col] == label]
        combined = pd.concat([positive, on_parental], ignore_index=True)
        y = np.array([1] * len(positive) + [0] * len(on_parental), dtype=float)
        transforms = charge_transforms(combined["q1"].to_numpy(), combined["q2"].to_numpy())
        for name in TRANSFORM_ORDER:
            fit = _fit_one(transforms[name], y)
            rows.append({"side": side, "side_label": label, "transform": name, **fit})
    table = pd.DataFrame(rows)
    table["delta_aic"] = table["aic"] - table.groupby("side")["aic"].transform("min")
    return table


def _unique_pairs(bispecific_pbs: pd.DataFrame) -> pd.DataFrame:
    """Average the two orientations, then keep one row per parental pair."""
    grouped = bispecific_pbs.copy()
    grouped["pair"] = grouped.apply(
        lambda row: tuple(sorted([row["parent_a"], row["parent_b"]])), axis=1
    )
    rows = []
    for pair, grp in grouped.groupby("pair"):
        arm_median: dict[str, float] = {}
        for _, row in grp.iterrows():
            arm_median[row["parent_a"]] = row["pa_median"]
            arm_median[row["parent_b"]] = row["pb_median"]
        observed = float(grp["observed"].mean())
        expected = float(grp["expected"].mean())
        rows.append(
            {
                "parent_a": pair[0],
                "parent_b": pair[1],
                "pa_median": arm_median[pair[0]],
                "pb_median": arm_median[pair[1]],
                "observed": observed,
                "expected": expected,
                "residual": observed - expected,
                "n_orientations": len(grp),
            }
        )
    return pd.DataFrame(rows)


def _categories(pairs: pd.DataFrame, half_width: float) -> pd.DataFrame:
    residual = pairs["residual"].to_numpy()
    observed = pairs["observed"].to_numpy()
    ungated = np.full(len(pairs), "", dtype=object)
    ungated[residual >= half_width] = "Enhancer"
    ungated[residual <= -half_width] = "Suppressor"
    gated = np.full(len(pairs), "", dtype=object)
    gated[(residual >= half_width) & (observed >= ENHANCER_THRESHOLD)] = "Enhancer"
    gated[(residual <= -half_width) & (observed <= SUPPRESSOR_CEILING)] = "Suppressor"
    out = pairs.copy()
    out["category_ungated"] = ungated
    out["category_gated"] = gated
    return out


def load_acsins_pbs_pairs() -> pd.DataFrame:
    """Unique PBS AC-SINS pairs with MOE charge and both category columns.

    ``attrs['noise_half_width']`` is the ± band used for classification.
    """
    summaries = pd.read_parquet(paths.S03 / "gdpa4_per_antibody.parquet")
    components = pd.read_parquet(paths.S02 / "bispecific_components.parquet")
    acsins = summaries[summaries["value_col"] == "acsins_delta_Lmax"].copy()

    monospecific = acsins[acsins["kind"] == schema.KIND_MONOSPECIFIC][
        ["antibody_name", "condition", "median", "std"]
    ].copy()
    monospecific["parent"] = monospecific["antibody_name"].map(strip_isotype_suffix)
    bispecific = acsins[acsins["kind"] == schema.KIND_BISPECIFIC][
        ["antibody_name", "condition", "median"]
    ].rename(columns={"median": "observed"})
    merged = bispecific.merge(components[["antibody_name", "parent_a", "parent_b"]], on="antibody_name")
    parent_median = monospecific[["parent", "condition", "median"]]
    merged = merged.merge(
        parent_median.rename(columns={"parent": "parent_a", "median": "pa_median"}),
        on=["parent_a", "condition"],
        how="left",
    ).merge(
        parent_median.rename(columns={"parent": "parent_b", "median": "pb_median"}),
        on=["parent_b", "condition"],
        how="left",
    )
    merged["expected"] = (merged["pa_median"] + merged["pb_median"]) / 2
    merged["residual"] = merged["observed"] - merged["expected"]

    pbs = acsins[acsins["condition"] == "1X PBS"]
    sigma_arm = float(np.nanmedian(pbs.loc[pbs["kind"] == schema.KIND_MONOSPECIFIC, "std"]))
    sigma_bsab = float(np.nanmedian(pbs.loc[pbs["kind"] == schema.KIND_BISPECIFIC, "std"]))
    half_width = NOISE_MULTIPLIER * float(np.sqrt(sigma_bsab**2 + sigma_arm**2 / 2))

    bispecific_pbs = merged[merged["condition"] == "1X PBS"].dropna(subset=["residual"])
    pairs = _categories(_unique_pairs(bispecific_pbs), half_width)

    charges = pd.read_csv(paths.RAW_IN_SILICO_DIR / "MOE_properties.csv")
    charges["parent"] = charges["antibody_name"].map(strip_isotype_suffix)
    charge_map = charges.drop_duplicates("parent").set_index("parent")["ens_charge"].to_dict()
    pairs["q1"] = pairs["parent_a"].map(charge_map)
    pairs["q2"] = pairs["parent_b"].map(charge_map)
    pairs.attrs["noise_half_width"] = half_width
    pairs.attrs["sigma_arm"] = sigma_arm
    pairs.attrs["sigma_bsab"] = sigma_bsab
    return pairs
