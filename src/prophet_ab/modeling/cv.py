"""CV runner with parent-aware splits and per-feature importance.

Two split strategies live here, both enforcing the same parent-leakage
guarantee (no test N3 shares a parent with any train N3 in the same fold)
but with different fold geometry:

- D-2026-04-27-CV-PARENTAWARE: 5-fold k-fold-style. The 65 unique N4 parents
  are partitioned into 5 sets. For each fold, the test set is N3s with BOTH
  parents in the held-out set; the train set is N3s with NEITHER parent in
  the held-out set; "bridge" N3s with one parent in each are excluded from
  that fold (they remain available in other folds). Built by
  `build_parent_aware_splits`. Reports metrics on the union of OOF
  predictions across all 5 folds. With one giant connected component of 61
  parents, only ~27 N3s receive an OOF prediction; the rest are bridges and
  never sit in any test fold.
- D-2026-04-30-CV-LOO-PARENT-DISJOINT: per-N3 leave-one-out generalization
  of the above. For each N3 (parents A, B), test = {that N3}, train = N3s
  whose parents are neither A nor B. Built by
  `build_parent_disjoint_loo_splits`. All 160 N3s receive an OOF prediction
  (subject to y observability and `min_train`).

`run_one` is split-strategy agnostic — it accepts an arbitrary
`splits: list[tuple[train_idx, test_idx]]` and reports Spearman, Pearson,
R², MSE, RMSE, MAE on the union of OOF predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import transforms


_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


class _FlatPLS(PLSRegression):
    """PLSRegression wrapper that returns 1-D predictions."""

    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def _linear_pipeline(est):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("est", est),
    ])


def make_model(name: str, label: str | None = None, n_features: int | None = None):
    if name == "ridge":
        base = _linear_pipeline(RidgeCV(alphas=_ALPHAS))
    elif name == "lasso":
        base = _linear_pipeline(LassoCV(alphas=_ALPHAS, max_iter=10_000))
    elif name == "elasticnet":
        base = _linear_pipeline(
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                alphas=_ALPHAS,
                max_iter=10_000,
            ),
        )
    elif name == "rf":
        base = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("est", RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=5,
                random_state=0,
                n_jobs=-1,
            )),
        ])
    elif name == "xgb":
        base = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
        )
    elif name == "hgbm":
        base = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=0,
        )
    elif name == "pls":
        n_comp = min(n_features or 5, 5)
        n_comp = max(n_comp, 1)
        base = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("est", _FlatPLS(n_components=n_comp)),
        ])
    else:
        raise ValueError(f"unknown model: {name}")
    spec = transforms.get_transform(label) if label else None
    if spec is None:
        return base
    return TransformedTargetRegressor(
        regressor=base, func=spec.func, inverse_func=spec.inverse_func,
    )


MODELS: tuple[str, ...] = ("ridge", "lasso", "elasticnet", "rf", "xgb", "hgbm", "pls")


def _component_arrays(
    antibody_names: list[str], components: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align `components` to `antibody_names`, returning parallel arrays for
    parent_a, parent_b, and the row position in `antibody_names`."""
    name_to_pos = {n: i for i, n in enumerate(antibody_names)}
    sub = components[components["antibody_name"].isin(name_to_pos)]
    pos = sub["antibody_name"].map(name_to_pos).to_numpy()
    return (sub["parent_a"].to_numpy(), sub["parent_b"].to_numpy(), pos)


def build_parent_aware_splits(
    antibody_names: list[str],
    components: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Strict disjoint-parent k-fold.

    Partitions the unique parents of `components` into `n_splits` random sets
    (using `random_state`). For each held-out parent set, returns
    `(train_idx, test_idx)` over the positions in `antibody_names`:

      - test_idx: rows where BOTH parents are in the held-out set
      - train_idx: rows where NEITHER parent is in the held-out set
      - "bridge" rows (one parent in each) are dropped from this fold

    `antibody_names` defines the canonical row ordering used by the caller's
    feature/label matrices.
    """
    pa, pb, pos = _component_arrays(antibody_names, components)
    parents = pd.unique(np.column_stack([pa, pb]).ravel())
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(parents))
    parent_folds = np.array_split(perm, n_splits)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in parent_folds:
        held = set(parents[fold])
        a_in = np.array([p in held for p in pa])
        b_in = np.array([p in held for p in pb])
        test_mask = a_in & b_in
        train_mask = (~a_in) & (~b_in)
        splits.append((np.sort(pos[train_mask]), np.sort(pos[test_mask])))
    return splits


def build_parent_disjoint_loo_splits(
    antibody_names: list[str],
    components: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-N3 leave-one-out with parent exclusion (D-2026-04-30-CV-LOO-PARENT-DISJOINT).

    For each N3 in `antibody_names` (with parents `(pa, pb)`), produces one
    fold:

      - test_idx: position of that N3
      - train_idx: positions of N3s whose parent_a AND parent_b are both
        NOT in {pa, pb}

    Returns `len(antibody_names)` splits, one per N3, each with a singleton
    test set. N3s in `antibody_names` that have no row in `components`
    (shouldn't happen in practice — the modeling pipeline aligns them) are
    skipped.

    Same parent-leakage guarantee as `build_parent_aware_splits`: no train
    N3 shares an arm with the test N3.
    """
    pa, pb, pos = _component_arrays(antibody_names, components)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(pos)):
        held = {pa[i], pb[i]}
        a_out = np.array([p not in held for p in pa])
        b_out = np.array([p not in held for p in pb])
        train_mask = a_out & b_out
        train_idx = np.sort(pos[train_mask])
        test_idx = np.array([pos[i]], dtype=train_idx.dtype)
        splits.append((train_idx, test_idx))
    return splits


@dataclass(frozen=True)
class CVResult:
    label: str
    config: str
    model: str
    target_transform: str
    n_features: int
    n_samples: int            # rows where y is observed (across all folds, pre-split)
    n_test_predictions: int   # OOF predictions accumulated across folds
    spearman_rho: float
    pearson_r: float
    r2: float
    mse: float
    rmse: float
    mae: float


def _empty_result(label: str, config: str, model: str, n_features: int,
                  n_samples: int, n_test: int = 0,
                  target_transform: str = "none") -> CVResult:
    return CVResult(label=label, config=config, model=model,
                    target_transform=target_transform,
                    n_features=n_features, n_samples=n_samples,
                    n_test_predictions=n_test,
                    spearman_rho=np.nan, pearson_r=np.nan, r2=np.nan,
                    mse=np.nan, rmse=np.nan, mae=np.nan)


_OOF_COLS = ["antibody_name", "y_true", "y_pred"]


def _empty_oof() -> pd.DataFrame:
    return pd.DataFrame(columns=_OOF_COLS)


def run_one(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    label: str,
    config: str,
    model_name: str,
    min_train: int = 20,
    min_test_total: int = 10,
) -> tuple[CVResult, pd.DataFrame]:
    """Run parent-aware CV for one (label, config, model) combination.

    For each (train_idx, test_idx) in `splits`, fits on rows where y is
    observed in train_idx and predicts on rows where y is observed in
    test_idx. OOF predictions are accumulated across folds; metrics are
    computed on the union. Folds with fewer than `min_train` train rows or
    zero test rows (after y.notna() masking) are skipped silently. The combo
    is reported as empty if total OOF predictions < `min_test_total`.

    Returns (CVResult, oof_df) where oof_df has columns
    ['antibody_name', 'y_true', 'y_pred']. Empty (zero-row) when the result
    is empty.
    """
    spec = transforms.get_transform(label)
    tname = spec.name if spec else "none"
    p = X.shape[1]
    mask = y.notna().to_numpy()
    n_observed = int(mask.sum())
    if p == 0 or n_observed == 0:
        return _empty_result(label, config, model_name, p, n_observed,
                             target_transform=tname), _empty_oof()

    y_arr = y.to_numpy()
    X_arr = X.to_numpy()
    name_arr = np.asarray(y.index)

    test_names: list = []
    test_y: list[float] = []
    test_pred: list[float] = []
    for train_idx, test_idx in splits:
        tr = train_idx[mask[train_idx]]
        te = test_idx[mask[test_idx]]
        if len(tr) < min_train or len(te) == 0:
            continue
        model = make_model(model_name, label=label, n_features=p)
        try:
            model.fit(X_arr[tr], y_arr[tr])
            pred = np.asarray(model.predict(X_arr[te])).ravel()
        except Exception:
            continue
        test_names.extend(name_arr[te].tolist())
        test_y.extend(y_arr[te].tolist())
        test_pred.extend(pred.tolist())

    n_test = len(test_y)
    if n_test < min_test_total:
        return _empty_result(label, config, model_name, p, n_observed, n_test,
                             target_transform=tname), _empty_oof()

    yn = np.asarray(test_y, dtype="float64")
    yp = np.asarray(test_pred, dtype="float64")
    rho, _ = stats.spearmanr(yn, yp)
    pear = float(np.corrcoef(yn, yp)[0, 1]) if np.std(yp) > 0 else np.nan
    mse = float(mean_squared_error(yn, yp))
    result = CVResult(
        label=label,
        config=config,
        model=model_name,
        target_transform=tname,
        n_features=p,
        n_samples=n_observed,
        n_test_predictions=n_test,
        spearman_rho=float(rho) if rho is not None and np.isfinite(rho) else np.nan,
        pearson_r=pear,
        r2=float(r2_score(yn, yp)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(yn, yp)),
    )
    oof = pd.DataFrame({
        "antibody_name": test_names,
        "y_true": yn,
        "y_pred": yp,
    })
    return result, oof


def results_to_frame(results: list[CVResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def _unwrap_ttr(fitted):
    """Unwrap TransformedTargetRegressor to access the inner estimator."""
    if hasattr(fitted, "regressor_"):
        return fitted.regressor_
    return fitted


def _builtin_importance(model_name: str, fitted, feature_names: list[str]) -> pd.DataFrame:
    """Per-feature built-in importance for the given model.

    - ridge/lasso/elasticnet: standardized coefficients from the Pipeline.
    - pls: regression coefficients from PLSRegression.
    - rf/xgb: MDI-based `feature_importances_`.
    - hgbm: NaN (sklearn HistGradientBoosting has no stable MDI attribute).
    """
    inner = _unwrap_ttr(fitted)
    p = len(feature_names)

    if model_name in ("ridge", "lasso", "elasticnet"):
        coef = inner.named_steps["est"].coef_
        return pd.DataFrame({
            "feature": feature_names,
            "builtin_importance": np.abs(coef),
            "builtin_signed": coef,
        })
    if model_name == "pls":
        coef = inner.named_steps["est"].coef_.ravel()
        return pd.DataFrame({
            "feature": feature_names,
            "builtin_importance": np.abs(coef),
            "builtin_signed": coef,
        })
    if model_name == "rf":
        imp = inner.named_steps["est"].feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "builtin_importance": imp,
            "builtin_signed": np.full(p, np.nan, dtype="float64"),
        })
    if model_name == "xgb":
        imp = inner.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "builtin_importance": imp,
            "builtin_signed": np.full(p, np.nan, dtype="float64"),
        })
    if model_name == "hgbm":
        return pd.DataFrame({
            "feature": feature_names,
            "builtin_importance": np.full(p, np.nan, dtype="float64"),
            "builtin_signed": np.full(p, np.nan, dtype="float64"),
        })
    raise ValueError(f"unknown model: {model_name}")


def fit_and_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    label: str | None = None,
    n_repeats: int = 5,
    random_state: int = 0,
    min_samples: int = 30,
    perm_n_jobs: int = -1,
) -> pd.DataFrame:
    """Refit on the full (rows where y is observed) data and return per-feature importance.

    Returns a DataFrame with columns:
        feature, builtin_importance, builtin_signed,
        perm_importance_mean, perm_importance_std

    Importance is a model attribution on the deployable model (trained on all
    available labelled rows); it is intentionally not split-aware. The CV
    metrics in `run_one` are the generalization story.
    """
    cols = ["feature", "builtin_importance", "builtin_signed",
            "perm_importance_mean", "perm_importance_std"]
    mask = y.notna()
    Xn, yn = X.loc[mask], y.loc[mask]
    n = int(len(yn))
    p = X.shape[1]
    if p == 0 or n < min_samples:
        return pd.DataFrame(columns=cols)

    feature_names = list(X.columns)
    model = make_model(model_name, label=label, n_features=p)
    try:
        model.fit(Xn.values, yn.values)
    except Exception:
        return pd.DataFrame(columns=cols)

    builtin = _builtin_importance(model_name, model, feature_names)

    try:
        perm = permutation_importance(
            model, Xn.values, yn.values,
            n_repeats=n_repeats, random_state=random_state,
            scoring="r2", n_jobs=perm_n_jobs,
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std
    except Exception:
        perm_mean = np.full(p, np.nan, dtype="float64")
        perm_std = np.full(p, np.nan, dtype="float64")

    return builtin.assign(
        perm_importance_mean=perm_mean,
        perm_importance_std=perm_std,
    )[cols]
