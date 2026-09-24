"""Stage 05b — cross-validation sweep and feature importance.

Reads the wide matrices produced by build_05_wide and runs the full
(label x config x model) CV sweep under two split strategies.

Outputs (data/processed/05_modeling/):
    cv_metrics.parquet                 — long-form metrics from the parent-aware
                                          5-fold sweep (D-2026-04-27-CV-PARENTAWARE).
    cv_oof_predictions.parquet         — long-form out-of-fold predictions
                                          for the 5-fold sweep.
    cv_metrics_loo.parquet             — long-form metrics from the per-N3
                                          parent-disjoint leave-one-out sweep
                                          (D-2026-04-30-CV-LOO-PARENT-DISJOINT).
    cv_oof_predictions_loo.parquet     — long-form OOF predictions for the
                                          LOO sweep (one row per N3 per
                                          (label x config x model) combo).
    feature_importance_long.parquet    — per (label x config x model x feature)
                                          built-in + permutation importances
                                          (D-2026-04-27-IMPORTANCE). Importance
                                          is split-agnostic (refits on all
                                          observed rows) — same answer for both
                                          CV strategies, computed once.

The (label x config x model) sweep is dispatched in parallel via joblib.
Permutation importance inside each worker uses `n_jobs=1` to avoid nested
parallelism (process-pool x thread-pool oversubscription).

Run: `python -m prophet_ab.pipelines.build_05_cv`
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .. import paths
from ..modeling import configs, cv


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  wrote {out.relative_to(paths.REPO_ROOT)}  shape={df.shape}")


def _run_combo(
    label_col: str,
    cfg_name: str,
    model_name: str,
    X_full: pd.DataFrame,
    y: pd.Series,
    splits: list,
    *,
    compute_importance: bool,
) -> tuple[cv.CVResult, pd.DataFrame, pd.DataFrame]:
    """One (label, config, model) cell of the sweep."""
    cfg_fn = configs.CONFIGS[cfg_name]
    cols = cfg_fn(list(X_full.columns), label_col)
    X = X_full[cols] if cols else X_full.iloc[:, :0]
    res, oof = cv.run_one(
        X, y, splits=splits,
        label=label_col, config=cfg_name, model_name=model_name,
    )
    if compute_importance and pd.notna(res.spearman_rho):
        imp = cv.fit_and_importance(
            X, y, model_name=model_name, label=label_col, perm_n_jobs=1,
        )
    else:
        imp = pd.DataFrame()
    return res, oof, imp


def _sweep(
    X_full: pd.DataFrame,
    Y: pd.DataFrame,
    splits: list,
    *,
    compute_importance: bool,
    label_tag: str,
    n_jobs: int = -1,
) -> tuple[list[cv.CVResult], list[pd.DataFrame], list[pd.DataFrame]]:
    """Run the (label x config x model) sweep against one set of splits."""
    combos = [
        (label_col, cfg_name, model_name)
        for label_col, cfg_name, model_name in product(
            Y.columns, configs.CONFIGS.keys(), cv.MODELS,
        )
    ]
    print(f"  [{label_tag}] dispatching {len(combos)} combos to joblib (n_jobs={n_jobs})")
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_run_combo)(
            label_col, cfg_name, model_name,
            X_full, Y[label_col], splits,
            compute_importance=compute_importance,
        )
        for (label_col, cfg_name, model_name) in combos
    )
    results = [r for r, _, _ in out]
    oof_frames = []
    importance_frames = []
    for (label_col, cfg_name, model_name), (_res, oof, imp) in zip(combos, out):
        if not oof.empty:
            oof_frames.append(
                oof.assign(label=label_col, config=cfg_name, model=model_name)
            )
        if not imp.empty:
            importance_frames.append(
                imp.assign(label=label_col, config=cfg_name, model=model_name)
            )
    return results, oof_frames, importance_frames


_OOF_COLS = ["label", "config", "model", "antibody_name", "y_true", "y_pred"]
_IMPORTANCE_COLS = [
    "label", "config", "model", "feature",
    "builtin_importance", "builtin_signed",
    "perm_importance_mean", "perm_importance_std",
]


def _concat_with_cols(frames: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    if frames:
        return pd.concat(frames, ignore_index=True)[cols]
    return pd.DataFrame(columns=cols)


def main() -> None:
    out = paths.S05

    print("[1/3] loading wide matrices")
    feats = pd.read_parquet(out / "n3_features_wide.parquet").set_index("antibody_name")
    labels_w = pd.read_parquet(out / "n3_labels_wide.parquet").set_index("antibody_name")
    components = pd.read_parquet(paths.S02 / "n3_components.parquet")

    common = feats.index.intersection(labels_w.index)
    X_full = feats.loc[common]
    Y = labels_w.loc[common]

    kfold_splits = cv.build_parent_aware_splits(list(common), components)
    _n_test = sum(len(te) for _, te in kfold_splits)
    _n_train_per_fold = [len(tr) for tr, _ in kfold_splits]
    print(f"  parent-aware k-fold splits (D-2026-04-27-CV-PARENTAWARE): "
          f"{len(kfold_splits)} folds, total OOF rows={_n_test}, "
          f"train sizes per fold={_n_train_per_fold}")

    loo_splits = cv.build_parent_disjoint_loo_splits(list(common), components)
    _loo_train_sizes = [len(tr) for tr, _ in loo_splits]
    print(f"  parent-disjoint LOO splits (D-2026-04-30-CV-LOO-PARENT-DISJOINT): "
          f"{len(loo_splits)} folds (one per N3), "
          f"train sizes min={min(_loo_train_sizes)} "
          f"median={int(pd.Series(_loo_train_sizes).median())} "
          f"max={max(_loo_train_sizes)}")

    n_combos = len(Y.columns) * len(configs.CONFIGS) * len(cv.MODELS)
    print(f"[2/3] CV sweep — parent-aware k-fold: {n_combos} combos "
          f"({len(Y.columns)} labels x {len(configs.CONFIGS)} configs x {len(cv.MODELS)} models)")
    kfold_results, kfold_oof_frames, importance_frames = _sweep(
        X_full, Y, kfold_splits,
        compute_importance=True, label_tag="kfold",
    )
    metrics = cv.results_to_frame(kfold_results)
    print(f"  cv_metrics: {metrics.shape}, "
          f"{metrics['spearman_rho'].notna().sum()} non-empty rows")
    _write(metrics, out / "cv_metrics.parquet")

    oof_all = _concat_with_cols(kfold_oof_frames, _OOF_COLS)
    print(f"  cv_oof_predictions: {oof_all.shape}")
    _write(oof_all, out / "cv_oof_predictions.parquet")

    importance = _concat_with_cols(importance_frames, _IMPORTANCE_COLS)
    print(f"  feature_importance_long: {importance.shape}")
    _write(importance, out / "feature_importance_long.parquet")

    print(f"[3/3] CV sweep — parent-disjoint LOO: {n_combos} combos")
    loo_results, loo_oof_frames, _ = _sweep(
        X_full, Y, loo_splits,
        compute_importance=False, label_tag="loo",
    )
    loo_metrics = cv.results_to_frame(loo_results)
    print(f"  cv_metrics_loo: {loo_metrics.shape}, "
          f"{loo_metrics['spearman_rho'].notna().sum()} non-empty rows")
    _write(loo_metrics, out / "cv_metrics_loo.parquet")

    loo_oof_all = _concat_with_cols(loo_oof_frames, _OOF_COLS)
    print(f"  cv_oof_predictions_loo: {loo_oof_all.shape}")
    _write(loo_oof_all, out / "cv_oof_predictions_loo.parquet")

    print("done.")


if __name__ == "__main__":
    main()
