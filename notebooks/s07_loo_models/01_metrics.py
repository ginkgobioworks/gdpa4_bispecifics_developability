import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from prophet_ab import paths, schema
    from prophet_ab.features.naming import display_config, display_label
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        get_colors_from_cmap,
        equalize_axes,
        grid_figsize,
        wrap_title,
        wrap_label,
        FONT_SIZE_TICK,
        FONT_SIZE_LABEL,
        FONT_SIZE_TITLE,
        FONT_SIZE_LEGEND,
        FONT_SIZE_LEGEND_TITLE,
    )

    set_manuscript_style()
    return (
        DATAPOINTS_COLORS,
        display_config,
        display_label,
        equalize_axes,
        get_colors_from_cmap,
        grid_figsize,
        mo,
        np,
        paths,
        pd,
        plt,
        schema,
        wrap_title,
    )


@app.cell
def _(mo):
    mo.md("""
    # s07 — Leave-one-bispecific-out predictive models

    Per-N3 leave-one-out CV (parent exclusion enforced). Mirror of s05 but
    each (label × config × model) cell now sees up to 160 held-out
    predictions instead of ~27.

    Decisions: D-2026-04-30-CV-LOO-PARENT-DISJOINT, D-2026-04-27-CV-PARENTAWARE,
    D-2026-04-27-MODELS, D-2026-04-29-TARGET-TRANSFORMS.
    """)
    return


@app.cell
def _(display_label, paths, pd):
    m = pd.read_parquet(paths.S05 / "cv_metrics_loo.parquet")
    m = m.assign(label_short=m["label"].map(display_label))
    m
    return (m,)


@app.cell
def _():
    from collections import namedtuple

    MetricSpec = namedtuple(
        "MetricSpec", ["key", "col", "display", "ascending", "vmin", "vmax", "fmt"]
    )

    METRIC_SPECS = {
        "spearman": MetricSpec(
            "spearman", "spearman_rho", "Spearman ρ", False, -0.2, 1.0, ".2f"
        ),
        "pearson": MetricSpec(
            "pearson", "pearson_r", "Pearson r", False, -0.2, 1.0, ".2f"
        ),
        "mse": MetricSpec("mse", "mse", "MSE", True, None, None, ".2g"),
        "rmse": MetricSpec("rmse", "rmse", "RMSE", True, None, None, ".3g"),
    }

    TABLE_COLS = [
        "label_short", "config", "model", "n_features", "n_samples",
        "spearman_rho", "pearson_r", "r2", "mse", "rmse", "mae",
    ]

    CFG_ORDER = [
        "compositional_baseline",
        "corresponding_experimental",
        "all_experimental",
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    ]

    ARM_CFG_ORDER = [
        "compositional_baseline",
        "arm_corresponding",
        "arm_all_experimental",
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    ]

    SECTION = "s07"
    CV_LABEL = "LOO"

    def save_best_table(m, spec, paths):
        best = (
            m.dropna(subset=[spec.col])
             .sort_values(spec.col, ascending=spec.ascending)
             .groupby("label").head(1)
             .sort_values(spec.col, ascending=spec.ascending)
             [TABLE_COLS]
        )
        _o = paths.TABLES / f"{SECTION}_best_per_label_{spec.key}.csv"
        _o.parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(_o, index=False)
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return best

    def save_summary_table(m, spec, paths):
        summary = (
            m.groupby(["config", "model"])
             .agg(
                n_labels=("label", "nunique"),
                spearman_mean=("spearman_rho", "mean"),
                spearman_median=("spearman_rho", "median"),
                pearson_mean=("pearson_r", "mean"),
                pearson_median=("pearson_r", "median"),
                r2_median=("r2", "median"),
                mse_median=("mse", "median"),
                rmse_median=("rmse", "median"),
                mae_median=("mae", "median"),
             )
             .round(3)
             .sort_values(f"{spec.key}_median", ascending=spec.ascending)
        )
        _o = paths.TABLES / f"{SECTION}_config_summary_{spec.key}.csv"
        _o.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(_o)
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return summary

    def plot_heatmap(m, spec, paths, np, plt, display_config):
        _pivot = m.assign(cm=m["config"].map(display_config) + "\n" + m["model"]).pivot_table(
            index="label_short", columns="cm", values=spec.col
        )
        _sort_asc = spec.ascending
        _pivot = _pivot.loc[
            _pivot.median(axis=1).sort_values(ascending=_sort_asc).index
        ]

        _vmin = spec.vmin if spec.vmin is not None else np.nanmin(_pivot.values)
        _vmax = spec.vmax if spec.vmax is not None else np.nanmax(_pivot.values)
        _cmap = "viridis_r" if spec.ascending else "viridis"
        _mid = (_vmin + _vmax) / 2

        fig, _ax = plt.subplots(
            figsize=(max(10, 1.0 * len(_pivot.columns) + 2), 8),
            layout="constrained",
        )
        _im = _ax.imshow(
            _pivot.values, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax
        )
        _ax.set_xticks(range(len(_pivot.columns)))
        _ax.set_xticklabels(_pivot.columns, rotation=45, ha="right")
        _ax.set_yticks(range(len(_pivot.index)))
        _ax.set_yticklabels(_pivot.index)
        for _i in range(_pivot.shape[0]):
            for _j in range(_pivot.shape[1]):
                _v = _pivot.values[_i, _j]
                if not np.isnan(_v):
                    _color = "white" if (
                        (_v < _mid and not spec.ascending) or
                        (_v > _mid and spec.ascending)
                    ) else "black"
                    _ax.text(
                        _j, _i, f"{_v:{spec.fmt}}", ha="center", va="center",
                        color=_color, fontsize=6,
                    )
        fig.colorbar(
            _im, ax=_ax, label=f"{spec.display} ({CV_LABEL})", shrink=0.7
        )
        _ax.set_title(
            f"Out-of-fold {spec.display} across labels x (config, model) "
            f"-- parent-disjoint {CV_LABEL}"
        )

        _o = paths.FIGURES / f"{SECTION}_{spec.key}_heatmap.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    def plot_bar_chart(m, spec, paths, np, plt, display_config,
                       cfg_order=None, suffix=""):
        if cfg_order is None:
            cfg_order = CFG_ORDER
        _agg = "min" if spec.ascending else "max"
        _md = m.assign(config_display=m["config"].map(display_config))
        _bpc = _md.groupby(["label_short", "config_display"])[spec.col].agg(_agg).unstack()
        _cfg_display_order = [display_config(c) for c in cfg_order]
        _bpc = _bpc[[c for c in _cfg_display_order if c in _bpc.columns]]
        _bpc = _bpc.loc[
            _bpc.median(axis=1).sort_values(ascending=spec.ascending).index
        ]

        fig, _ax = plt.subplots(figsize=(10, 6), layout="constrained")
        _x = np.arange(len(_bpc.index))
        _w = 0.85 / len(_bpc.columns)
        for _i, _cfg in enumerate(_bpc.columns):
            _ax.bar(_x + _i * _w, _bpc[_cfg].values, width=_w, label=_cfg)
        _ax.set_xticks(_x + _w * (len(_bpc.columns) - 1) / 2)
        _ax.set_xticklabels(_bpc.index, rotation=60, ha="right")
        _ax.set_ylabel(f"{spec.display} (best model, {CV_LABEL})")
        _ax.axhline(0, color="k", lw=0.5)
        _ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", ncols=2)
        _ax.set_title(
            f"Per-label {spec.display} across feature configs\n"
            f"parent-disjoint {CV_LABEL} (best model per config)"
        )

        _o = paths.FIGURES / f"{SECTION}_{spec.key}_comparison{suffix}.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    def plot_oof_scatter(m, oof, spec, paths, np, plt, display_config,
                         equalize_axes=None, DATAPOINTS_COLORS=None,
                         grid_figsize=None, wrap_title=None):
        _best = (
            m.dropna(subset=[spec.col])
             .sort_values(spec.col, ascending=spec.ascending)
             .groupby("label").head(1)
             .sort_values(spec.col, ascending=spec.ascending)
             [["label", "label_short", "config", "model", spec.col,
               "n_test_predictions"]]
             .reset_index(drop=True)
        )

        _ncols = 3
        _nrows = (len(_best) + _ncols - 1) // _ncols
        if grid_figsize is not None:
            _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
        else:
            _fw = 5 * _ncols / 1.4
            _fh = 5 * _nrows / 1.4
            _cw = _fw / _ncols
        fig, _axes = plt.subplots(
            _nrows, _ncols, figsize=(_fw, _fh),
            layout="constrained",
        )
        _axes = np.atleast_1d(_axes).flatten()

        _edge_kw = {}
        if DATAPOINTS_COLORS is not None:
            _edge_kw = dict(edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3)

        for _ax, (_, _row) in zip(_axes, _best.iterrows()):
            _sub = oof[
                (oof["label"] == _row["label"])
                & (oof["config"] == _row["config"])
                & (oof["model"] == _row["model"])
            ]
            _yt = _sub["y_true"].to_numpy()
            _yp = _sub["y_pred"].to_numpy()
            _ax.scatter(_yt, _yp, s=14, alpha=0.7, **_edge_kw)
            if len(_yt) > 0:
                _lo = float(np.nanmin([_yt.min(), _yp.min()]))
                _hi = float(np.nanmax([_yt.max(), _yp.max()]))
                _pad = 0.05 * (_hi - _lo) if _hi > _lo else 1.0
                _ax.plot(
                    [_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad],
                    ls="--", lw=0.8, color="k",
                )
                _ax.set_xlim(_lo - _pad, _hi + _pad)
                _ax.set_ylim(_lo - _pad, _hi + _pad)
            if equalize_axes is not None:
                equalize_axes(_ax)
            _metric_val = _row[spec.col]
            _title_text = (
                f"{_row['label_short']}\n{display_config(_row['config'])} / {_row['model']}  "
                f"{spec.display}={_metric_val:{spec.fmt}}  "
                f"n={int(_row['n_test_predictions'])}"
            )
            _ax.set_title(_title_text, fontsize=7)
            _ax.tick_params(labelsize=6)
            _ax.set_xlabel("experimental", fontsize=7)
            _ax.set_ylabel(f"predicted ({CV_LABEL})", fontsize=7)

        for _ax in _axes[len(_best):]:
            _ax.axis("off")

        _suptitle_text = (
            f"{CV_LABEL} predicted vs experimental -- best (config x model) per label "
            f"by {spec.display} (parent-disjoint leave-one-bispecific-out)"
        )
        fig.suptitle(_suptitle_text, fontsize=9)

        _o = paths.FIGURES / f"{SECTION}_oof_scatter_{spec.key}.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    return (
        ARM_CFG_ORDER,
        METRIC_SPECS,
        plot_bar_chart,
        plot_heatmap,
        plot_oof_scatter,
        save_best_table,
        save_summary_table,
    )


@app.cell
def _(paths, pd):
    oof = pd.read_parquet(paths.S05 / "cv_oof_predictions_loo.parquet")
    return (oof,)


@app.cell
def _(
    ARM_CFG_ORDER,
    DATAPOINTS_COLORS,
    METRIC_SPECS,
    display_config,
    equalize_axes,
    grid_figsize,
    m,
    mo,
    np,
    oof,
    paths,
    plot_bar_chart,
    plot_heatmap,
    plot_oof_scatter,
    plt,
    save_best_table,
    save_summary_table,
    wrap_title,
):
    _spec = METRIC_SPECS["spearman"]
    _best = save_best_table(m, _spec, paths)
    _summary = save_summary_table(m, _spec, paths)
    _fig_h = plot_heatmap(m, _spec, paths, np, plt, display_config)
    _fig_b = plot_bar_chart(m, _spec, paths, np, plt, display_config)
    _fig_b_arm = plot_bar_chart(m, _spec, paths, np, plt, display_config,
                                cfg_order=ARM_CFG_ORDER, suffix="_arm")
    _fig_s = plot_oof_scatter(m, oof, _spec, paths, np, plt, display_config,
                              equalize_axes=equalize_axes,
                              DATAPOINTS_COLORS=DATAPOINTS_COLORS,
                              grid_figsize=grid_figsize,
                              wrap_title=wrap_title)
    mo.vstack([
        mo.md(f"## {_spec.display}"),
        _best, _summary, _fig_h, _fig_b, _fig_b_arm, _fig_s,
    ])
    return


@app.cell
def _(display_config, m, np, paths, plt):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _CFG_ORDER = [
        "compositional_baseline",
        "corresponding_experimental",
        "all_experimental",
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    ]
    _ms = m[m["label_short"].isin(_SUBSET_LABELS)]
    _md = _ms.assign(config_display=_ms["config"].map(display_config))
    _bpc = _md.groupby(["label_short", "config_display"])["spearman_rho"].max().unstack()
    _cfg_display_order = [display_config(c) for c in _CFG_ORDER]
    _bpc = _bpc[[c for c in _cfg_display_order if c in _bpc.columns]]
    _bpc = _bpc.loc[_bpc.median(axis=1).sort_values(ascending=False).index]

    fig_subset, _ax = plt.subplots(figsize=(8, 5), layout="constrained")
    _x = np.arange(len(_bpc.index))
    _w = 0.85 / len(_bpc.columns)
    for _i, _cfg in enumerate(_bpc.columns):
        _ax.bar(_x + _i * _w, _bpc[_cfg].values, width=_w, label=_cfg)
    _ax.set_xticks(_x + _w * (len(_bpc.columns) - 1) / 2)
    _ax.set_xticklabels([_TICK_LABELS.get(l, l) for l in _bpc.index], rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (best model, LOO)")
    _ax.axhline(0, color="k", lw=0.5)
    _ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    _ax.set_title(
        "Per-label Spearman ρ across feature configs\n"
        "-- parent-disjoint LOO (best model per config)",
    )

    _o = paths.FIGURES / "s07_spearman_comparison_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_subset.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_subset
    return


@app.cell
def _(display_config, m, np, paths, plt):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _ARM_CFG_ORDER = [
        "compositional_baseline",
        "arm_corresponding",
        "arm_all_experimental",
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    ]
    _ms = m[m["label_short"].isin(_SUBSET_LABELS)]
    _md = _ms.assign(config_display=_ms["config"].map(display_config))
    _bpc = _md.groupby(["label_short", "config_display"])["spearman_rho"].max().unstack()
    _cfg_display_order = [display_config(c) for c in _ARM_CFG_ORDER]
    _bpc = _bpc[[c for c in _cfg_display_order if c in _bpc.columns]]
    _bpc = _bpc.loc[_bpc.median(axis=1).sort_values(ascending=False).index]

    fig_subset_arm, _ax = plt.subplots(figsize=(8, 5), layout="constrained")
    _x = np.arange(len(_bpc.index))
    _w = 0.85 / len(_bpc.columns)
    for _i, _cfg in enumerate(_bpc.columns):
        _ax.bar(_x + _i * _w, _bpc[_cfg].values, width=_w, label=_cfg)
    _ax.set_xticks(_x + _w * (len(_bpc.columns) - 1) / 2)
    _ax.set_xticklabels([_TICK_LABELS.get(l, l) for l in _bpc.index], rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (best model, LOO)")
    _ax.axhline(0, color="k", lw=0.5)
    _ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    _ax.set_title(
        "Per-label Spearman ρ across arm feature configs\n"
        "-- parent-disjoint LOO (best model per config)",
    )

    _o = paths.FIGURES / "s07_spearman_comparison_subset_arm.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_subset_arm.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_subset_arm
    return


@app.cell
def _(display_label, paths, pd):
    from prophet_ab.features import naming
    from prophet_ab.features.naming import display_feature

    imp = pd.read_parquet(paths.S05 / "feature_importance_long.parquet")
    _parsed = imp["feature"].map(naming.parse_feature_name)
    imp = imp.assign(
        kind=_parsed.map(lambda p: p.kind),
        source=_parsed.map(lambda p: p.source),
        label_short=lambda d: d["label"].map(display_label),
        abs_perm=lambda d: d["perm_importance_mean"].abs(),
    )
    return display_feature, imp


@app.cell
def _(
    DATAPOINTS_COLORS,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _ms = m[m["label_short"].isin(_SUBSET_LABELS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _kind_color = {"is": DATAPOINTS_COLORS["blue"], "meas": DATAPOINTS_COLORS["purple"]}
    fig_imp_subset, _axes = plt.subplots(1, len(_best), figsize=(5 * len(_best) / 1.4, 5 / 1.4),
                                         layout="constrained")
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [_kind_color.get(k, "gray") for k in _sl["kind"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=6)
        _disp = _TICK_LABELS.get(_row['label_short'], _row['label_short'])
        _ax.set_title(
            f"{_disp}\n{display_config(_row['config'])} / {_row['model']}"
            f" / rho={_row['spearman_rho']:.2f}",
            fontsize=7,
        )
        _ax.tick_params(axis="x", labelsize=6)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[len(_best):]:
        _ax.axis("off")

    _kind_labels = {"is": "In Silico", "meas": "Measured"}
    _handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _kind_color.values()]
    fig_imp_subset.legend(_handles, [_kind_labels[k] for k in _kind_color],
                          loc="outside right upper")
    fig_imp_subset.suptitle(
        "Top permutation-importance features -- subset labels\n"
        "(best config x model by LOO Spearman ρ)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_subset.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_subset
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m[m["label_short"].isin(_SUBSET_LABELS) & m["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _kind_color = {"arm_is": DATAPOINTS_COLORS["blue"], "arm_meas": DATAPOINTS_COLORS["purple"]}
    _nrows = max(1, len(_best))
    fig_imp_subset_arm, _axes = plt.subplots(_nrows, 1, figsize=(3.75, 2.25 * _nrows),
                                             layout="constrained")
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [_kind_color.get(k, "gray") for k in _sl["kind"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=6)
        _disp = _TICK_LABELS.get(_row['label_short'], _row['label_short'])
        _ax.set_title(
            f"{_disp}\n{display_config(_row['config'])} / {_row['model']}"
            f" / rho={_row['spearman_rho']:.2f}",
            fontsize=7,
        )
        _ax.tick_params(axis="x", labelsize=6)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_nrows:]:
        _ax.axis("off")

    _kind_labels = {"arm_is": "Arm In Silico", "arm_meas": "Arm Measured"}
    _handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _kind_color.values()]
    fig_imp_subset_arm.legend(_handles, [_kind_labels[k] for k in _kind_color],
                              loc="outside lower center")
    fig_imp_subset_arm.suptitle(
        "Top permutation-importance features -- subset labels\n"
        "(best arm config x model by LOO Spearman ρ)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_subset_arm.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_subset_arm.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_subset_arm
    return


@app.cell
def _(DATAPOINTS_COLORS):
    from prophet_ab.features.naming import parse_feature_name, parse_label_name

    FEATURE_GROUP_COLORS = {
        "Same Assay": DATAPOINTS_COLORS["navy"],
        "Same Assay Diff Condition": DATAPOINTS_COLORS["blue"],
        "Different Assay": DATAPOINTS_COLORS["teal"],
        "In-silico": DATAPOINTS_COLORS["amber"],
    }
    FEATURE_GROUP_ORDER = list(FEATURE_GROUP_COLORS.keys())

    def classify_feature(feature_col: str, label_col: str) -> str:
        pf = parse_feature_name(feature_col)
        if pf.kind in ("is", "arm_is"):
            return "In-silico"
        lbl_vc, lbl_cond = parse_label_name(label_col)
        if pf.value_col == lbl_vc:
            if pf.condition == lbl_cond:
                return "Same Assay"
            return "Same Assay Diff Condition"
        return "Different Assay"

    return FEATURE_GROUP_COLORS, FEATURE_GROUP_ORDER, classify_feature


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m[m["label_short"].isin(_SUBSET_LABELS) & m["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ncols = len(_best)
    fig_imp_arm_groups, _axes = plt.subplots(1, _ncols, figsize=(0.75 * 5 * _ncols / 1.4, 0.75 * 5 / 1.4),
                                             layout="constrained")
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=6)
        _disp = _TICK_LABELS.get(_row['label_short'], _row['label_short'])
        _ax.set_title(
            f"{_disp}\n{display_config(_row['config'])} / {_row['model']}"
            f" / rho={_row['spearman_rho']:.2f}",
            fontsize=7,
        )
        _ax.tick_params(axis="x", labelsize=6)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ncols:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_arm_groups.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside right upper",
    )
    fig_imp_arm_groups.suptitle(
        "Top permutation-importance features -- subset labels\n"
        "(best arm config x model by LOO Spearman ρ)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_subset_arm_groups.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_arm_groups.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_arm_groups
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m[m["label_short"].isin(_SUBSET_LABELS) & m["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ncols = len(_best)
    fig_imp_nonarm_groups, _axes = plt.subplots(1, _ncols, figsize=(0.75 * 5 * _ncols / 1.4, 0.75 * 5 / 1.4),
                                                layout="constrained")
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=6)
        _disp = _TICK_LABELS.get(_row['label_short'], _row['label_short'])
        _ax.set_title(
            f"{_disp}\n{display_config(_row['config'])} / {_row['model']}"
            f" / rho={_row['spearman_rho']:.2f}",
            fontsize=7,
        )
        _ax.tick_params(axis="x", labelsize=6)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ncols:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_nonarm_groups.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside right upper",
    )
    fig_imp_nonarm_groups.suptitle(
        "Top permutation-importance features -- subset labels\n"
        "(best operator config x model by LOO Spearman ρ)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_subset_groups.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_nonarm_groups.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_nonarm_groups
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m[m["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best_all = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ntotal = len(_best_all)
    _grid_cols = 3
    _grid_rows = (_ntotal + _grid_cols - 1) // _grid_cols
    _pw, _ph = 3.0, 2.5
    fig_imp_all_groups, _axes = plt.subplots(
        _grid_rows, _grid_cols, figsize=(_pw * _grid_cols, _ph * _grid_rows),
        layout="constrained",
    )
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best_all.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=5)
        _ax.set_title(
            f"{_row['label_short']}\n{display_config(_row['config'])} · {_row['model']}"
            f" · ρ={_row['spearman_rho']:.2f}",
            fontsize=6,
        )
        _ax.tick_params(axis="x", labelsize=5)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ntotal:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_all_groups.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside lower center", ncols=4,
    )
    fig_imp_all_groups.suptitle(
        "Top permutation-importance features — all labels\n"
        "(best arm config × model by LOO Spearman ρ, ordered by performance)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_all_arm_groups.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_all_groups.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_all_groups
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m[m["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best_all = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ntotal = len(_best_all)
    _grid_cols = 3
    _grid_rows = (_ntotal + _grid_cols - 1) // _grid_cols
    _pw, _ph = 3.0, 2.5
    fig_imp_all_nonarm_groups, _axes = plt.subplots(
        _grid_rows, _grid_cols, figsize=(_pw * _grid_cols, _ph * _grid_rows),
        layout="constrained",
    )
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best_all.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=5)
        _ax.set_title(
            f"{_row['label_short']}\n{display_config(_row['config'])} · {_row['model']}"
            f" · ρ={_row['spearman_rho']:.2f}",
            fontsize=6,
        )
        _ax.tick_params(axis="x", labelsize=5)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ntotal:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_all_nonarm_groups.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside lower center", ncols=4,
    )
    fig_imp_all_nonarm_groups.suptitle(
        "Top permutation-importance features — all labels\n"
        "(best operator config × model by LOO Spearman ρ, ordered by performance)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_all_groups.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_all_nonarm_groups.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_all_nonarm_groups
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp,
    m,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m[m["label_short"].isin(_SUBSET_LABELS) & m["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_subset, _ax = plt.subplots(figsize=(1.9 * _n, 3.75),
                                           layout="constrained")

    _rev_order = list(reversed(FEATURE_GROUP_ORDER))
    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.6,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            if _frac > 0.05:
                _ax.text(_i, _bottom + _h / 2, f"{_frac:.0%}",
                         ha="center", va="center", fontsize=6, color="white", fontweight="bold")
            _bottom += _h

    _tick_labels_sub = [_TICK_LABELS.get(r['label_short'], r['label_short'])
                        for _, r in _best.iterrows()]
    _ax.set_xticks(range(_n))
    _ax.set_xticklabels(_tick_labels_sub)
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_subset.suptitle(
        "Performance & importance breakdown by feature group\n"
        "subset labels (best arm config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_subset_arm.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_subset.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_subset
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp,
    m,
    paths,
    plt,
):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS His/NaCl pH 6",
        "PR Score @ CHO": "PR-CHO",
    }
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m[m["label_short"].isin(_SUBSET_LABELS) & m["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_subset_nonarm, _ax = plt.subplots(figsize=(1.9 * _n, 3.75),
                                                  layout="constrained")

    _rev_order = list(reversed(FEATURE_GROUP_ORDER))
    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.6,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            if _frac > 0.05:
                _ax.text(_i, _bottom + _h / 2, f"{_frac:.0%}",
                         ha="center", va="center", fontsize=6, color="white", fontweight="bold")
            _bottom += _h

    _tick_labels_sub = [_TICK_LABELS.get(r['label_short'], r['label_short'])
                        for _, r in _best.iterrows()]
    _ax.set_xticks(range(_n))
    _ax.set_xticklabels(_tick_labels_sub)
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_subset_nonarm.suptitle(
        "Performance & importance breakdown by feature group\n"
        "subset labels (best operator config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_subset_nonarm.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_subset_nonarm
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m[m["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_all, _ax = plt.subplots(figsize=(max(6, 0.55 * _n), 3.75),
                                        layout="constrained")

    _x_positions = np.arange(_n)
    _tick_labels_all = []
    _rev_order = list(reversed(FEATURE_GROUP_ORDER))

    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.7,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            _bottom += _h

        _tick_labels_all.append(_row['label_short'])

    _ax.set_xticks(_x_positions)
    _ax.set_xticklabels(_tick_labels_all, rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)
    _ax.axhline(0, color="k", lw=0.4)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_all.suptitle(
        "Performance & importance breakdown by feature group\n"
        "all labels (best arm config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_all_arm.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_all.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_all
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp,
    m,
    np,
    paths,
    plt,
):
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m[m["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_all_nonarm, _ax = plt.subplots(figsize=(max(6, 0.55 * _n), 3.75),
                                               layout="constrained")

    _x_positions = np.arange(_n)
    _tick_labels_all = []
    _rev_order = list(reversed(FEATURE_GROUP_ORDER))

    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.7,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            _bottom += _h

        _tick_labels_all.append(_row['label_short'])

    _ax.set_xticks(_x_positions)
    _ax.set_xticklabels(_tick_labels_all, rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)
    _ax.axhline(0, color="k", lw=0.4)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_all_nonarm.suptitle(
        "Performance & importance breakdown by feature group\n"
        "all labels (best operator config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_all.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_all_nonarm.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_all_nonarm
    return


@app.cell
def _(display_config, display_feature, imp, m, paths, pd):
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _ms = m[m["label_short"].isin(_SUBSET_LABELS)].dropna(subset=["spearman_rho"])
    _best_per_cfg = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby(["label", "config"]).head(1)
           [["label", "label_short", "config", "model", "spearman_rho"]]
    )

    _rows = []
    for _, _row in _best_per_cfg.iterrows():
        _sl = imp[(imp["label"] == _row["label"])
                  & (imp["config"] == _row["config"])
                  & (imp["model"] == _row["model"])]
        _top = _sl.sort_values("abs_perm", ascending=False).head(10).copy()
        _top = _top.assign(
            config_display=display_config(_row["config"]),
            best_model=_row["model"],
            loo_spearman=_row["spearman_rho"],
            feature_display=[display_feature(f) for f in _top["feature"]],
        )
        _rows.append(_top)

    _table = pd.concat(_rows) if _rows else pd.DataFrame()
    if not _table.empty:
        _table = _table[
            ["label_short", "config", "config_display", "best_model",
             "loo_spearman", "feature", "feature_display", "kind", "source",
             "perm_importance_mean", "perm_importance_std",
             "builtin_importance", "builtin_signed"]
        ].round(4)

    _o = paths.TABLES / "s07_top_features_subset.csv"
    _o.parent.mkdir(parents=True, exist_ok=True)
    _table.to_csv(_o, index=False)
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}  rows={len(_table)}")
    _table.head(20)
    return


@app.cell
def _(
    ARM_CFG_ORDER,
    DATAPOINTS_COLORS,
    METRIC_SPECS,
    display_config,
    equalize_axes,
    grid_figsize,
    m,
    mo,
    np,
    oof,
    paths,
    plot_bar_chart,
    plot_heatmap,
    plot_oof_scatter,
    plt,
    save_best_table,
    save_summary_table,
    wrap_title,
):
    _spec = METRIC_SPECS["mse"]
    _best = save_best_table(m, _spec, paths)
    _summary = save_summary_table(m, _spec, paths)
    _fig_h = plot_heatmap(m, _spec, paths, np, plt, display_config)
    _fig_b = plot_bar_chart(m, _spec, paths, np, plt, display_config)
    _fig_b_arm = plot_bar_chart(m, _spec, paths, np, plt, display_config,
                                cfg_order=ARM_CFG_ORDER, suffix="_arm")
    _fig_s = plot_oof_scatter(m, oof, _spec, paths, np, plt, display_config,
                              equalize_axes=equalize_axes,
                              DATAPOINTS_COLORS=DATAPOINTS_COLORS,
                              grid_figsize=grid_figsize,
                              wrap_title=wrap_title)
    mo.vstack([
        mo.md(f"## {_spec.display}"),
        _best, _summary, _fig_h, _fig_b, _fig_b_arm, _fig_s,
    ])
    return


@app.cell
def _(
    ARM_CFG_ORDER,
    DATAPOINTS_COLORS,
    METRIC_SPECS,
    display_config,
    equalize_axes,
    grid_figsize,
    m,
    mo,
    np,
    oof,
    paths,
    plot_bar_chart,
    plot_heatmap,
    plot_oof_scatter,
    plt,
    save_best_table,
    save_summary_table,
    wrap_title,
):
    _spec = METRIC_SPECS["rmse"]
    _best = save_best_table(m, _spec, paths)
    _summary = save_summary_table(m, _spec, paths)
    _fig_h = plot_heatmap(m, _spec, paths, np, plt, display_config)
    _fig_b = plot_bar_chart(m, _spec, paths, np, plt, display_config)
    _fig_b_arm = plot_bar_chart(m, _spec, paths, np, plt, display_config,
                                cfg_order=ARM_CFG_ORDER, suffix="_arm")
    _fig_s = plot_oof_scatter(m, oof, _spec, paths, np, plt, display_config,
                              equalize_axes=equalize_axes,
                              DATAPOINTS_COLORS=DATAPOINTS_COLORS,
                              grid_figsize=grid_figsize,
                              wrap_title=wrap_title)
    mo.vstack([
        mo.md(f"## {_spec.display}"),
        _best, _summary, _fig_h, _fig_b, _fig_b_arm, _fig_s,
    ])
    return


@app.cell
def _(
    ARM_CFG_ORDER,
    DATAPOINTS_COLORS,
    METRIC_SPECS,
    display_config,
    equalize_axes,
    grid_figsize,
    m,
    mo,
    np,
    oof,
    paths,
    plot_bar_chart,
    plot_heatmap,
    plot_oof_scatter,
    plt,
    save_best_table,
    save_summary_table,
    wrap_title,
):
    _spec = METRIC_SPECS["pearson"]
    _best = save_best_table(m, _spec, paths)
    _summary = save_summary_table(m, _spec, paths)
    _fig_h = plot_heatmap(m, _spec, paths, np, plt, display_config)
    _fig_b = plot_bar_chart(m, _spec, paths, np, plt, display_config)
    _fig_b_arm = plot_bar_chart(m, _spec, paths, np, plt, display_config,
                                cfg_order=ARM_CFG_ORDER, suffix="_arm")
    _fig_s = plot_oof_scatter(m, oof, _spec, paths, np, plt, display_config,
                              equalize_axes=equalize_axes,
                              DATAPOINTS_COLORS=DATAPOINTS_COLORS,
                              grid_figsize=grid_figsize,
                              wrap_title=wrap_title)
    mo.vstack([
        mo.md(f"## {_spec.display}"),
        _best, _summary, _fig_h, _fig_b, _fig_b_arm, _fig_s,
    ])
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    equalize_axes,
    get_colors_from_cmap,
    m,
    np,
    paths,
    pd,
    plt,
):
    kf = pd.read_parquet(paths.S05 / "cv_metrics.parquet")
    join = (
        m[["label", "label_short", "config", "model", "spearman_rho"]]
         .rename(columns={"spearman_rho": "rho_loo"})
         .merge(
             kf[["label", "config", "model", "spearman_rho"]]
              .rename(columns={"spearman_rho": "rho_kfold"}),
             on=["label", "config", "model"],
         )
         .dropna(subset=["rho_loo", "rho_kfold"])
    )

    fig_c, _ax = plt.subplots(figsize=(6, 6), layout="constrained")
    _models = sorted(join["model"].unique())
    _colors = get_colors_from_cmap(number_of_colors=len(_models))
    for _i, _mn in enumerate(_models):
        _sub = join[join["model"] == _mn]
        _ax.scatter(
            _sub["rho_kfold"], _sub["rho_loo"],
            s=22, alpha=0.7,
            edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
            color=_colors[_i], label=f"{_mn} (n={len(_sub)})",
        )
    _lo = float(np.nanmin([join["rho_kfold"].min(), join["rho_loo"].min()]))
    _hi = float(np.nanmax([join["rho_kfold"].max(), join["rho_loo"].max()]))
    _pad = 0.05 * (_hi - _lo) if _hi > _lo else 0.1
    _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], ls="--", lw=0.8, color="k")
    _ax.set_xlim(_lo - _pad, _hi + _pad)
    _ax.set_ylim(_lo - _pad, _hi + _pad)
    equalize_axes(_ax)
    _ax.set_xlabel("Spearman ρ -- parent-aware 5-fold")
    _ax.set_ylabel("Spearman ρ -- parent-disjoint LOO")
    _r = float(join[["rho_kfold", "rho_loo"]].corr().iloc[0, 1])
    _ax.set_title(
        f"LOO vs k-fold Spearman ρ\n"
        f"(n={len(join)} combos, Pearson r={_r:.2f})",
    )
    _ax.legend(loc="lower right")
    _ax.grid(alpha=0.3)

    _o = paths.FIGURES / "s07_loo_vs_kfold_scatter.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_c
    return


@app.cell
def _(DATAPOINTS_COLORS, display_config, m, paths, plt, schema):
    import seaborn as sns

    _cfg_order = list(schema.CONFIG_DISPLAY.keys())
    _plot = m.dropna(subset=["spearman_rho"]).copy()
    _plot["config_label"] = _plot["config"].map(display_config)
    _plot["model_upper"] = _plot["model"].str.upper()
    _model_order = sorted(_plot["model_upper"].unique())

    _palette = [
        DATAPOINTS_COLORS["blue"],
        DATAPOINTS_COLORS["purple"],
        DATAPOINTS_COLORS["pink"],
        DATAPOINTS_COLORS["red"],
        DATAPOINTS_COLORS["teal"],
        DATAPOINTS_COLORS["gray"],
        DATAPOINTS_COLORS["amber"],
        DATAPOINTS_COLORS["navy"],
        DATAPOINTS_COLORS["green"],
        DATAPOINTS_COLORS["coral"],
        DATAPOINTS_COLORS["slate"],
    ]
    _hue_order = [display_config(c) for c in _cfg_order]

    fig_box, _ax = plt.subplots(figsize=(10, 5), layout="constrained")
    sns.boxenplot(
        data=_plot,
        x="model_upper",
        y="spearman_rho",
        hue="config_label",
        order=_model_order,
        hue_order=_hue_order,
        palette=_palette,
        linewidth=0.6,
        ax=_ax,
    )
    _ax.set_xlabel("Model")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.axhline(0, color="k", lw=0.5, ls="--")
    _ax.legend(
        title="Feature config",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    _ax.set_title(
        "Spearman ρ by model and feature config\n"
        "parent-disjoint LOO",
    )

    _o = paths.FIGURES / "s07_boxenplot_model_config.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_box.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_box
    return


@app.cell
def _(display_config, m, np, paths, plt, schema):
    from matplotlib.colors import LinearSegmentedColormap

    _red = "#C94040"
    _green = "#3A9A5C"
    _cmap = LinearSegmentedColormap.from_list(
        "rho_diverging", [_red, "white", _green], N=256
    )

    _spec_col = "spearman_rho"
    _hic = m[m["label_short"] == "HIC RT (norm)"].copy()

    _all_cfgs = [c for c in schema.CONFIG_DISPLAY if not c.startswith("arm_")]
    _hic = _hic[_hic["config"].isin(_all_cfgs)]
    _cfg_labels = [display_config(c) for c in _all_cfgs if c in _hic["config"].values]
    _cfg_map = {c: display_config(c) for c in _all_cfgs if c in _hic["config"].values}
    _model_order = sorted(_hic["model"].unique())

    _pivot = _hic.assign(
        config_display=_hic["config"].map(_cfg_map),
    ).pivot_table(index="model", columns="config_display", values=_spec_col)
    _pivot = _pivot[[c for c in _cfg_labels if c in _pivot.columns]]
    _pivot = _pivot.loc[[m for m in _model_order if m in _pivot.index]]

    _vmin, _vmax = -1.0, 1.0

    _cell = 0.4
    _ncols, _nrows = len(_pivot.columns), len(_pivot.index)
    fig_hic, _ax = plt.subplots(
        figsize=(_cell * _ncols + 3.5, _cell * _nrows + 2.5),
        layout="constrained",
    )
    _im = _ax.imshow(
        _pivot.values, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax
    )
    _ax.set_xticks(range(len(_pivot.columns)))
    _ax.set_xticklabels(_pivot.columns, rotation=45, ha="right")
    _ax.set_yticks(range(len(_pivot.index)))
    _ax.set_yticklabels(_pivot.index)
    for _i in range(_pivot.shape[0]):
        for _j in range(_pivot.shape[1]):
            _v = _pivot.values[_i, _j]
            if not np.isnan(_v):
                _lum = 0.299 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[0] * 255) \
                     + 0.587 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[1] * 255) \
                     + 0.114 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[2] * 255)
                _color = "black" if _lum > 140 else "white"
                _ax.text(
                    _j, _i, f"{_v:.2f}", ha="center", va="center",
                    color=_color, fontsize=8,
                )
    fig_hic.colorbar(_im, ax=_ax, label="Spearman ρ (LOO)", shrink=0.8)
    _ax.set_xlabel("Feature config")
    _ax.set_ylabel("Model")
    _ax.set_title(
        "HIC RT (norm) -- LOO Spearman ρ by feature config x model",
    )

    _o = paths.FIGURES / "s07_hic_spearman_heatmap.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_hic.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_hic
    return


@app.cell
def _(display_config, m, np, paths, plt, schema):
    from matplotlib.colors import LinearSegmentedColormap as _LSC

    _red = "#C94040"
    _green = "#3A9A5C"
    _cmap = _LSC.from_list(
        "rho_diverging", [_red, "white", _green], N=256
    )

    _spec_col = "spearman_rho"
    _hic = m[m["label_short"] == "HIC RT (norm)"].copy()

    _arm_cfgs = ["compositional_baseline"] + [
        c for c in schema.CONFIG_DISPLAY if c.startswith("arm_")
    ]
    _hic = _hic[_hic["config"].isin(_arm_cfgs)]
    _cfg_labels = [display_config(c) for c in _arm_cfgs if c in _hic["config"].values]
    _cfg_map = {c: display_config(c) for c in _arm_cfgs if c in _hic["config"].values}
    _model_order = sorted(_hic["model"].unique())

    _pivot = _hic.assign(
        config_display=_hic["config"].map(_cfg_map),
    ).pivot_table(index="model", columns="config_display", values=_spec_col)
    _pivot = _pivot[[c for c in _cfg_labels if c in _pivot.columns]]
    _pivot = _pivot.loc[[_m for _m in _model_order if _m in _pivot.index]]

    _vmin, _vmax = -1.0, 1.0

    _cell = 0.4
    _ncols, _nrows = len(_pivot.columns), len(_pivot.index)
    fig_hic_arm, _ax = plt.subplots(
        figsize=(_cell * _ncols + 3.5, _cell * _nrows + 2.5),
        layout="constrained",
    )
    _im = _ax.imshow(
        _pivot.values, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax
    )
    _ax.set_xticks(range(len(_pivot.columns)))
    _ax.set_xticklabels(_pivot.columns, rotation=45, ha="right")
    _ax.set_yticks(range(len(_pivot.index)))
    _ax.set_yticklabels(_pivot.index)
    for _i in range(_pivot.shape[0]):
        for _j in range(_pivot.shape[1]):
            _v = _pivot.values[_i, _j]
            if not np.isnan(_v):
                _lum = 0.299 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[0] * 255) \
                     + 0.587 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[1] * 255) \
                     + 0.114 * int(_cmap((_v - _vmin) / (_vmax - _vmin))[2] * 255)
                _color = "black" if _lum > 140 else "white"
                _ax.text(
                    _j, _i, f"{_v:.2f}", ha="center", va="center",
                    color=_color, fontsize=8,
                )
    fig_hic_arm.colorbar(_im, ax=_ax, label="Spearman ρ (LOO)", shrink=0.8)
    _ax.set_xlabel("Feature config")
    _ax.set_ylabel("Model")
    _ax.set_title(
        "HIC RT (norm) -- LOO Spearman ρ by arm feature config x model",
    )

    _o = paths.FIGURES / "s07_hic_spearman_heatmap_arm.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_hic_arm.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_hic_arm
    return


@app.cell
def _(imp, m, oof, schema):
    from prophet_ab.features.naming import parse_label_name as _parse

    def _is_reported(label_col: str) -> bool:
        _vc, _ = _parse(label_col)
        return _vc in schema.REPORTED_VALUE_COLS

    m_reported = m[m["label"].map(_is_reported)].copy()
    oof_reported = oof[oof["label"].map(_is_reported)].copy()
    imp_reported = imp[imp["label"].map(_is_reported)].copy()
    return imp_reported, m_reported, oof_reported


@app.cell
def _(
    DATAPOINTS_COLORS,
    display_config,
    equalize_axes,
    m_reported,
    np,
    oof_reported,
    paths,
    plt,
):
    from collections import namedtuple as _nt

    _MetricSpec = _nt(
        "_MetricSpec", ["key", "col", "display", "ascending", "vmin", "vmax", "fmt"]
    )

    _RS_METRIC_SPECS = {
        "spearman": _MetricSpec(
            "spearman", "spearman_rho", "Spearman ρ", False, -0.2, 1.0, ".2f"
        ),
        "pearson": _MetricSpec(
            "pearson", "pearson_r", "Pearson r", False, -0.2, 1.0, ".2f"
        ),
        "mse": _MetricSpec("mse", "mse", "MSE", True, None, None, ".2g"),
        "rmse": _MetricSpec("rmse", "rmse", "RMSE", True, None, None, ".3g"),
    }

    _RS_TABLE_COLS = [
        "label_short", "config", "model", "n_features", "n_samples",
        "spearman_rho", "pearson_r", "r2", "mse", "rmse", "mae",
    ]

    _RS_CFG_ORDER = [
        "compositional_baseline",
        "corresponding_experimental",
        "all_experimental",
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    ]

    _RS_ARM_CFG_ORDER = [
        "compositional_baseline",
        "arm_corresponding",
        "arm_all_experimental",
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    ]

    _RS_CV_LABEL = "LOO"
    _RS_SECTION = "s07"
    _RS_SUFFIX = "_reported_subset"

    def rs_save_best_table(m_r, spec, paths):
        _best = (
            m_r.dropna(subset=[spec.col])
               .sort_values(spec.col, ascending=spec.ascending)
               .groupby("label").head(1)
               .sort_values(spec.col, ascending=spec.ascending)
               [_RS_TABLE_COLS]
        )
        _o = paths.TABLES / f"{_RS_SECTION}_best_per_label_{spec.key}{_RS_SUFFIX}.csv"
        _o.parent.mkdir(parents=True, exist_ok=True)
        _best.to_csv(_o, index=False)
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return _best

    def rs_save_summary_table(m_r, spec, paths):
        _summary = (
            m_r.groupby(["config", "model"])
               .agg(
                  n_labels=("label", "nunique"),
                  spearman_mean=("spearman_rho", "mean"),
                  spearman_median=("spearman_rho", "median"),
                  pearson_mean=("pearson_r", "mean"),
                  pearson_median=("pearson_r", "median"),
                  r2_median=("r2", "median"),
                  mse_median=("mse", "median"),
                  rmse_median=("rmse", "median"),
                  mae_median=("mae", "median"),
               )
               .round(3)
               .sort_values(f"{spec.key}_median", ascending=spec.ascending)
        )
        _o = paths.TABLES / f"{_RS_SECTION}_config_summary_{spec.key}{_RS_SUFFIX}.csv"
        _o.parent.mkdir(parents=True, exist_ok=True)
        _summary.to_csv(_o)
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return _summary

    def rs_plot_heatmap(m_r, spec, paths, np, plt, display_config):
        _pivot = m_r.assign(
            cm=m_r["config"].map(display_config) + "\n" + m_r["model"]
        ).pivot_table(index="label_short", columns="cm", values=spec.col)
        _sort_asc = spec.ascending
        _pivot = _pivot.loc[
            _pivot.median(axis=1).sort_values(ascending=_sort_asc).index
        ]

        _n_labels = len(_pivot.index)
        _vmin = spec.vmin if spec.vmin is not None else np.nanmin(_pivot.values)
        _vmax = spec.vmax if spec.vmax is not None else np.nanmax(_pivot.values)
        _cmap = "viridis_r" if spec.ascending else "viridis"
        _mid = (_vmin + _vmax) / 2

        _fig_h = max(4, 0.4 * _n_labels + 2)
        fig, _ax = plt.subplots(
            figsize=(max(10, 1.0 * len(_pivot.columns) + 2), _fig_h),
            layout="constrained",
        )
        _im = _ax.imshow(
            _pivot.values, aspect="auto", cmap=_cmap, vmin=_vmin, vmax=_vmax
        )
        _ax.set_xticks(range(len(_pivot.columns)))
        _ax.set_xticklabels(_pivot.columns, rotation=45, ha="right")
        _ax.set_yticks(range(len(_pivot.index)))
        _ax.set_yticklabels(_pivot.index)
        for _i in range(_pivot.shape[0]):
            for _j in range(_pivot.shape[1]):
                _v = _pivot.values[_i, _j]
                if not np.isnan(_v):
                    _color = "white" if (
                        (_v < _mid and not spec.ascending) or
                        (_v > _mid and spec.ascending)
                    ) else "black"
                    _ax.text(
                        _j, _i, f"{_v:{spec.fmt}}", ha="center", va="center",
                        color=_color, fontsize=6,
                    )
        fig.colorbar(
            _im, ax=_ax, label=f"{spec.display} ({_RS_CV_LABEL})", shrink=0.7
        )
        _ax.set_title(
            f"Out-of-fold {spec.display} across labels x (config, model) "
            f"-- parent-disjoint {_RS_CV_LABEL} (reported subset)"
        )

        _o = paths.FIGURES / f"{_RS_SECTION}_{spec.key}_heatmap{_RS_SUFFIX}.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    def rs_plot_bar_chart(m_r, spec, paths, np, plt, display_config,
                          cfg_order=None, suffix=""):
        if cfg_order is None:
            cfg_order = _RS_CFG_ORDER
        _agg = "min" if spec.ascending else "max"
        _md = m_r.assign(config_display=m_r["config"].map(display_config))
        _bpc = _md.groupby(["label_short", "config_display"])[spec.col].agg(_agg).unstack()
        _cfg_display_order = [display_config(c) for c in cfg_order]
        _bpc = _bpc[[c for c in _cfg_display_order if c in _bpc.columns]]
        _bpc = _bpc.loc[
            _bpc.median(axis=1).sort_values(ascending=spec.ascending).index
        ]

        _n_labels = len(_bpc.index)
        _fig_w = max(6, 0.5 * _n_labels + 3)
        fig, _ax = plt.subplots(figsize=(_fig_w, 5), layout="constrained")
        _x = np.arange(len(_bpc.index))
        _w = 0.85 / len(_bpc.columns)
        for _i, _cfg in enumerate(_bpc.columns):
            _ax.bar(_x + _i * _w, _bpc[_cfg].values, width=_w, label=_cfg)
        _ax.set_xticks(_x + _w * (len(_bpc.columns) - 1) / 2)
        _ax.set_xticklabels(_bpc.index, rotation=60, ha="right")
        _ax.set_ylabel(f"{spec.display} (best model, {_RS_CV_LABEL})")
        _ax.axhline(0, color="k", lw=0.5)
        _ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", ncols=2)
        _ax.set_title(
            f"Per-label {spec.display} across feature configs\n"
            f"parent-disjoint {_RS_CV_LABEL} (reported subset, best model per config)"
        )

        _o = paths.FIGURES / f"{_RS_SECTION}_{spec.key}_comparison{suffix}{_RS_SUFFIX}.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    def rs_plot_oof_scatter(m_r, oof_r, spec, paths, np, plt, display_config,
                            equalize_axes=None, DATAPOINTS_COLORS=None):
        _best = (
            m_r.dropna(subset=[spec.col])
               .sort_values(spec.col, ascending=spec.ascending)
               .groupby("label").head(1)
               .sort_values(spec.col, ascending=spec.ascending)
               [["label", "label_short", "config", "model", spec.col,
                 "n_test_predictions"]]
               .reset_index(drop=True)
        )

        _ncols = 3
        _nrows = max(1, (len(_best) + _ncols - 1) // _ncols)
        fig, _axes = plt.subplots(
            _nrows, _ncols, figsize=(5 * _ncols / 1.4, 5 * _nrows / 1.4),
            layout="constrained",
        )
        _axes = np.atleast_1d(_axes).flatten()

        _edge_kw = {}
        if DATAPOINTS_COLORS is not None:
            _edge_kw = dict(edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3)

        for _ax, (_, _row) in zip(_axes, _best.iterrows()):
            _sub = oof_r[
                (oof_r["label"] == _row["label"])
                & (oof_r["config"] == _row["config"])
                & (oof_r["model"] == _row["model"])
            ]
            _yt = _sub["y_true"].to_numpy()
            _yp = _sub["y_pred"].to_numpy()
            _ax.scatter(_yt, _yp, s=14, alpha=0.7, **_edge_kw)
            if len(_yt) > 0:
                _lo = float(np.nanmin([_yt.min(), _yp.min()]))
                _hi = float(np.nanmax([_yt.max(), _yp.max()]))
                _pad = 0.05 * (_hi - _lo) if _hi > _lo else 1.0
                _ax.plot(
                    [_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad],
                    ls="--", lw=0.8, color="k",
                )
                _ax.set_xlim(_lo - _pad, _hi + _pad)
                _ax.set_ylim(_lo - _pad, _hi + _pad)
            if equalize_axes is not None:
                equalize_axes(_ax)
            _metric_val = _row[spec.col]
            _ax.set_title(
                f"{_row['label_short']}\n{display_config(_row['config'])} . {_row['model']}  "
                f"{spec.display}={_metric_val:{spec.fmt}}  "
                f"n={int(_row['n_test_predictions'])}",
                fontsize=7,
            )
            _ax.tick_params(labelsize=6)
            _ax.set_xlabel("experimental", fontsize=7)
            _ax.set_ylabel(f"predicted ({_RS_CV_LABEL})", fontsize=7)

        for _ax in _axes[len(_best):]:
            _ax.axis("off")

        fig.suptitle(
            f"{_RS_CV_LABEL} predicted vs experimental -- best (config x model) per label "
            f"by {spec.display} (reported subset, parent-disjoint LOO)",
            fontsize=9,
        )

        _o = paths.FIGURES / f"{_RS_SECTION}_oof_scatter_{spec.key}{_RS_SUFFIX}.png"
        _o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_o, dpi=300, bbox_inches="tight")
        print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
        return fig

    _rs_spec = _RS_METRIC_SPECS["spearman"]
    rs_save_best_table(m_reported, _rs_spec, paths)
    rs_save_summary_table(m_reported, _rs_spec, paths)
    rs_plot_heatmap(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config,
                      cfg_order=_RS_ARM_CFG_ORDER, suffix="_arm")
    rs_plot_oof_scatter(m_reported, oof_reported, _rs_spec, paths, np, plt,
                        display_config, equalize_axes=equalize_axes,
                        DATAPOINTS_COLORS=DATAPOINTS_COLORS)

    _rs_spec = _RS_METRIC_SPECS["mse"]
    rs_save_best_table(m_reported, _rs_spec, paths)
    rs_save_summary_table(m_reported, _rs_spec, paths)
    rs_plot_heatmap(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config,
                      cfg_order=_RS_ARM_CFG_ORDER, suffix="_arm")
    rs_plot_oof_scatter(m_reported, oof_reported, _rs_spec, paths, np, plt,
                        display_config, equalize_axes=equalize_axes,
                        DATAPOINTS_COLORS=DATAPOINTS_COLORS)

    _rs_spec = _RS_METRIC_SPECS["rmse"]
    rs_save_best_table(m_reported, _rs_spec, paths)
    rs_save_summary_table(m_reported, _rs_spec, paths)
    rs_plot_heatmap(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config,
                      cfg_order=_RS_ARM_CFG_ORDER, suffix="_arm")
    rs_plot_oof_scatter(m_reported, oof_reported, _rs_spec, paths, np, plt,
                        display_config, equalize_axes=equalize_axes,
                        DATAPOINTS_COLORS=DATAPOINTS_COLORS)

    _rs_spec = _RS_METRIC_SPECS["pearson"]
    rs_save_best_table(m_reported, _rs_spec, paths)
    rs_save_summary_table(m_reported, _rs_spec, paths)
    rs_plot_heatmap(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config)
    rs_plot_bar_chart(m_reported, _rs_spec, paths, np, plt, display_config,
                      cfg_order=_RS_ARM_CFG_ORDER, suffix="_arm")
    rs_plot_oof_scatter(m_reported, oof_reported, _rs_spec, paths, np, plt,
                        display_config, equalize_axes=equalize_axes,
                        DATAPOINTS_COLORS=DATAPOINTS_COLORS)
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp_reported,
    m_reported,
    np,
    paths,
    plt,
):
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m_reported[m_reported["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best_all = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ntotal = len(_best_all)
    _grid_cols = 3
    _grid_rows = max(1, (_ntotal + _grid_cols - 1) // _grid_cols)
    _pw, _ph = 3.0, 2.5
    fig_imp_all_groups_rs, _axes = plt.subplots(
        _grid_rows, _grid_cols, figsize=(_pw * _grid_cols, _ph * _grid_rows),
        layout="constrained",
    )
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best_all.iterrows()):
        _sl = imp_reported[
            (imp_reported["label"] == _row["label"])
            & (imp_reported["config"] == _row["config"])
            & (imp_reported["model"] == _row["model"])
        ]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=5)
        _ax.set_title(
            f"{_row['label_short']}\n{display_config(_row['config'])} . {_row['model']}"
            f" . rho={_row['spearman_rho']:.2f}",
            fontsize=6,
        )
        _ax.tick_params(axis="x", labelsize=5)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ntotal:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_all_groups_rs.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside lower center", ncols=4,
    )
    fig_imp_all_groups_rs.suptitle(
        "Top permutation-importance features -- reported subset\n"
        "(best arm config x model by LOO Spearman ρ, ordered by performance)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_all_arm_groups_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_all_groups_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_all_groups_rs
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    display_config,
    display_feature,
    imp_reported,
    m_reported,
    np,
    paths,
    plt,
):
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m_reported[m_reported["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best_all = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _ntotal = len(_best_all)
    _grid_cols = 3
    _grid_rows = max(1, (_ntotal + _grid_cols - 1) // _grid_cols)
    _pw, _ph = 3.0, 2.5
    fig_imp_all_nonarm_groups_rs, _axes = plt.subplots(
        _grid_rows, _grid_cols, figsize=(_pw * _grid_cols, _ph * _grid_rows),
        layout="constrained",
    )
    _axes = np.atleast_1d(_axes).flatten()

    for _ax, (_, _row) in zip(_axes, _best_all.iterrows()):
        _sl = imp_reported[
            (imp_reported["label"] == _row["label"])
            & (imp_reported["config"] == _row["config"])
            & (imp_reported["model"] == _row["model"])
        ]
        _sl = _sl.sort_values("abs_perm", ascending=False).head(8).iloc[::-1]
        _colors = [FEATURE_GROUP_COLORS[classify_feature(f, _row["label"])]
                   for f in _sl["feature"]]
        _labels = [display_feature(r["feature"]) for _, r in _sl.iterrows()]
        _ax.barh(range(len(_sl)), _sl["perm_importance_mean"].values,
                 xerr=_sl["perm_importance_std"].values, color=_colors)
        _ax.set_yticks(range(len(_sl)))
        _ax.set_yticklabels(_labels, fontsize=5)
        _ax.set_title(
            f"{_row['label_short']}\n{display_config(_row['config'])} . {_row['model']}"
            f" . rho={_row['spearman_rho']:.2f}",
            fontsize=6,
        )
        _ax.tick_params(axis="x", labelsize=5)
        _ax.axvline(0, color="k", lw=0.4)

    for _ax in _axes[_ntotal:]:
        _ax.axis("off")

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    fig_imp_all_nonarm_groups_rs.legend(
        _handles, FEATURE_GROUP_ORDER,
        loc="outside lower center", ncols=4,
    )
    fig_imp_all_nonarm_groups_rs.suptitle(
        "Top permutation-importance features -- reported subset\n"
        "(best operator config x model by LOO Spearman ρ, ordered by performance)",
        fontsize=9,
    )

    _o = paths.FIGURES / "s07_top_features_all_groups_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_imp_all_nonarm_groups_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_imp_all_nonarm_groups_rs
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp_reported,
    m_reported,
    np,
    paths,
    plt,
):
    _ARM_IS_CONFIGS = (
        "arm_in_silico_only",
        "arm_in_silico_plus_corresponding",
        "arm_in_silico_plus_all_experimental",
    )
    _ms = m_reported[m_reported["config"].isin(_ARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_all_rs, _ax = plt.subplots(figsize=(max(6, 0.55 * _n), 3.75),
                                           layout="constrained")

    _x_positions = np.arange(_n)
    _tick_labels_all = []
    _rev_order = list(reversed(FEATURE_GROUP_ORDER))

    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp_reported[
            (imp_reported["label"] == _row["label"])
            & (imp_reported["config"] == _row["config"])
            & (imp_reported["model"] == _row["model"])
        ]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.7,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            _bottom += _h

        _tick_labels_all.append(_row['label_short'])

    _ax.set_xticks(_x_positions)
    _ax.set_xticklabels(_tick_labels_all, rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)
    _ax.axhline(0, color="k", lw=0.4)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_all_rs.suptitle(
        "Performance & importance breakdown by feature group\n"
        "reported subset (best arm config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_all_arm_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_all_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_all_rs
    return


@app.cell
def _(
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    classify_feature,
    imp_reported,
    m_reported,
    np,
    paths,
    plt,
):
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _ms = m_reported[m_reported["config"].isin(_NONARM_IS_CONFIGS)].dropna(subset=["spearman_rho"])
    _best = (
        _ms.sort_values("spearman_rho", ascending=False)
           .groupby("label").head(1)
           .sort_values("spearman_rho", ascending=False)
    )

    _n = len(_best)
    fig_stacked_all_nonarm_rs, _ax = plt.subplots(figsize=(max(6, 0.55 * _n), 3.75),
                                                  layout="constrained")

    _x_positions = np.arange(_n)
    _tick_labels_all = []
    _rev_order = list(reversed(FEATURE_GROUP_ORDER))

    for _i, (_, _row) in enumerate(_best.iterrows()):
        _sl = imp_reported[
            (imp_reported["label"] == _row["label"])
            & (imp_reported["config"] == _row["config"])
            & (imp_reported["model"] == _row["model"])
        ]
        _sl = _sl.assign(
            group=_sl["feature"].map(lambda f, lbl=_row["label"]: classify_feature(f, lbl)),
        )
        _group_imp = _sl.groupby("group")["abs_perm"].sum()
        _total_imp = _group_imp.sum()

        _rho = _row["spearman_rho"]
        _bottom = 0.0
        for _g in _rev_order:
            if _g not in _group_imp.index:
                continue
            _frac = _group_imp[_g] / _total_imp if _total_imp > 0 else 0
            _h = _frac * _rho
            _ax.bar(_i, _h, bottom=_bottom, width=0.7,
                    color=FEATURE_GROUP_COLORS[_g], edgecolor="white", linewidth=0.3)
            _bottom += _h

        _tick_labels_all.append(_row['label_short'])

    _ax.set_xticks(_x_positions)
    _ax.set_xticklabels(_tick_labels_all, rotation=45, ha="right")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.set_ylim(0, None)
    _ax.axhline(0, color="k", lw=0.4)

    _handles = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                for g in FEATURE_GROUP_ORDER]
    _ax.legend(
        _handles, FEATURE_GROUP_ORDER,
        bbox_to_anchor=(1.02, 0.5), loc="center left",
    )
    fig_stacked_all_nonarm_rs.suptitle(
        "Performance & importance breakdown by feature group\n"
        "reported subset (best operator config x model by LOO Spearman ρ)",
    )

    _o = paths.FIGURES / "s07_stacked_importance_all_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_stacked_all_nonarm_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_stacked_all_nonarm_rs
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    equalize_axes,
    get_colors_from_cmap,
    m_reported,
    np,
    paths,
    pd,
    plt,
):
    _kf = pd.read_parquet(paths.S05 / "cv_metrics.parquet")
    _join = (
        m_reported[["label", "label_short", "config", "model", "spearman_rho"]]
         .rename(columns={"spearman_rho": "rho_loo"})
         .merge(
             _kf[["label", "config", "model", "spearman_rho"]]
              .rename(columns={"spearman_rho": "rho_kfold"}),
             on=["label", "config", "model"],
         )
         .dropna(subset=["rho_loo", "rho_kfold"])
    )

    fig_c_rs, _ax = plt.subplots(figsize=(6, 6), layout="constrained")
    _models = sorted(_join["model"].unique())
    _colors = get_colors_from_cmap(number_of_colors=len(_models))
    for _i, _mn in enumerate(_models):
        _sub = _join[_join["model"] == _mn]
        _ax.scatter(
            _sub["rho_kfold"], _sub["rho_loo"],
            s=22, alpha=0.7,
            edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
            color=_colors[_i], label=f"{_mn} (n={len(_sub)})",
        )
    _lo = float(np.nanmin([_join["rho_kfold"].min(), _join["rho_loo"].min()]))
    _hi = float(np.nanmax([_join["rho_kfold"].max(), _join["rho_loo"].max()]))
    _pad = 0.05 * (_hi - _lo) if _hi > _lo else 0.1
    _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], ls="--", lw=0.8, color="k")
    _ax.set_xlim(_lo - _pad, _hi + _pad)
    _ax.set_ylim(_lo - _pad, _hi + _pad)
    equalize_axes(_ax)
    _ax.set_xlabel("Spearman ρ -- parent-aware 5-fold")
    _ax.set_ylabel("Spearman ρ -- parent-disjoint LOO")
    _r = float(_join[["rho_kfold", "rho_loo"]].corr().iloc[0, 1])
    _ax.set_title(
        f"LOO vs k-fold Spearman ρ (reported subset,\n"
        f"n={len(_join)} combos, Pearson r={_r:.2f})",
    )
    _ax.legend(loc="lower right")
    _ax.grid(alpha=0.3)

    _o = paths.FIGURES / "s07_loo_vs_kfold_scatter_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_c_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_c_rs
    return


@app.cell
def _(DATAPOINTS_COLORS, display_config, m_reported, paths, plt, schema):
    import seaborn as _sns

    _cfg_order = list(schema.CONFIG_DISPLAY.keys())
    _plot = m_reported.dropna(subset=["spearman_rho"]).copy()
    _plot["config_label"] = _plot["config"].map(display_config)
    _plot["model_upper"] = _plot["model"].str.upper()
    _model_order = sorted(_plot["model_upper"].unique())

    _palette = [
        DATAPOINTS_COLORS["blue"],
        DATAPOINTS_COLORS["purple"],
        DATAPOINTS_COLORS["pink"],
        DATAPOINTS_COLORS["red"],
        DATAPOINTS_COLORS["teal"],
        DATAPOINTS_COLORS["gray"],
        DATAPOINTS_COLORS["amber"],
        DATAPOINTS_COLORS["navy"],
        DATAPOINTS_COLORS["green"],
        DATAPOINTS_COLORS["coral"],
        DATAPOINTS_COLORS["slate"],
    ]
    _hue_order = [display_config(c) for c in _cfg_order]

    fig_box_rs, _ax = plt.subplots(figsize=(10, 5), layout="constrained")
    _sns.boxenplot(
        data=_plot,
        x="model_upper",
        y="spearman_rho",
        hue="config_label",
        order=_model_order,
        hue_order=_hue_order,
        palette=_palette,
        linewidth=0.6,
        ax=_ax,
    )
    _ax.set_xlabel("Model")
    _ax.set_ylabel("Spearman ρ (LOO)")
    _ax.axhline(0, color="k", lw=0.5, ls="--")
    _ax.legend(
        title="Feature config",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    _ax.set_title(
        "Spearman ρ by model and feature config\n"
        "parent-disjoint LOO (reported subset)",
    )

    _o = paths.FIGURES / "s07_boxenplot_model_config_reported_subset.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_box_rs.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_box_rs
    return


@app.cell
def _(mo):
    mo.md("""
    ## Marginal value of supervised models over the compositional baseline

    Dumbbell view: for each label, the compositional baseline (Parent Mean
    operator, best model) versus the best supervised model across all other
    feature configs. The connector length is the per-label Spearman ρ gain
    from learning beyond the parent average. Standard-style replacement for
    figure_5 panel D.
    """)
    return


@app.function
def plot_supervised_vs_baseline(m_in, paths, np, pd, plt, DATAPOINTS_COLORS,
                                suffix="", subtitle=""):
    _md = m_in.dropna(subset=["spearman_rho"])

    _base = (
        _md[_md["config"] == "compositional_baseline"]
        .sort_values("spearman_rho", ascending=False)
        .groupby("label").head(1)
        .set_index("label")
    )
    _sup = (
        _md[_md["config"] != "compositional_baseline"]
        .sort_values("spearman_rho", ascending=False)
        .groupby("label").head(1)
        .set_index("label")
    )
    _labels = [_l for _l in _sup.index if _l in _base.index]
    _df = (
        pd.DataFrame({
            "label_short": _base.loc[_labels, "label_short"],
            "baseline": _base.loc[_labels, "spearman_rho"],
            "supervised": _sup.loc[_labels, "spearman_rho"],
        })
        .assign(gain=lambda d: d["supervised"] - d["baseline"])
        .sort_values("supervised", ascending=True)
        .reset_index(drop=True)
    )

    _n = len(_df)
    _fig_h = max(3.0, 0.42 * _n + 1.2)
    fig, _ax = plt.subplots(figsize=(7.2, _fig_h), layout="constrained")
    _y = np.arange(_n)

    _c_base = DATAPOINTS_COLORS["gray"]
    _c_sup = DATAPOINTS_COLORS["blue"]
    _c_up = DATAPOINTS_COLORS["green"]
    _c_down = DATAPOINTS_COLORS["red"]

    for _i, _row in _df.iterrows():
        _col = _c_up if _row["gain"] >= 0 else _c_down
        _ax.plot([_row["baseline"], _row["supervised"]], [_i, _i],
                 color=_col, lw=1.6, alpha=0.8, zorder=1)
    _ax.scatter(_df["baseline"], _y, s=42, color=_c_base, zorder=2,
                edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                label="Compositional baseline (Parent Mean)")
    _ax.scatter(_df["supervised"], _y, s=42, color=_c_sup, zorder=3,
                edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                label="Best supervised model")

    _xmax = float(np.nanmax([_df["baseline"].max(), _df["supervised"].max()]))
    _ax.axvline(0, color="k", lw=0.5)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_df["label_short"])
    _ax.set_ylim(-0.6, _n - 0.4)
    _ax.set_xlim(min(-0.05, float(_df[["baseline", "supervised"]].min().min()) - 0.05),
                 _xmax + 0.05)
    _ax.set_xlabel("Spearman ρ (out-of-fold, parent-disjoint LOO)")
    _ax.legend(loc="lower right", frameon=False)
    _ax.set_title(
        "Supervised models add marginal value over the compositional baseline"
        + (f"\n{subtitle}" if subtitle else "")
    )

    _o = paths.FIGURES / f"s07_supervised_vs_baseline{suffix}.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    return fig


@app.cell
def _(DATAPOINTS_COLORS, m, mo, np, paths, pd, plt):
    _fig = plot_supervised_vs_baseline(
        m, paths, np, pd, plt, DATAPOINTS_COLORS,
        subtitle="all labels (best model per config, parent-disjoint LOO)",
    )
    mo.as_html(_fig)
    return


@app.cell
def _(DATAPOINTS_COLORS, m_reported, mo, np, paths, pd, plt):
    _fig = plot_supervised_vs_baseline(
        m_reported, paths, np, pd, plt, DATAPOINTS_COLORS,
        suffix="_reported_subset",
        subtitle="reported subset (best model per config, parent-disjoint LOO)",
    )
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
