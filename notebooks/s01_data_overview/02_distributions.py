import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from prophet_ab import paths, schema
    from prophet_ab.features.naming import display_value_col
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        equalize_axes,
        grid_figsize,
        wrap_title,
        FULL_WIDTH,
        SINGLE_COL_WIDTH,
        FONT_SIZE_LABEL,
        FONT_SIZE_LEGEND,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
    )

    set_manuscript_style()
    return (
        DATAPOINTS_COLORS,
        FONT_SIZE_LABEL,
        FONT_SIZE_LEGEND,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
        FULL_WIDTH,
        SINGLE_COL_WIDTH,
        display_value_col,
        equalize_axes,
        grid_figsize,
        mo,
        np,
        paths,
        pd,
        plt,
        schema,
        stats,
        wrap_title,
    )


@app.cell
def _(mo):
    mo.md("""
    # s01 — Property distributions and correlations

    Box-and-strip distributions of all assay and production metrics
    (bispecific vs monospecific), plus Spearman and Pearson correlation
    heatmaps (assay-vs-assay and production-vs-assay) for both bispecific
    and monospecific antibodies.

    Reads `n3n4_per_antibody.parquet` (stage 03 aggregated).
    """)
    return


@app.cell
def _(paths, pd, schema):
    per_ab = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")

    _assay_mask = (
        ~per_ab["value_col"].isin(schema.DEPRECATED_VALUE_COLS)
        & ~per_ab["value_col"].isin(schema.PRODUCTION_VALUE_COLS)
    )
    assay_data = per_ab[_assay_mask].copy()
    prod_data = per_ab[per_ab["value_col"].isin(schema.PRODUCTION_VALUE_COLS)].copy()
    return assay_data, prod_data


@app.cell
def _(schema):
    _vc_to_assay = {}
    for _assay, _cond, _vc in schema.ASSAY_PANEL:
        _vc_to_assay.setdefault(_vc, _assay)

    ASSAY_GROUP_ORDER = [
        "AC-SINS", "BVP", "HPLC-HAC", "HPLC-HIC", "HPLC-SEC",
        "HPLC-SMAC", "IntactMS", "PR", "PTS-IF",
    ]

    PROD_ORDER = list(schema.PRODUCTION_VALUE_COLS)

    vc_to_assay = _vc_to_assay
    return ASSAY_GROUP_ORDER, PROD_ORDER, vc_to_assay


@app.cell
def _(assay_data, vc_to_assay):
    assay_data_tagged = assay_data.assign(
        assay_group=assay_data["value_col"].map(vc_to_assay),
    )
    return (assay_data_tagged,)


@app.cell
def _(DATAPOINTS_COLORS, FONT_SIZE_TICK, FONT_SIZE_TITLE, np):
    _KIND_LABEL = {"N3": "Bispecific", "N4": "Monospecific"}

    def plot_box_strip(ax, df, metric_label):
        kinds = ["N3", "N4"]
        colors = {"N3": DATAPOINTS_COLORS["blue"], "N4": DATAPOINTS_COLORS["purple"]}
        positions = {k: i for i, k in enumerate(kinds)}

        for kind in kinds:
            _vals = df[df["kind"] == kind]["median"].dropna().values
            if len(_vals) == 0:
                continue
            _pos = positions[kind]
            ax.boxplot(
                _vals,
                positions=[_pos],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=colors[kind], alpha=0.3),
                medianprops=dict(color=colors[kind], linewidth=2),
                whiskerprops=dict(color=colors[kind]),
                capprops=dict(color=colors[kind]),
                flierprops=dict(marker="", markersize=0),
                showfliers=False,
            )
            _jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(_vals))
            ax.scatter(
                _pos + _jitter, _vals,
                alpha=0.4, s=12, color=colors[kind],
                edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                zorder=3,
            )

        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([_KIND_LABEL[k] for k in kinds])
        ax.set_title(metric_label, fontsize=FONT_SIZE_TITLE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    return (plot_box_strip,)


@app.cell
def _(np):
    from scipy.stats import gaussian_kde as _gaussian_kde
    from scipy.signal import find_peaks as _find_peaks

    def top_modes(vals, bw=0.15, n=2, grid_points=1024):
        # Top-n KDE peaks (by density), using the same bandwidth as the violin
        # so the reported modes line up with the drawn shape.
        _vals = np.asarray(vals, dtype=float)
        _kde = _gaussian_kde(_vals, bw_method=bw)
        _grid = np.linspace(_vals.min(), _vals.max(), grid_points)
        _dens = _kde(_grid)
        _peaks, _ = _find_peaks(_dens)
        if len(_peaks) == 0:
            _peaks = np.array([int(_dens.argmax())])
        _order = np.argsort(_dens[_peaks])[::-1]
        _top = _peaks[_order][:n]
        return sorted(float(_grid[i]) for i in _top)

    # Fine-resolution single violin at a given x position; small bw_method
    # keeps the KDE from over-smoothing so multiple peaks stay visible.
    # Horizontal bars mark the top-2 modes (KDE peaks) of the distribution.
    def draw_violin(ax, vals, pos, color, width=0.7, bw=0.15):
        _parts = ax.violinplot(
            vals,
            positions=[pos],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=bw,
            points=2000,
        )
        for _body in _parts["bodies"]:
            _body.set_facecolor(color)
            _body.set_edgecolor(color)
            _body.set_alpha(0.3)
            _body.set_linewidth(0.8)

        _modes = top_modes(vals, bw=bw, n=2)
        for _m in _modes:
            ax.hlines(
                _m, pos - width * 0.42, pos + width * 0.42,
                color=color, linewidth=2, zorder=4,
            )
        return _modes

    return (draw_violin,)


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    SINGLE_COL_WIDTH,
    assay_data_tagged,
    display_value_col,
    draw_violin,
    paths,
    plt,
    schema,
):
    from matplotlib.patches import Patch as _Patch

    _pts = assay_data_tagged[assay_data_tagged["assay_group"] == "PTS-IF"]
    _combos = (
        _pts.groupby(["value_col", "condition"])
        .size()
        .reset_index(name="_n")
        .sort_values(["value_col", "condition"])
    )
    _combos = _combos[~_combos["value_col"].isin(schema.DEPRECATED_VALUE_COLS)]
    _combos = list(_combos[["value_col", "condition"]].itertuples(index=False, name=None))

    # Order by thermal event: Tonset, then Tm1, then Tm2.
    _vc_order = {
        "thermostability_tonset": 0,
        "thermostability_tm1": 1,
        "thermostability_tm2": 2,
    }
    _combos = sorted(_combos, key=lambda t: _vc_order.get(t[0], 99))

    _kinds = [("N3", "Bispecific", DATAPOINTS_COLORS["blue"]),
              ("N4", "Monospecific", DATAPOINTS_COLORS["purple"])]
    _group_gap = 2.6
    _offset = 0.45

    _fig, _ax = plt.subplots(
        figsize=(SINGLE_COL_WIDTH * 1.7, SINGLE_COL_WIDTH),
        layout="constrained",
    )

    _centers = []
    for _gi, (_vc, _cond) in enumerate(_combos):
        _center = _gi * _group_gap
        _centers.append(_center)
        for _ki, (_kind, _kind_lbl, _color) in enumerate(_kinds):
            _vals = _pts[
                (_pts["value_col"] == _vc)
                & (_pts["condition"] == _cond)
                & (_pts["kind"] == _kind)
            ]["median"].dropna().values
            if len(_vals) == 0:
                continue
            _pos = _center + (_offset if _ki else -_offset)
            _modes = draw_violin(_ax, _vals, _pos, _color)
            print(
                f"  {display_value_col(_vc, _cond):>8} {_kind_lbl:<13} "
                f"top-2 modes (°C): {', '.join(f'{_m:.1f}' for _m in _modes)}"
            )

    _ax.set_xticks(_centers)
    _ax.set_xticklabels(
        [display_value_col(_vc, _cond) for _vc, _cond in _combos],
        fontsize=FONT_SIZE_TICK,
    )
    _ax.set_ylabel("Temperature (°C)", fontsize=FONT_SIZE_LABEL)
    _ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    _ax.set_title("PTS-IF thermostability", fontsize=FONT_SIZE_TITLE, fontweight="bold")

    _ax.legend(
        handles=[_Patch(facecolor=_c, edgecolor=_c, alpha=0.3, label=_l)
                 for _, _l, _c in _kinds],
        loc="upper right", frameon=False, fontsize=FONT_SIZE_LEGEND,
    )

    _out = paths.FIGURES / "s01_violin_pts_if.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    SINGLE_COL_WIDTH,
    assay_data_tagged,
    draw_violin,
    paths,
    plt,
):
    # Pool Tm1 and Tm2 within each group so each violin is the full set of
    # thermal-transition temperatures for that antibody kind.
    _pts = assay_data_tagged[
        (assay_data_tagged["assay_group"] == "PTS-IF")
        & assay_data_tagged["value_col"].isin(
            ["thermostability_tm1", "thermostability_tm2"]
        )
    ]

    _kinds = [("N3", "Bispecific", DATAPOINTS_COLORS["blue"]),
              ("N4", "Monospecific", DATAPOINTS_COLORS["purple"])]

    _fig, _ax = plt.subplots(
        figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH),
        layout="constrained",
    )

    for _pos, (_kind, _kind_lbl, _color) in enumerate(_kinds):
        _vals = _pts[_pts["kind"] == _kind]["median"].dropna().values
        if len(_vals) == 0:
            continue
        _modes = draw_violin(_ax, _vals, _pos, _color, width=0.6)
        print(
            f"  pooled Tm1+Tm2 {_kind_lbl:<13} "
            f"top-2 modes (°C): {', '.join(f'{_m:.1f}' for _m in _modes)}"
        )

    _ax.set_xticks(range(len(_kinds)))
    _ax.set_xticklabels([_l for _, _l, _ in _kinds], fontsize=FONT_SIZE_TICK)
    _ax.set_ylabel("Tm1 + Tm2 (°C)", fontsize=FONT_SIZE_LABEL)
    _ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    _ax.set_title(
        "PTS-IF pooled thermal transitions",
        fontsize=FONT_SIZE_TITLE, fontweight="bold",
    )

    _out = paths.FIGURES / "s01_violin_pts_if_tm_pooled.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    ASSAY_GROUP_ORDER,
    FONT_SIZE_TITLE,
    assay_data_tagged,
    display_value_col,
    grid_figsize,
    np,
    paths,
    plot_box_strip,
    plt,
    schema,
):
    _saved_assay_figs = []

    for _assay in ASSAY_GROUP_ORDER:
        _sub = assay_data_tagged[assay_data_tagged["assay_group"] == _assay]
        _combos = (
            _sub.groupby(["value_col", "condition"])
            .size()
            .reset_index(name="_n")
            .sort_values(["value_col", "condition"])
        )
        _combos = _combos[~_combos["value_col"].isin(schema.DEPRECATED_VALUE_COLS)]
        _n_panels = len(_combos)
        if _n_panels == 0:
            continue

        _ncols = min(_n_panels, 4)
        _nrows = (_n_panels + _ncols - 1) // _ncols
        _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
        _fig, _axes = plt.subplots(
            _nrows, _ncols, figsize=(_fw, _fh),
            layout="constrained",
        )
        if _n_panels == 1:
            _axes = [_axes]
        else:
            _axes = list(np.array(_axes).flatten())

        for _i, (_, _row) in enumerate(_combos.iterrows()):
            _vc, _cond = _row["value_col"], _row["condition"]
            _panel_data = _sub[
                (_sub["value_col"] == _vc) & (_sub["condition"] == _cond)
            ]
            _label = display_value_col(_vc, _cond)
            plot_box_strip(_axes[_i], _panel_data, _label)

        for _j in range(_n_panels, len(_axes)):
            _axes[_j].set_visible(False)

        _fig.suptitle(_assay, fontsize=FONT_SIZE_TITLE, fontweight="bold")

        _slug = _assay.lower().replace("-", "_")
        _out = paths.FIGURES / f"s01_dist_{_slug}.png"
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=300, bbox_inches="tight")
        _saved_assay_figs.append(str(_out.relative_to(paths.REPO_ROOT)))
        plt.close(_fig)

    print(f"Saved {len(_saved_assay_figs)} assay distribution figures")
    for _p in _saved_assay_figs:
        print(f"  {_p}")
    return


@app.cell
def _(
    FONT_SIZE_TITLE,
    PROD_ORDER,
    display_value_col,
    grid_figsize,
    np,
    paths,
    plot_box_strip,
    plt,
    prod_data,
):
    _n_prod = len(PROD_ORDER)
    _ncols = min(_n_prod, 3)
    _nrows = (_n_prod + _ncols - 1) // _ncols
    _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_fw, _fh),
        layout="constrained",
    )
    _axes = list(np.array(_axes).flatten())

    for _i, _vc in enumerate(PROD_ORDER):
        _panel_data = prod_data[prod_data["value_col"] == _vc]
        plot_box_strip(_axes[_i], _panel_data, display_value_col(_vc))

    for _j in range(_n_prod, len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle("Production QC", fontsize=FONT_SIZE_TITLE, fontweight="bold")

    _out = paths.FIGURES / "s01_dist_production.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(assay_data, display_value_col, np, pd, schema, stats):
    def _pivot_wide(df):
        _df = df[~df["value_col"].isin(schema.DEPRECATED_VALUE_COLS)].copy()
        _df["metric_label"] = _df.apply(
            lambda r: display_value_col(r["value_col"], r["condition"]), axis=1
        )
        return _df.pivot_table(
            index="antibody_name", columns="metric_label",
            values="median", aggfunc="first",
        )

    def _corr_matrix(wide, method):
        cols = wide.columns
        n = len(cols)
        _mat = pd.DataFrame(np.nan, index=cols, columns=cols)
        for _i in range(n):
            for _j in range(_i, n):
                _valid = wide[[cols[_i], cols[_j]]].dropna()
                if len(_valid) < 5:
                    continue
                if method == "spearman":
                    _r, _ = stats.spearmanr(_valid.iloc[:, 0], _valid.iloc[:, 1])
                else:
                    _r, _ = stats.pearsonr(_valid.iloc[:, 0], _valid.iloc[:, 1])
                _mat.iloc[_i, _j] = _r
                _mat.iloc[_j, _i] = _r
        return _mat

    assay_wide_n3 = _pivot_wide(assay_data[assay_data["kind"] == "N3"])
    assay_wide_n4 = _pivot_wide(assay_data[assay_data["kind"] == "N4"])

    assay_corr = {}
    for _kind, _wide in [("N3", assay_wide_n3), ("N4", assay_wide_n4)]:
        for _method in ["spearman", "pearson"]:
            assay_corr[(_kind, _method)] = _corr_matrix(_wide, _method)
    return assay_corr, assay_wide_n3, assay_wide_n4


@app.cell
def _(assay_wide_n3, assay_wide_n4, np, pd, prod_data, schema, stats):
    def _pivot_prod(df):
        _df = df.copy()
        return _df.pivot_table(
            index="antibody_name", columns="value_col",
            values="median", aggfunc="first",
        )

    prod_wide_n3 = _pivot_prod(prod_data[prod_data["kind"] == "N3"])
    prod_wide_n4 = _pivot_prod(prod_data[prod_data["kind"] == "N4"])

    def _cross_corr(prod_wide, assay_wide, method):
        _prod_cols = [c for c in prod_wide.columns if c in schema.PRODUCTION_VALUE_COLS]
        _assay_cols = list(assay_wide.columns)
        _mat = pd.DataFrame(np.nan, index=_prod_cols, columns=_assay_cols)
        for _pc in _prod_cols:
            for _ac in _assay_cols:
                _merged = pd.concat(
                    [prod_wide[_pc], assay_wide[_ac]], axis=1, join="inner"
                ).dropna()
                if len(_merged) < 5:
                    continue
                if method == "spearman":
                    _r, _ = stats.spearmanr(_merged.iloc[:, 0], _merged.iloc[:, 1])
                else:
                    _r, _ = stats.pearsonr(_merged.iloc[:, 0], _merged.iloc[:, 1])
                _mat.loc[_pc, _ac] = _r
        return _mat

    prod_vs_assay_corr = {}
    for _kind, _pw, _aw in [
        ("N3", prod_wide_n3, assay_wide_n3),
        ("N4", prod_wide_n4, assay_wide_n4),
    ]:
        for _method in ["spearman", "pearson"]:
            prod_vs_assay_corr[(_kind, _method)] = _cross_corr(_pw, _aw, _method)
    return (prod_vs_assay_corr,)


@app.cell
def _(
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FULL_WIDTH,
    assay_corr,
    display_value_col,
    np,
    paths,
    plt,
    prod_vs_assay_corr,
):
    _fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.95))
    _gs = _fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.45, hspace=0.6)

    _tick_fs = FONT_SIZE_TICK - 4
    _annot_fs = FONT_SIZE_LEGEND - 4

    _titles = [
        ("N3 Spearman", "N3 Pearson",
         "N4 Spearman", "N4 Pearson"),
        ("N3 prod vs assay\nSpearman", "N3 prod vs assay\nPearson",
         "N4 prod vs assay\nSpearman", "N4 prod vs assay\nPearson"),
    ]
    _keys_top = [("N3", "spearman"), ("N3", "pearson"), ("N4", "spearman"), ("N4", "pearson")]
    _keys_bot = _keys_top

    _axes_top = [_fig.add_subplot(_gs[0, _c]) for _c in range(4)]
    _axes_bot = [_fig.add_subplot(_gs[1, _c]) for _c in range(4)]

    for _col, _key in enumerate(_keys_top):
        _mat = assay_corr[_key]
        _ax = _axes_top[_col]
        _im = _ax.imshow(_mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        _ax.set_xticks(range(len(_mat.columns)))
        _ax.set_xticklabels(_mat.columns, rotation=90, fontsize=_tick_fs)
        _ax.set_yticks(range(len(_mat.columns)))
        _ax.set_yticklabels(_mat.columns, fontsize=_tick_fs)
        _ax.set_title(_titles[0][_col], fontsize=_tick_fs + 1)

    for _col, _key in enumerate(_keys_bot):
        _mat = prod_vs_assay_corr[_key]
        _ax = _axes_bot[_col]
        _vals = _mat.values.astype(float)
        _im = _ax.imshow(_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        _disp_x = list(_mat.columns)
        _disp_y = [display_value_col(r) for r in _mat.index]
        _ax.set_xticks(range(len(_disp_x)))
        _ax.set_xticklabels(_disp_x, rotation=90, fontsize=_tick_fs)
        _ax.set_yticks(range(len(_disp_y)))
        _ax.set_yticklabels(_disp_y, fontsize=_tick_fs)
        _ax.set_title(_titles[1][_col], fontsize=_tick_fs + 1)

        for _i in range(_vals.shape[0]):
            for _j in range(_vals.shape[1]):
                _v = _vals[_i, _j]
                if not np.isnan(_v):
                    _ax.text(
                        _j, _i, f"{_v:.2f}", ha="center", va="center",
                        fontsize=_annot_fs, color="black" if abs(_v) < 0.5 else "white",
                    )

    _cax = _fig.add_subplot(_gs[:, 4])
    _fig.colorbar(_axes_top[0].images[0], cax=_cax, label="correlation")

    _out = paths.FIGURES / "s01_correlation_heatmaps.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FULL_WIDTH,
    assay_data,
    display_value_col,
    np,
    paths,
    pd,
    plt,
    schema,
    stats,
):
    _reported = assay_data[assay_data["value_col"].isin(schema.REPORTED_VALUE_COLS)].copy()
    _reported["metric_label"] = _reported.apply(
        lambda r: display_value_col(r["value_col"], r["condition"]), axis=1,
    )

    def _pivot_reported(df):
        return df.pivot_table(
            index="antibody_name", columns="metric_label",
            values="median", aggfunc="first",
        )

    def _corr_mat(wide, method):
        _cols = wide.columns
        _n = len(_cols)
        _mat = pd.DataFrame(np.nan, index=_cols, columns=_cols)
        for _i in range(_n):
            for _j in range(_i, _n):
                _valid = wide[[_cols[_i], _cols[_j]]].dropna()
                if len(_valid) < 5:
                    continue
                if method == "spearman":
                    _r, _ = stats.spearmanr(_valid.iloc[:, 0], _valid.iloc[:, 1])
                else:
                    _r, _ = stats.pearsonr(_valid.iloc[:, 0], _valid.iloc[:, 1])
                _mat.iloc[_i, _j] = _r
                _mat.iloc[_j, _i] = _r
        return _mat

    _wide_n3 = _pivot_reported(_reported[_reported["kind"] == "N3"])
    _wide_n4 = _pivot_reported(_reported[_reported["kind"] == "N4"])

    _corr = {}
    for _kind, _wide in [("N3", _wide_n3), ("N4", _wide_n4)]:
        for _method in ["spearman", "pearson"]:
            _corr[(_kind, _method)] = _corr_mat(_wide, _method)

    _tick_fs = FONT_SIZE_TICK - 3
    _annot_fs = FONT_SIZE_LEGEND - 3

    _fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.55))
    _gs = _fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.45)

    _titles = ("N3 Spearman", "N3 Pearson",
               "N4 Spearman", "N4 Pearson")
    _keys = [("N3", "spearman"), ("N3", "pearson"), ("N4", "spearman"), ("N4", "pearson")]

    _axes = [_fig.add_subplot(_gs[0, _c]) for _c in range(4)]

    for _col, _key in enumerate(_keys):
        _mat = _corr[_key]
        _ax = _axes[_col]
        _vals = _mat.values.astype(float)
        _im = _ax.imshow(_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        _ax.set_xticks(range(len(_mat.columns)))
        _ax.set_xticklabels(_mat.columns, rotation=90, fontsize=_tick_fs)
        _ax.set_yticks(range(len(_mat.columns)))
        _ax.set_yticklabels(_mat.columns, fontsize=_tick_fs)
        _ax.set_title(_titles[_col], fontsize=_tick_fs + 1)

        for _i in range(_vals.shape[0]):
            for _j in range(_vals.shape[1]):
                _v = _vals[_i, _j]
                if not np.isnan(_v):
                    _ax.text(
                        _j, _i, f"{_v:.2f}", ha="center", va="center",
                        fontsize=_annot_fs, color="black" if abs(_v) < 0.5 else "white",
                    )

    _cax = _fig.add_subplot(_gs[0, 4])
    _fig.colorbar(_axes[0].images[0], cax=_cax, label="correlation")

    _out = paths.FIGURES / "s01_correlation_heatmaps_reported_subset.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(assay_data, np, pd, schema):
    from prophet_ab.normalize import parse_n3_components as _parse

    _n3 = assay_data[
        (assay_data["kind"] == "N3")
        & assay_data["value_col"].isin(schema.REPORTED_VALUE_COLS)
    ].copy()

    _comps = _n3["antibody_name"].map(_parse)
    _n3["pair"] = _comps.map(lambda t: tuple(sorted(t)))

    _orient_counts = (
        _n3.groupby(["pair", "value_col", "condition"])["antibody_name"]
        .nunique()
        .reset_index(name="_n_orient")
    )
    swap_pair_keys = frozenset(
        _orient_counts.loc[_orient_counts["_n_orient"] == 2, "pair"]
    )
    _n3 = _n3[_n3["pair"].isin(swap_pair_keys)]

    _rows = []
    for (_pair, _vc, _cond), _grp in _n3.groupby(["pair", "value_col", "condition"]):
        if _grp["antibody_name"].nunique() != 2:
            continue
        _sorted = _grp.sort_values("median")
        _r_lo, _r_hi = _sorted.iloc[0], _sorted.iloc[1]
        _std_lo = _r_lo["std"] if np.isfinite(_r_lo["std"]) else 0.0
        _std_hi = _r_hi["std"] if np.isfinite(_r_hi["std"]) else 0.0
        _se_lo = _std_lo / np.sqrt(_r_lo["n_replicates"]) if _r_lo["n_replicates"] > 0 else 0.0
        _se_hi = _std_hi / np.sqrt(_r_hi["n_replicates"]) if _r_hi["n_replicates"] > 0 else 0.0
        _rows.append({
            "pair": _pair,
            "value_col": _vc,
            "condition": _cond,
            "lo": float(_r_lo["median"]),
            "hi": float(_r_hi["median"]),
            "se_lo": float(_se_lo),
            "se_hi": float(_se_hi),
        })

    swap_pairs = pd.DataFrame(_rows)

    _VC_ORDER = [
        "hihplc_normretentiontime", "smachplc_retentiontime",
        "hachplc_retentiontime",
        "acsins_delta_Lmax",
        "pr_score",
        "bvp_score_norm",
        "thermostability_tm1", "thermostability_tm2",
    ]
    _vc_rank = {vc: i for i, vc in enumerate(_VC_ORDER)}
    _COND_ORDER = {
        "1X PBS": 0, "His/NaCl, pH 6": 1, "His/Arg, pH 6": 2,
        "CHO": 0, "Ovalbumin": 1,
    }
    swap_combos = sorted(
        swap_pairs[["value_col", "condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None),
        key=lambda t: (_vc_rank.get(t[0], 999), _COND_ORDER.get(t[1], 999)),
    )
    swap_pair_keys = swap_pair_keys
    return swap_combos, swap_pair_keys, swap_pairs


@app.cell
def _(
    DATAPOINTS_COLORS,
    display_value_col,
    grid_figsize,
    np,
    paths,
    plt,
    stats,
    swap_combos,
    swap_pairs,
    wrap_title,
):
    from matplotlib.ticker import MaxNLocator

    def _sync_ticks(ax, nticks=3):
        _lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        _hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        _loc = MaxNLocator(nbins=nticks * 2)
        _candidates = [t for t in _loc.tick_values(_lo, _hi) if _lo <= t <= _hi]
        _ticks = [_candidates[0], (_candidates[0] + _candidates[-1]) / 2, _candidates[-1]]
        _span = _ticks[-1] - _ticks[0]
        _pad = _span * 0.30
        ax.set_xticks(_ticks)
        ax.set_yticks(_ticks)
        ax.set_xlim(_ticks[0] - _pad, _ticks[-1] + _pad)
        ax.set_ylim(_ticks[0] - _pad, _ticks[-1] + _pad)

    _ncols = 3
    _nrows = (len(swap_combos) + _ncols - 1) // _ncols
    _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_fw, _fh),
        layout="constrained",
    )
    _axes_flat = np.array(_axes).flatten()

    for _i, (_vc, _cond) in enumerate(swap_combos):
        _ax = _axes_flat[_i]
        _g = swap_pairs[(swap_pairs["value_col"] == _vc) & (swap_pairs["condition"] == _cond)]

        if _g.empty:
            _ax.set_visible(False)
            continue

        _lo, _hi = _g["lo"].values, _g["hi"].values

        _ax.scatter(
            _lo, _hi,
            s=14, alpha=0.7,
            edgecolors=DATAPOINTS_COLORS["gray"],
            linewidths=0.3,
        )

        _sync_ticks(_ax)
        _lim = _ax.get_xlim()
        _ax.plot(_lim, _lim, "k--", lw=0.6, alpha=0.5)

        _rho, _ = stats.spearmanr(_lo, _hi)
        _n = len(_g)
        _lbl = display_value_col(_vc, _cond)
        if _lbl.startswith("PR Score @ "):
            _lbl = f"PR-{_lbl.removeprefix('PR Score @ ')} Score"
        elif _lbl == "BVP Score":
            _lbl = "PR-BVP Score"
        else:
            _lbl = _lbl.replace(" @ ", ", ")
        _title = _lbl + f"\nρ={_rho:.2f}, n={_n}"
        _ax.set_title(wrap_title(_title, _cw))
        _ax.set_xlabel("min(A×B, B×A)")
        _ax.set_ylabel("max(A×B, B×A)")

    for _ax in _axes_flat[len(swap_combos):]:
        _ax.axis("off")

    _out = paths.FIGURES / "s01_swap_pair_scatters.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    display_value_col,
    grid_figsize,
    np,
    paths,
    plt,
    stats,
    swap_combos,
    swap_pairs,
    wrap_title,
):
    from matplotlib.ticker import MaxNLocator as _MaxNLocator

    def _sync_ticks(ax, nticks=3):
        _lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        _hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        _loc = _MaxNLocator(nbins=nticks * 2)
        _candidates = [t for t in _loc.tick_values(_lo, _hi) if _lo <= t <= _hi]
        _ticks = [_candidates[0], (_candidates[0] + _candidates[-1]) / 2, _candidates[-1]]
        _span = _ticks[-1] - _ticks[0]
        _pad = _span * 0.30
        ax.set_xticks(_ticks)
        ax.set_yticks(_ticks)
        ax.set_xlim(_ticks[0] - _pad, _ticks[-1] + _pad)
        ax.set_ylim(_ticks[0] - _pad, _ticks[-1] + _pad)

    _ncols = 3
    _nrows = (len(swap_combos) + _ncols - 1) // _ncols
    _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_fw, _fh),
        layout="constrained",
    )
    _axes_flat = np.array(_axes).flatten()

    for _i, (_vc, _cond) in enumerate(swap_combos):
        _ax = _axes_flat[_i]
        _g = swap_pairs[(swap_pairs["value_col"] == _vc) & (swap_pairs["condition"] == _cond)]

        if _g.empty:
            _ax.set_visible(False)
            continue

        _lo, _hi = _g["lo"].values, _g["hi"].values

        _ax.scatter(
            _lo, _hi,
            s=14, alpha=0.7,
            edgecolors=DATAPOINTS_COLORS["gray"],
            linewidths=0.3,
        )

        _sync_ticks(_ax)
        _lim = _ax.get_xlim()
        _ax.plot(_lim, _lim, "k--", lw=0.6, alpha=0.5)

        _r, _ = stats.pearsonr(_lo, _hi)
        _n = len(_g)
        _lbl = display_value_col(_vc, _cond)
        if _lbl.startswith("PR Score @ "):
            _lbl = f"PR-{_lbl.removeprefix('PR Score @ ')} Score"
        elif _lbl == "BVP Score":
            _lbl = "PR-BVP Score"
        else:
            _lbl = _lbl.replace(" @ ", ", ")
        _title = _lbl + f"\nr={_r:.2f}, n={_n}"
        _ax.set_title(wrap_title(_title, _cw))
        _ax.set_xlabel("min(A×B, B×A)")
        _ax.set_ylabel("max(A×B, B×A)")

    for _ax in _axes_flat[len(swap_combos):]:
        _ax.axis("off")

    _out = paths.FIGURES / "s01_swap_pair_scatters_pearson.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    display_value_col,
    equalize_axes,
    grid_figsize,
    np,
    paths,
    plt,
    stats,
    swap_combos,
    swap_pair_keys,
    swap_pairs,
    wrap_title,
):
    _ncols = 4
    _nrows = (len(swap_combos) + _ncols - 1) // _ncols
    _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_fw, _fh),
        layout="constrained",
    )
    _axes = list(np.array(_axes).flatten())

    for _i, (_vc, _cond) in enumerate(swap_combos):
        _ax = _axes[_i]
        _g = swap_pairs[(swap_pairs["value_col"] == _vc) & (swap_pairs["condition"] == _cond)]

        if _g.empty:
            _ax.set_visible(False)
            continue

        _lo, _hi = _g["lo"].values, _g["hi"].values
        _se_lo, _se_hi = _g["se_lo"].values, _g["se_hi"].values

        _ax.errorbar(
            _lo, _hi,
            xerr=_se_lo, yerr=_se_hi,
            fmt="none",
            ecolor=DATAPOINTS_COLORS["gray"],
            elinewidth=0.6, capsize=0, alpha=0.5,
            zorder=2,
        )
        _ax.scatter(
            _lo, _hi,
            s=30, alpha=0.8,
            color=DATAPOINTS_COLORS["blue"],
            edgecolors=DATAPOINTS_COLORS["gray"],
            linewidths=0.3,
            zorder=3,
        )

        _all = np.concatenate([_lo - _se_lo, _hi + _se_hi])
        _pad = (_all.max() - _all.min()) * 0.08 or 0.5
        _lim = (_all.min() - _pad, _all.max() + _pad)
        _ax.plot(_lim, _lim, "--", color=DATAPOINTS_COLORS["gray"], linewidth=0.5, alpha=0.5)
        _ax.set_xlim(_lim)
        _ax.set_ylim(_lim)
        equalize_axes(_ax)

        _rho, _ = stats.spearmanr(_lo, _hi)
        _lbl = display_value_col(_vc, _cond)
        if _lbl.startswith("PR Score @ "):
            _lbl = f"PR-{_lbl.removeprefix('PR Score @ ')} Score"
        elif _lbl == "BVP Score":
            _lbl = "PR-BVP Score"
        else:
            _lbl = _lbl.replace(" @ ", ", ")
        _title_text = wrap_title(_lbl, _cw)
        _ax.set_title(f"{_title_text}\nρ = {_rho:.2f}", fontsize=FONT_SIZE_TITLE)
        _ax.set_xlabel("min(A×B, B×A)", fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel("max(A×B, B×A)", fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)

    for _j in range(len(swap_combos), len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle(
        f"Orientation swap pairs with SE (n = {len(swap_pair_keys)})",
        fontsize=FONT_SIZE_TITLE, fontweight="bold",
    )

    _out = paths.FIGURES / "s01_swap_pair_scatters_se.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    assay_data,
    display_value_col,
    grid_figsize,
    np,
    paths,
    plt,
    schema,
    wrap_title,
):
    _n3_reported = assay_data[
        (assay_data["kind"] == "N3")
        & assay_data["value_col"].isin(schema.REPORTED_VALUE_COLS)
    ].copy()

    _VC_ORDER = [
        "hihplc_normretentiontime", "smachplc_retentiontime",
        "hachplc_retentiontime",
        "acsins_delta_Lmax",
        "pr_score",
        "bvp_score_norm",
        "thermostability_tm1", "thermostability_tm2",
    ]
    _vc_rank = {vc: i for i, vc in enumerate(_VC_ORDER)}
    _COND_ORDER = {
        "1X PBS": 0, "His/NaCl, pH 6": 1, "His/Arg, pH 6": 2,
        "CHO": 0, "Ovalbumin": 1,
    }

    _combos = sorted(
        _n3_reported.groupby(["value_col", "condition"])
        .size()
        .reset_index()[["value_col", "condition"]]
        .itertuples(index=False, name=None),
        key=lambda t: (_vc_rank.get(t[0], 999), _COND_ORDER.get(t[1], 999)),
    )

    _ncols = 3
    _nrows = (len(_combos) + _ncols - 1) // _ncols
    _fw, _fh, _cw = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(_fw, _fh),
        layout="constrained",
    )
    _axes_flat = np.array(_axes).flatten()

    for _i, (_vc, _cond) in enumerate(_combos):
        _ax = _axes_flat[_i]
        _vals = _n3_reported[
            (_n3_reported["value_col"] == _vc)
            & (_n3_reported["condition"] == _cond)
        ]["median"].dropna().values

        _ax.hist(
            _vals,
            bins=20,
            color=DATAPOINTS_COLORS["blue"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
        )

        _lbl = display_value_col(_vc, _cond)
        if _lbl.startswith("PR Score @ "):
            _lbl = f"PR-{_lbl.removeprefix('PR Score @ ')} Score"
        elif _lbl == "BVP Score":
            _lbl = "PR-BVP Score"
        else:
            _lbl = _lbl.replace(" @ ", ", ")
        _ax.set_title(
            wrap_title(f"{_lbl}\n(n={len(_vals)})", _cw),
            fontsize=FONT_SIZE_TITLE,
        )
        _ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)

    for _ax in _axes_flat[len(_combos):]:
        _ax.set_visible(False)

    _out = paths.FIGURES / "s01_n3_reported_histograms.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"Saved {_out.relative_to(paths.REPO_ROOT)}")
    plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
