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
        FULL_WIDTH,
        equalize_axes,
        grid_figsize,
        wrap_title,
    )

    set_manuscript_style()
    return (
        DATAPOINTS_COLORS,
        display_value_col,
        grid_figsize,
        mo,
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
    # s02 — Cross-platform reproducibility

    For each measurement that exists in both GDPa4 (N4
    monospecifics) and the published GDPa1 dataset, compute Pearson
    correlation across the antibodies in common, and plot. GDPa1
    is filtered to IgG1 antibodies only (D-2026-05-08-GDPA1-IGG1);
    SEC and thermostability are excluded (D-2026-05-08-DROP-SEC-THERMO).
    AC-SINS ΔLmax is z-scored per platform before plotting so the two
    campaigns share a common scale (Pearson r is unchanged by the
    linear rescaling).

    Decisions: D-2026-04-27-AC-SINS, D-2026-04-27-PR, D-2026-04-27-AGG,
    D-2026-05-08-DROP-SEC-THERMO, D-2026-05-08-GDPA1-IGG1.
    """)
    return


@app.cell
def _(paths, pd):
    n3n4 = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    _gdpa1_all = pd.read_parquet(paths.S03 / "gdpa1_per_antibody.parquet")
    gdpa1 = _gdpa1_all[_gdpa1_all["hc_subtype"] == "IgG1"]
    n4_map = pd.read_parquet(paths.S02 / "n4_gdpa1_map.parquet")
    return gdpa1, n3n4, n4_map


@app.cell
def _(display_value_col, gdpa1, n3n4, n4_map, pd, schema):
    # N4 only, attach the GDPa1 stripped-name key.
    n4 = n3n4[n3n4["kind"] == schema.KIND_N4].merge(
        n4_map[["n4_antibody_name", "n4_stripped"]],
        left_on="antibody_name", right_on="n4_antibody_name",
        how="left",
    )

    pairs = []
    for (this_vc, this_cond), gdpa1_col in schema.PROPHET_TO_GDPA1.items():
        ours = n4[(n4["value_col"] == this_vc) & (n4["condition"] == this_cond)][
            ["n4_stripped", "median"]
        ].rename(columns={"median": "this_median"})
        theirs = gdpa1[gdpa1["value_col"] == gdpa1_col][
            ["antibody_name", "median"]
        ].rename(columns={"antibody_name": "n4_stripped", "median": "gdpa1_median"})
        merged = ours.merge(theirs, on="n4_stripped", how="inner").dropna()
        merged["this_value_col"] = this_vc
        merged["this_condition"] = this_cond
        merged["gdpa1_col"] = gdpa1_col
        merged["panel"] = display_value_col(this_vc, this_cond)
        pairs.append(merged)
    paired = pd.concat(pairs, ignore_index=True)

    # AC-SINS ΔLmax: z-score per platform (per panel) so both campaigns
    # share a common scale. Pearson r is invariant to this linear rescaling.
    paired["this_plot"] = paired["this_median"]
    paired["gdpa1_plot"] = paired["gdpa1_median"]
    _acsins = paired["this_value_col"] == "acsins_delta_Lmax"
    for _src, _dst in (("this_median", "this_plot"), ("gdpa1_median", "gdpa1_plot")):
        paired.loc[_acsins, _dst] = (
            paired.loc[_acsins]
            .groupby("panel")[_src]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        )
    paired
    return (paired,)


@app.cell
def _(paired, pd, stats):
    _rows = []
    for _panel, _g in paired.groupby("panel", sort=False):
        if len(_g) < 3:
            _rows.append(dict(panel=_panel, n=len(_g), pearson_r=None, p=None))
            continue
        _r, _p = stats.pearsonr(_g["this_median"], _g["gdpa1_median"])
        _rows.append(dict(panel=_panel, n=len(_g), pearson_r=_r, p=_p,
                          this_value_col=_g["this_value_col"].iloc[0],
                          this_condition=_g["this_condition"].iloc[0],
                          gdpa1_col=_g["gdpa1_col"].iloc[0]))
    metrics = pd.DataFrame(_rows).sort_values("pearson_r", ascending=False)
    metrics
    return (metrics,)


@app.cell
def _(
    DATAPOINTS_COLORS,
    grid_figsize,
    metrics,
    paired,
    paths,
    plt,
    wrap_title,
):
    import numpy as _np
    from matplotlib.ticker import MaxNLocator

    def _sync_ticks(ax, nticks=3):
        """Make both axes share identical ticks with the range max on a tick."""
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

    _panels = list(metrics["panel"])
    _ncols = 4
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols)
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout="constrained")
    _axes_flat = axes.flatten()
    for _ax, _panel in zip(_axes_flat, _panels):
        _g = paired[paired["panel"] == _panel]
        _r = metrics.loc[metrics["panel"] == _panel, "pearson_r"].iloc[0]
        _n = metrics.loc[metrics["panel"] == _panel, "n"].iloc[0]
        _ax.scatter(
            _g["this_plot"], _g["gdpa1_plot"], s=14, alpha=0.7,
            edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
        )
        _sync_ticks(_ax)
        _lim = _ax.get_xlim()
        _ax.plot(_lim, _lim, "k--", lw=0.6, alpha=0.5)
        _label = _panel.replace("AC-SINS ΔLmax", "AC-SINS ΔLmax (norm)")
        if _label.startswith("PR Score @ "):
            _label = f"PR-{_label.removeprefix('PR Score @ ')} Score"
        else:
            _label = _label.replace(" @ ", ", ")
        _title = _label + f"\nr={_r:.2f}, n={_n}" if _r is not None else _label + f"\nn={_n}"
        _ax.set_title(wrap_title(_title, _cell_w))
        _ax.set_xlabel("GDPa4")
        _ax.set_ylabel("GDPa1")
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis("off")

    _out = paths.FIGURES / "s02_cross_platform_scatter.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"wrote {_out.relative_to(paths.REPO_ROOT)}")
    fig
    return


@app.cell
def _(metrics, paths):
    _out = paths.TABLES / "s02_cross_platform_correlations.csv"
    _out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(_out, index=False)
    print(f"wrote {_out.relative_to(paths.REPO_ROOT)}  rows={len(metrics)}")
    return


if __name__ == "__main__":
    app.run()
