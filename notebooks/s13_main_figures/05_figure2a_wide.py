"""s13 -- figure_2a_wide: panel A scatter grid in landscape for slides.

Transposed layout of figure_2_tiers Panel A: tiers flow left-to-right as
column groups instead of top-to-bottom as row groups.  Optimized for
16:9 slide decks.

Open interactively:    marimo edit notebooks/s13_main_figures/05_figure2a_wide.py
Re-run headless:       python notebooks/s13_main_figures/05_figure2a_wide.py
"""

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import spearmanr

    from prophet_ab import paths, schema
    from prophet_ab import normalize as nz
    from prophet_ab.features.transforms import (
        TRANSFORMS,
        BEST_TRANSFORM,
        SHORT_LABEL,
        TIER_COLORS,
        TIER_LABELS,
    )
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        equalize_axes,
        FONT_SIZE_LABEL,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
        FONT_SIZE_LEGEND,
    )

    set_manuscript_style()
    return (
        BEST_TRANSFORM,
        DATAPOINTS_COLORS,
        FONT_SIZE_LABEL,
        FONT_SIZE_LEGEND,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
        GridSpec,
        SHORT_LABEL,
        TIER_COLORS,
        TIER_LABELS,
        TRANSFORMS,
        equalize_axes,
        np,
        nz,
        paths,
        pd,
        plt,
        schema,
        spearmanr,
    )


@app.cell
def _(nz, paths, pd, schema):
    _summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    _components = pd.read_parquet(paths.S02 / "n3_components.parquet")

    _n4 = _summaries[_summaries["kind"] == schema.KIND_N4][
        ["antibody_name", "value_col", "condition", "median"]
    ].copy()
    _n4["parent"] = _n4["antibody_name"].map(nz.strip_isotype_suffix)

    _n3 = _summaries[_summaries["kind"] == schema.KIND_N3][
        ["antibody_name", "value_col", "condition", "median"]
    ].rename(columns={"median": "observed"})

    _n3p = _n3.merge(
        _components[["antibody_name", "parent_a", "parent_b"]],
        on="antibody_name",
    )
    _n3p = (
        _n3p.merge(
            _n4[["parent", "value_col", "condition", "median"]].rename(
                columns={"parent": "parent_a", "median": "pa_median"}
            ),
            on=["parent_a", "value_col", "condition"],
            how="left",
        )
        .merge(
            _n4[["parent", "value_col", "condition", "median"]].rename(
                columns={"parent": "parent_b", "median": "pb_median"}
            ),
            on=["parent_b", "value_col", "condition"],
            how="left",
        )
    )
    n3_with_parents = _n3p.dropna(subset=["pa_median", "pb_median"])
    return (n3_with_parents,)


@app.cell
def _(
    BEST_TRANSFORM,
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    SHORT_LABEL,
    TIER_COLORS,
    TIER_LABELS,
    TRANSFORMS,
    equalize_axes,
    n3_with_parents,
    np,
    paths,
    plt,
    spearmanr,
):
    from matplotlib.ticker import MaxNLocator as _MaxNLocator

    def _sync_ticks(ax, nticks=3):
        _lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        _hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        _loc = _MaxNLocator(nbins=nticks * 2)
        _candidates = [t for t in _loc.tick_values(_lo, _hi) if _lo <= t <= _hi]
        _ticks = [
            _candidates[0],
            (_candidates[0] + _candidates[-1]) / 2,
            _candidates[-1],
        ]
        _span = _ticks[-1] - _ticks[0]
        _pad = _span * 0.30
        ax.set_xticks(_ticks)
        ax.set_yticks(_ticks)
        ax.set_xlim(_ticks[0] - _pad, _ticks[-1] + _pad)
        ax.set_ylim(_ticks[0] - _pad, _ticks[-1] + _pad)

    # Transposed layout: tiers as column groups, metrics stacked vertically.
    # Grid cols: [T1, spacer, T2_acsins, T2_pr, spacer, T3]
    _LAYOUT = [
        (0, 0, 1, ("hihplc_normretentiontime", "default")),
        (1, 0, 1, ("smachplc_retentiontime", "default")),
        (2, 0, 1, ("hachplc_retentiontime", "default")),
        (0, 2, 2, ("acsins_delta_Lmax", "1X PBS")),
        (1, 2, 2, ("acsins_delta_Lmax", "His/NaCl, pH 6")),
        (2, 2, 2, ("acsins_delta_Lmax", "His/Arg, pH 6")),
        (0, 3, 2, ("pr_score", "CHO")),
        (1, 3, 2, ("pr_score", "Ovalbumin")),
        (2, 3, 2, ("bvp_score_norm", "default")),
        (0, 5, 3, ("thermostability_tm1", "Tm1")),
        (1, 5, 3, ("thermostability_tm2", "Tm2")),
    ]

    _FIG_W, _FIG_H = 11.0, 7.8
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    _gs = GridSpec(
        3,
        6,
        width_ratios=[1, 0.12, 1, 1, 0.12, 1],
        hspace=0.55,
        wspace=0.30,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.06,
        figure=fig,
    )

    _tier_axes = {1: [], 2: [], 3: []}

    for _r, _c, _tier, (_vc, _cond) in _LAYOUT:
        _ax = fig.add_subplot(_gs[_r, _c])
        _tier_axes[_tier].append(_ax)
        _color = TIER_COLORS[_tier]

        _g = n3_with_parents[
            (n3_with_parents["value_col"] == _vc)
            & (n3_with_parents["condition"] == _cond)
        ]
        _op = BEST_TRANSFORM[(_vc, _cond)]
        _pred = TRANSFORMS[_op](_g["pa_median"], _g["pb_median"]).values
        _obs = _g["observed"].values
        _mask = ~(np.isnan(_pred) | np.isnan(_obs))
        _pred, _obs = _pred[_mask], _obs[_mask]
        _rho, _ = spearmanr(_pred, _obs)
        _n = int(_mask.sum())

        _ax.scatter(
            _pred,
            _obs,
            s=26,
            alpha=0.75,
            color=_color,
            edgecolors=DATAPOINTS_COLORS["gray"],
            linewidths=0.3,
            zorder=2,
        )
        equalize_axes(_ax)
        _sync_ticks(_ax)
        _lim = _ax.get_xlim()
        _ax.plot(
            _lim,
            _lim,
            ls="--",
            color=DATAPOINTS_COLORS["gray"],
            lw=0.6,
            alpha=0.6,
            zorder=1,
        )
        _ax.set_box_aspect(1)
        _ax.set_title(
            SHORT_LABEL[(_vc, _cond)],
            fontsize=FONT_SIZE_TITLE,
            fontweight="semibold",
            pad=3,
        )
        _ax.set_xlabel(f"parental-{_op}", fontsize=FONT_SIZE_LABEL)
        if _c == 0:
            _ax.set_ylabel("bispecific value", fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
        _ax.text(
            0.04,
            0.96,
            f"ρ = {_rho:.2f}\nn = {_n}",
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=FONT_SIZE_LEGEND,
            color=DATAPOINTS_COLORS["gray"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        )

    for _tier, _axes in _tier_axes.items():
        _positions = [a.get_position() for a in _axes]
        _x0 = min(p.x0 for p in _positions)
        _x1 = max(p.x1 for p in _positions)
        _y_top = max(p.y1 for p in _positions)
        _label = TIER_LABELS[_tier][0].replace(": ", ":\n")
        fig.text(
            (_x0 + _x1) / 2,
            _y_top + 0.045,
            _label,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_TITLE + 3,
            fontweight="bold",
            color=TIER_COLORS[_tier],
        )

    _o = paths.FIGURES / "main" / "figure_2a_wide.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig
    return


if __name__ == "__main__":
    app.run()
