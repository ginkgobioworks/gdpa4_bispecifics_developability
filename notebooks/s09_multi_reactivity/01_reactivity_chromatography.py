"""s09 -- Multi-reactivity vs. chromatographic character

HIC (hydrophobic character) vs. HAC (charge character), colored by
self-interaction (AC-SINS) and polyreactivity (PR, BVP) measures.

Open interactively:    marimo edit notebooks/s09_multi_reactivity/01_reactivity_chromatography.py
Re-run headless:       python notebooks/s09_multi_reactivity/01_reactivity_chromatography.py
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
    from matplotlib.lines import Line2D
    from prophet_ab import paths, schema
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        WHITE_TO_PURPLE,
        FULL_WIDTH,
        FONT_SIZE_TICK,
        FONT_SIZE_LABEL,
        FONT_SIZE_TITLE,
        grid_figsize,
        wrap_title,
    )

    set_manuscript_style()
    return (
        DATAPOINTS_COLORS,
        FONT_SIZE_LABEL,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
        FULL_WIDTH,
        WHITE_TO_PURPLE,
        grid_figsize,
        mo,
        np,
        paths,
        pd,
        plt,
        wrap_title,
    )


@app.cell
def _(mo):
    mo.md("""
    # s09 -- Multi-reactivity vs. chromatographic character

    HIC retention time (hydrophobic character) on X, HAC retention time
    (charge character) on Y, colored by reactivity measures across six
    panels: AC-SINS self-interaction (three buffer conditions), PR
    polyreactivity (CHO, Ovalbumin), and BVP non-specific binding.
    """)
    return


@app.cell
def _(paths, pd):
    per_ab = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    return (per_ab,)


@app.cell
def _(per_ab):
    _metrics = [
        ("hihplc_normretentiontime", "default", "hic"),
        ("hachplc_retentiontime", "default", "hac"),
        ("acsins_delta_Lmax", "1X PBS", "acsins_pbs"),
        ("acsins_delta_Lmax", "His/Arg, pH 6", "acsins_his_arg"),
        ("acsins_delta_Lmax", "His/NaCl, pH 6", "acsins_his_nacl"),
        ("pr_score", "CHO", "pr_cho"),
        ("pr_score", "Ovalbumin", "pr_ova"),
        ("bvp_score_norm", "default", "bvp"),
    ]

    _frames = []
    for _vc, _cond, _alias in _metrics:
        _sub = (
            per_ab[(per_ab["value_col"] == _vc) & (per_ab["condition"] == _cond)]
            [["antibody_name", "kind", "is_control", "median"]]
            .rename(columns={"median": _alias})
        )
        _frames.append((_alias, _sub))

    wide = _frames[0][1]
    for _alias, _f in _frames[1:]:
        wide = wide.merge(
            _f[["antibody_name", _alias]], on="antibody_name", how="outer",
        )

    print(f"Antibodies with any data: {len(wide)}")
    return (wide,)


@app.cell
def _(np, wide):
    _n4 = wide[wide["kind"] == "N4"].dropna(subset=["hic", "hac"]).copy().reset_index(drop=True)
    _pts = _n4[["hic", "hac"]].values
    _n = len(_pts)
    _is_pareto = np.ones(_n, dtype=bool)
    for _i in range(_n):
        if not _is_pareto[_i]:
            continue
        for _j in range(_n):
            if _i == _j:
                continue
            if (
                _pts[_j, 0] >= _pts[_i, 0]
                and _pts[_j, 1] >= _pts[_i, 1]
                and (_pts[_j, 0] > _pts[_i, 0] or _pts[_j, 1] > _pts[_i, 1])
            ):
                _is_pareto[_i] = False
                break
    _fpts = _pts[_is_pareto]
    frontier_pts = _fpts[np.argsort(_fpts[:, 0])]
    print(f"N4 Pareto frontier: {len(frontier_pts)} points")
    return (frontier_pts,)


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FULL_WIDTH,
    WHITE_TO_PURPLE,
    grid_figsize,
    paths,
    plt,
    wide,
    wrap_title,
):

    PANELS = [
        ("acsins_pbs", "AC-SINS ΔLmax 1× PBS"),
        ("acsins_his_arg", "AC-SINS ΔLmax His/Arg pH 6"),
        ("acsins_his_nacl", "AC-SINS ΔLmax His/NaCl pH 6"),
        ("pr_cho", "PR-CHO Score"),
        ("pr_ova", "PR-Ovalbumin Score"),
        ("bvp", "PR-BVP Score"),
    ]

    _PAD = 0.05
    _hic_all = wide["hic"].dropna()
    _hac_all = wide["hac"].dropna()
    _hic_range = _hic_all.max() - _hic_all.min()
    _hac_range = _hac_all.max() - _hac_all.min()
    _xlim = (_hic_all.min() - _PAD * _hic_range, _hic_all.max() + _PAD * _hic_range)
    _ylim = (_hac_all.min() - _PAD * _hac_range, _hac_all.max() + _PAD * _hac_range)

    _clims = {}
    for _col, _ in PANELS:
        _vals = wide[_col].dropna()
        _clims[_col] = (_vals.min(), _vals.max())

    _fw, _fh, _cell_w = grid_figsize(2, 3, full_width=FULL_WIDTH)

    def plot_reactivity_grid(df, suptitle, save_name):
        _fig, _axes = plt.subplots(2, 3, figsize=(_fw, _fh), layout="constrained")
        _axes = _axes.ravel()
        for _i, (_col, _label) in enumerate(PANELS):
            _ax = _axes[_i]
            _mask = df[["hic", "hac", _col]].notna().all(axis=1)
            _sub = df[_mask]
            _vmin, _vmax = _clims[_col]
            if len(_sub) == 0:
                _ax.set_title(wrap_title(_label, _cell_w), fontsize=FONT_SIZE_TITLE)
                _ax.text(0.5, 0.5, "no data", transform=_ax.transAxes,
                         ha="center", va="center", fontsize=FONT_SIZE_TICK)
                continue
            _sc = _ax.scatter(
                _sub["hic"], _sub["hac"],
                c=_sub[_col], cmap=WHITE_TO_PURPLE,
                vmin=_vmin, vmax=_vmax,
                s=30, edgecolors=DATAPOINTS_COLORS["gray"],
                linewidths=0.3, alpha=0.85,
            )
            _cb = _fig.colorbar(_sc, ax=_ax, shrink=0.8, pad=0.02)
            _cb.ax.tick_params(labelsize=FONT_SIZE_TICK)
            _ax.set_xlim(_xlim)
            _ax.set_ylim(_ylim)
            _ax.set_xlabel("HIC RT (norm)", fontsize=FONT_SIZE_LABEL)
            _ax.set_ylabel("HAC RT", fontsize=FONT_SIZE_LABEL)
            _ax.set_title(wrap_title(_label, _cell_w), fontsize=FONT_SIZE_TITLE)
            _ax.tick_params(labelsize=FONT_SIZE_TICK)
        _fig.suptitle(suptitle, fontsize=FONT_SIZE_TITLE)
        _out = paths.FIGURES / save_name
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=300, bbox_inches="tight")
        print(f"wrote {_out.relative_to(paths.REPO_ROOT)}")
        return _fig

    fig_n4 = plot_reactivity_grid(
        wide[wide["kind"] == "N4"],
        "Monospecifics",
        "s09_reactivity_vs_chromatography_n4.png",
    )
    fig_n3 = plot_reactivity_grid(
        wide[wide["kind"] == "N3"],
        "Bispecifics",
        "s09_reactivity_vs_chromatography_n3.png",
    )
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FULL_WIDTH,
    WHITE_TO_PURPLE,
    frontier_pts,
    np,
    paths,
    plt,
    wide,
    wrap_title,
):

    _PR_PANELS = [
        ("pr_cho", "PR-CHO Score"),
        ("pr_ova", "PR-Ovalbumin Score"),
        ("bvp", "PR-BVP Score"),
    ]
    _ROW_KINDS = [
        ("N4", "Monospecifics"),
        ("N3", "Bispecifics"),
    ]

    _PAD = 0.05
    _hic_all = wide["hic"].dropna()
    _hac_all = wide["hac"].dropna()
    _hic_range = _hic_all.max() - _hic_all.min()
    _hac_range = _hac_all.max() - _hac_all.min()
    _xlim = (_hic_all.min() - _PAD * _hic_range, _hic_all.max() + _PAD * _hic_range)
    _ylim = (_hac_all.min() - _PAD * _hac_range, _hac_all.max() + _PAD * _hac_range)

    _clims = {}
    for _col, _ in _PR_PANELS:
        _vals = wide[_col].dropna()
        _clims[_col] = (_vals.min(), _vals.max())

    _cell_w = 2.0
    _fw = FULL_WIDTH + 0.6
    _fh = 2 * _cell_w + 1.0
    _fig, _axes = plt.subplots(
        2, 3, figsize=(_fw, _fh), layout="constrained",
    )

    for _row, (_kind, _row_label) in enumerate(_ROW_KINDS):
        _df = wide[wide["kind"] == _kind]
        for _col_idx, (_col, _panel_label) in enumerate(_PR_PANELS):
            _ax = _axes[_row, _col_idx]
            _mask = _df[["hic", "hac", _col]].notna().all(axis=1)
            _sub = _df[_mask].sort_values(_col, ascending=True)
            _vmin, _vmax = _clims[_col]
            if len(_sub) == 0:
                _ax.set_title(wrap_title(_panel_label, _cell_w), fontsize=FONT_SIZE_TITLE)
                _ax.text(0.5, 0.5, "no data", transform=_ax.transAxes,
                         ha="center", va="center", fontsize=FONT_SIZE_TICK)
                continue
            _sc = _ax.scatter(
                _sub["hic"], _sub["hac"],
                c=_sub[_col], cmap=WHITE_TO_PURPLE,
                vmin=_vmin, vmax=_vmax,
                s=30, edgecolors=DATAPOINTS_COLORS["gray"],
                linewidths=0.3, alpha=0.85, zorder=2,
            )
            _cb = _fig.colorbar(_sc, ax=_ax, shrink=0.7, pad=0.03,
                                fraction=0.046, aspect=20)
            _cb.ax.tick_params(labelsize=FONT_SIZE_TICK)
            _ax.set_box_aspect(1)
            _ax.set_xlim(_xlim)
            _ax.set_ylim(_ylim)
            _ax.set_xlabel("HIC RT (norm)", fontsize=FONT_SIZE_LABEL)
            _ax.set_ylabel("HAC RT", fontsize=FONT_SIZE_LABEL)
            if _row == 0:
                _ax.set_title(wrap_title(_panel_label, _cell_w), fontsize=FONT_SIZE_TITLE)
            _ax.tick_params(labelsize=FONT_SIZE_TICK)
            _ext_x = np.r_[_xlim[0], frontier_pts[:, 0], _xlim[1]]
            _ext_y = np.r_[frontier_pts[0, 1], frontier_pts[:, 1], frontier_pts[-1, 1]]
            _ax.plot(_ext_x, _ext_y, ls=":", color=DATAPOINTS_COLORS["coral"],
                     lw=1.2, zorder=3)

        _axes[_row, 0].annotate(
            _row_label, xy=(0, 0.5),
            xytext=(-55, 0),
            xycoords="axes fraction", textcoords="offset points",
            ha="center", va="center", fontsize=FONT_SIZE_TITLE,
            rotation=90, fontweight="bold",
        )

    _out = paths.FIGURES / "s09_polyreactivity_vs_chromatography.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches="tight")
    print(f"wrote {_out.relative_to(paths.REPO_ROOT)}")
    fig_pr = _fig
    return


if __name__ == "__main__":
    app.run()
