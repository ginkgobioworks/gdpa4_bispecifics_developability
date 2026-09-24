"""s13 -- figure_2: three-tier compositional predictability (combined).

  Panel A  tier-colored scatter grid -- observed bsAb median vs the
           mechanistically-chosen arm operator on the two parents, grouped
           into three inheritance tiers.
  Panel B  mean/min/max Spearman rho heatmap, chosen operator boxed per row.

Open interactively:    marimo edit notebooks/s13_main_figures/01_figure2_tiers.py
Re-run headless:       python notebooks/s13_main_figures/01_figure2_tiers.py
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
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle
    from scipy.stats import spearmanr

    from prophet_ab import paths, schema
    from prophet_ab import normalize as nz
    from prophet_ab.features.transforms import (
        TRANSFORMS,
        BEST_TRANSFORM,
        SHORT_LABEL,
        PRIMARY_METRICS,
        TIER_DEFS,
        TIER_COLORS,
        TIER_LABELS,
    )
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
        equalize_axes,
        wrap_title,
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
        GridSpecFromSubplotSpec,
        PRIMARY_METRICS,
        Rectangle,
        SHORT_LABEL,
        TIER_COLORS,
        TIER_DEFS,
        TIER_LABELS,
        TRANSFORMS,
        equalize_axes,
        mo,
        np,
        nz,
        paths,
        pd,
        plt,
        schema,
        sns,
        spearmanr,
        wrap_title,
    )


@app.cell
def _(mo):
    mo.md("""
    # s13 -- figure_2: three-tier compositional predictability

    Combined figure built from the s03 compositional baselines.

    - **Panel A**: observed bispecific median vs the mechanistically
      chosen arm operator applied to the two parent monospecifics, one
      panel per assay, grouped into three inheritance tiers.
    - **Panel B**: Spearman rho of the mean / min / max operators per
      metric, chosen operator boxed, rows sorted by chosen-operator rho.
    """)
    return


@app.cell
def _(nz, paths, pd, schema):
    # Per-N4 (antibody, value_col, condition) -> median, parent name without
    # the _IgG1 tag so it joins to N3 components. Then attach both parents'
    # medians to every N3 (the join used by s03).
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
    PRIMARY_METRICS,
    SHORT_LABEL,
    TRANSFORMS,
    n3_with_parents,
    np,
    pd,
    spearmanr,
):
    # Per-metric x per-transform Spearman rho (Panel B matrix + Panel A stats).
    _rows = []
    for _vc, _cond in PRIMARY_METRICS:
        _g = n3_with_parents[
            (n3_with_parents["value_col"] == _vc)
            & (n3_with_parents["condition"] == _cond)
        ]
        _obs = _g["observed"].values
        _a = _g["pa_median"]
        _b = _g["pb_median"]

        _rec = {
            "value_col": _vc,
            "condition": _cond,
            "label": SHORT_LABEL[(_vc, _cond)],
            "chosen_transform": BEST_TRANSFORM[(_vc, _cond)],
        }
        for _t, _tfun in TRANSFORMS.items():
            _pred = _tfun(_a, _b).values
            _mask = ~(np.isnan(_pred) | np.isnan(_obs))
            _n = int(_mask.sum())
            if _n < 5:
                _rec[f"{_t}_rho"] = np.nan
                _rec[f"{_t}_n"] = _n
                continue
            _rho, _ = spearmanr(_pred[_mask], _obs[_mask])
            _rec[f"{_t}_rho"] = float(_rho)
            _rec[f"{_t}_n"] = _n
        _rows.append(_rec)

    rho_table = pd.DataFrame(_rows)
    rho_table
    return (rho_table,)


@app.cell
def _(PRIMARY_METRICS, TIER_DEFS):
    # Metrics dropped from this figure (SEC %monomer and Tonset).
    _EXCLUDE = {
        ("sehplc_pct_mono", "default"),
        ("thermostability_tonset", "Tonset"),
    }

    # Group consecutive tier rows into per-tier sections so each tier gets a
    # single title header, dropping excluded members and any emptied rows.
    _sections = []
    for _tier_num, _members in TIER_DEFS:
        _kept = [m for m in _members if m not in _EXCLUDE]
        if not _kept:
            continue
        if _sections and _sections[-1][0] == _tier_num:
            _sections[-1][1].append(_kept)
        else:
            _sections.append((_tier_num, [_kept]))
    tier_sections = _sections

    heatmap_metrics = [m for m in PRIMARY_METRICS if m not in _EXCLUDE]
    return heatmap_metrics, tier_sections


@app.cell
def _(
    BEST_TRANSFORM,
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    GridSpecFromSubplotSpec,
    Rectangle,
    SHORT_LABEL,
    TIER_COLORS,
    TIER_LABELS,
    TRANSFORMS,
    equalize_axes,
    heatmap_metrics,
    n3_with_parents,
    np,
    paths,
    plt,
    rho_table,
    sns,
    spearmanr,
    tier_sections,
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

    _FIG_W, _FIG_H = 10.5, 12.4
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    _outer = GridSpec(
        1, 2, width_ratios=[2.7, 1.0], wspace=0.10,
        left=0.05, right=0.97, top=0.97, bottom=0.04, figure=fig,
    )

    # ---- Panel A: scatter blocks with tier-title section headers --------
    # Flatten the per-tier sections into a row layout: each tier contributes
    # one thin title row followed by its scatter row(s).
    _row_specs = []
    for _tier_num, _rows in tier_sections:
        _row_specs.append(("title", _tier_num, None))
        for _members in _rows:
            _row_specs.append(("scatter", _tier_num, _members))

    _hr = [0.18 if _kind == "title" else 1.0 for _kind, _, _ in _row_specs]
    _gsA = GridSpecFromSubplotSpec(
        len(_row_specs), 3, subplot_spec=_outer[0, 0],
        height_ratios=_hr, hspace=0.62, wspace=0.22,
    )

    for _r, (_kind, _tier_num, _payload) in enumerate(_row_specs):
        if _kind == "title":
            _axt = fig.add_subplot(_gsA[_r, :])
            _axt.axis("off")
            _axt.text(0.0, 0.05, TIER_LABELS[_tier_num][0],
                      transform=_axt.transAxes, va="bottom", ha="left",
                      fontsize=FONT_SIZE_TITLE + 3, fontweight="bold",
                      color=TIER_COLORS[_tier_num])
            continue

        _color = TIER_COLORS[_tier_num]
        for _ci, (_vc, _cond) in enumerate(_payload):
            _ax = fig.add_subplot(_gsA[_r, _ci])
            _g = n3_with_parents[
                (n3_with_parents["value_col"] == _vc)
                & (n3_with_parents["condition"] == _cond)
            ]
            _op = BEST_TRANSFORM[(_vc, _cond)]
            _pred = TRANSFORMS[_op](_g["pa_median"], _g["pb_median"]).values
            _obs = _g["observed"].values
            _mask = ~(np.isnan(_pred) | np.isnan(_obs))
            _pred, _obs = _pred[_mask], _obs[_mask]
            _rho, _p = spearmanr(_pred, _obs)
            _n = int(_mask.sum())

            _ax.scatter(_pred, _obs, s=26, alpha=0.75, color=_color,
                        edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                        zorder=2)
            equalize_axes(_ax)
            _sync_ticks(_ax)
            _lim = _ax.get_xlim()
            _ax.plot(_lim, _lim, ls="--", color=DATAPOINTS_COLORS["gray"],
                     lw=0.6, alpha=0.6, zorder=1)
            _ax.set_box_aspect(1)
            _ax.set_title(SHORT_LABEL[(_vc, _cond)], fontsize=FONT_SIZE_TITLE,
                          fontweight="semibold", pad=3)
            _ax.set_xlabel(f"parental-{_op}", fontsize=FONT_SIZE_LABEL)
            if _ci == 0:
                _ax.set_ylabel("bispecific value", fontsize=FONT_SIZE_LABEL)
            _ax.tick_params(labelsize=FONT_SIZE_TICK)
            _ax.text(0.04, 0.96, f"ρ = {_rho:.2f}\nn = {_n}",
                     transform=_ax.transAxes, va="top", ha="left",
                     fontsize=FONT_SIZE_LEGEND, color=DATAPOINTS_COLORS["gray"],
                     bbox=dict(facecolor="white", edgecolor="none",
                               alpha=0.85, pad=2))

    # ---- Panel B: small square rho heatmap, vertically centered ---------
    _hm_set = set(heatmap_metrics)
    _hm = rho_table[rho_table.apply(
        lambda r: (r["value_col"], r["condition"]) in _hm_set, axis=1)].copy()

    _shown = ["mean", "min", "max"]
    _mat = _hm[[f"{_t}_rho" for _t in _shown]].copy()
    _mat.index = _hm["label"].values
    _mat.columns = _shown

    _panel_a_order = []
    for _tn, _rows in tier_sections:
        for _members in _rows:
            for _vc, _cond in _members:
                _panel_a_order.append(SHORT_LABEL[(_vc, _cond)])
    _order = [l for l in _panel_a_order if l in _mat.index]
    _mat = _mat.reindex(_order)

    _mat.index = [l.replace("AC-SINS ", "AC-SINS\n") for l in _mat.index]

    # Place the heatmap with explicit, square-celled geometry and center it
    # vertically in the figure. Box aspect = 3 cols : N rows keeps cells square.
    _cell = 0.46  # inches per heatmap cell (bumped for two-line AC-SINS labels)
    _hm_w = (3 * _cell) / _FIG_W
    _hm_h = (len(_mat) * _cell) / _FIG_H
    # Sit the heatmap well clear of Panel A: its long y-tick labels extend to
    # the left of _hm_x, so this offset is what defines the A/B separation.
    _hm_x = 0.86
    _hm_y = (1.0 - _hm_h) / 2.0
    _axB = fig.add_axes([_hm_x, _hm_y, _hm_w, _hm_h])
    _cax = fig.add_axes([_hm_x + _hm_w + 0.012, _hm_y + _hm_h * 0.2,
                         0.009, _hm_h * 0.6])
    sns.heatmap(
        _mat.abs(), ax=_axB, cmap="RdYlGn", vmin=0, vmax=1,
        annot=_mat, fmt=".2f",
        annot_kws=dict(size=FONT_SIZE_LEGEND - 1, fontweight="bold"),
        cbar_ax=_cax, cbar_kws=dict(label="|Spearman ρ|"),
        linewidths=0.5, linecolor="white",
    )
    for _i, _label in enumerate(_mat.index):
        _orig = _label.replace("AC-SINS\n", "AC-SINS ")
        _row = _hm[_hm["label"] == _orig].iloc[0]
        _chosen = _row["chosen_transform"]
        if _chosen not in _mat.columns:
            continue
        _j = list(_mat.columns).index(_chosen)
        _axB.add_patch(Rectangle((_j + 0.04, _i + 0.04), 0.92, 0.92,
                                 fill=False, edgecolor="#1A1A2E", lw=2.5,
                                 zorder=4))
    _axB.set_title(wrap_title("Spearman ρ\n(boxed = chosen operator)",
                              3.2), fontsize=FONT_SIZE_TITLE)
    _axB.set_xlabel("")
    _axB.set_ylabel("")
    _axB.tick_params(axis="y", labelsize=FONT_SIZE_TICK, rotation=0)
    _axB.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    for _lab in _axB.get_yticklabels():
        _lab.set_fontweight("semibold")
    for _lab in _axB.get_xticklabels():
        _lab.set_fontweight("semibold")

    # ---- Panel letters --------------------------------------------------
    fig.text(0.03, 0.99, "A", fontsize=16, fontweight="bold",
             color="black", ha="left", va="top")
    fig.text(0.77, _hm_y + _hm_h + 0.03, "B", fontsize=16, fontweight="bold",
             color="black", ha="left", va="top")

    _o = paths.FIGURES / "main" / "figure_2_tiers.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig
    return


if __name__ == "__main__":
    app.run()
