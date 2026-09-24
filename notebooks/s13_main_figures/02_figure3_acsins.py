"""s13 -- figure_3 & figure_4: AC-SINS enhancer / suppressor charge mechanism.

Writes reports/figures/main/figure_3.png and figure_4.png. Unique-pair
classification (orientations averaged) plus an error-propagation noise band.

  Figure 3
  Panel A  observed vs expected (parent-mean) AC-SINS delta-Lmax at PBS pH 7.4,
           with the 5x error-propagation noise band, the 17.51 nm enhancer
           threshold and the 5 nm suppressor ceiling; unique-pair classification
           (13 enhancers, 20 suppressors of 139 pairs).
  Panel B  arm-pair charges in the canonical lower-triangle layout, split into
           an enhancer-highlighted and a suppressor-highlighted sub-panel
           (same unit; all 139 pairs carry MOE ens_charge: 13 enhancers,
           20 suppressors).

  figure_3_version_2 classifies from the noise band only (no 17.51 / 5 nm
  ΔLmax cutoffs), updates Panel A accordingly, and uses those labels on the
  q1 vs q2 Panel B. Does not overwrite figure_3.png.

  Figure 4
  Panel A  charge-transform horserace (delta-AIC bar chart, single-predictor
           logistic regression).
  Panel B  enhancer best-match charge violin (|q1+q2|).
  Panel C  suppressor best-match charge violin (signed geometric mean).

Open interactively:    marimo edit notebooks/s13_main_figures/02_figure3_acsins.py
Re-run headless:       python notebooks/s13_main_figures/02_figure3_acsins.py
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
    from scipy.stats import mannwhitneyu
    from sklearn.linear_model import LogisticRegression

    from prophet_ab import paths, schema
    from prophet_ab import normalize as nz
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
        DATAPOINTS_COLORS,
        FONT_SIZE_LABEL,
        FONT_SIZE_LEGEND,
        FONT_SIZE_TICK,
        FONT_SIZE_TITLE,
        Line2D,
        LogisticRegression,
        equalize_axes,
        mannwhitneyu,
        mo,
        np,
        nz,
        paths,
        pd,
        plt,
        schema,
    )


@app.cell
def _(mo):
    mo.md("""
    # s13 -- figure_3 & figure_4: AC-SINS enhancer / suppressor charge mechanism

    Manuscript main figures for the Tier 2 self-association section,
    built from the s10 AC-SINS analysis (PBS pH 7.4 reference condition).

    **Figure 3** (AC-SINS classification + arm charges)
    - **Panel A**: observed bispecific vs expected (parent-mean) AC-SINS
      delta-Lmax with the 5x error-propagation noise band, 17.51 nm enhancer
      threshold and 5 nm suppressor ceiling; unique-pair classification.
    - **Panel B**: arm-pair MOE ens_charge in the canonical lower-triangle
      layout, enhancers and suppressors split into two sub-panels.

    **figure_3_version_2** reclassifies pairs from the noise band only (no
    17.51 nm / 5 nm ΔLmax cutoffs). Panel A drops those horizontal lines.
    Panel B is q1 vs q2 with the updated labels, marker size |q1| + |q2|,
    and marginal histograms.

    **Figure 4** (charge-transform horserace)
    - **Panel A**: charge-transform horserace (delta-AIC, single-predictor
      logistic regression).
    - **Panels B, C**: best-matched charge violins per population.

    """)
    return


@app.cell
def _(np, nz, paths, pd, schema):
    # PBS-only per-N3 AC-SINS: observed bispecific median, parent medians, and
    # expected = parent mean. Mirrors s10/01_classification.py.
    _summaries = pd.read_parquet(paths.S03 / "n3n4_per_antibody.parquet")
    _components = pd.read_parquet(paths.S02 / "n3_components.parquet")

    _acsins = _summaries[_summaries["value_col"] == "acsins_delta_Lmax"].copy()

    _n4 = _acsins[_acsins["kind"] == schema.KIND_N4][
        ["antibody_name", "condition", "median", "std"]
    ].copy()
    _n4["parent"] = _n4["antibody_name"].map(nz.strip_isotype_suffix)

    _n3 = _acsins[_acsins["kind"] == schema.KIND_N3][
        ["antibody_name", "condition", "median"]
    ].rename(columns={"median": "observed"})

    _n3p = _n3.merge(
        _components[["antibody_name", "parent_a", "parent_b"]],
        on="antibody_name",
    )
    _n3p = (
        _n3p.merge(
            _n4[["parent", "condition", "median"]].rename(
                columns={"parent": "parent_a", "median": "pa_median"}
            ),
            on=["parent_a", "condition"],
            how="left",
        )
        .merge(
            _n4[["parent", "condition", "median"]].rename(
                columns={"parent": "parent_b", "median": "pb_median"}
            ),
            on=["parent_b", "condition"],
            how="left",
        )
    )
    _n3p["expected"] = (_n3p["pa_median"] + _n3p["pb_median"]) / 2
    _n3p["residual"] = _n3p["observed"] - _n3p["expected"]

    # Error-propagation noise band (Ammar pattern, D-2026-06-05-ACSINS-AMMAR-ALIGN):
    # noise_pbs = sigma_noise = sqrt(sigma_bsab^2 + sigma_arm^2 / 2) from the
    # median replicate SD of N3 (sigma_bsab) and N4 (sigma_arm); band = 5 * noise.
    _pbs = _acsins[_acsins["condition"] == "1X PBS"]
    _sigma_arm = float(np.nanmedian(_pbs[_pbs["kind"] == schema.KIND_N4]["std"]))
    _sigma_bsab = float(np.nanmedian(_pbs[_pbs["kind"] == schema.KIND_N3]["std"]))
    noise_pbs = float(np.sqrt(_sigma_bsab**2 + _sigma_arm**2 / 2))

    n3_acsins_pbs = _n3p[_n3p["condition"] == "1X PBS"].dropna(
        subset=["residual"]
    ).copy()
    return n3_acsins_pbs, noise_pbs


@app.cell
def _(n3_acsins_pbs, noise_pbs, pd):
    # Unique-pair, average-then-classify (Ammar pattern, PBS): collapse the two
    # orientations, average observed, then classify the averaged pair. Enhancer
    # above band AND above the Arsiwala 17.51 nm threshold; suppressor below band
    # AND at/under the 5 nm floor. Single classification unit for all panels.
    NOISE_MULTIPLIER = 5
    ENHANCER_THRESHOLD = 17.51
    SUPPRESSOR_CEILING = 5.0
    _band = NOISE_MULTIPLIER * noise_pbs

    _g = n3_acsins_pbs.copy()
    _g["pair"] = _g.apply(
        lambda r: tuple(sorted([r["parent_a"], r["parent_b"]])), axis=1
    )
    _rows = []
    for _pair, _grp in _g.groupby("pair"):
        _med = {}
        for _, _rr in _grp.iterrows():
            _med[_rr["parent_a"]] = _rr["pa_median"]
            _med[_rr["parent_b"]] = _rr["pb_median"]
        _observed = float(_grp["observed"].mean())
        _expected = float(_grp["expected"].mean())
        _residual = _observed - _expected
        if _residual >= _band and _observed >= ENHANCER_THRESHOLD:
            _cat = "Enhancer"
        elif _residual <= -_band and _observed <= SUPPRESSOR_CEILING:
            _cat = "Suppressor"
        else:
            _cat = ""
        _rows.append({
            "pair": _pair,
            "parent_a": _pair[0],
            "parent_b": _pair[1],
            "pa_median": _med[_pair[0]],
            "pb_median": _med[_pair[1]],
            "observed": _observed,
            "expected": _expected,
            "residual": _residual,
            "category": _cat,
            "n_orientations": len(_grp),
        })

    classified_pbs = pd.DataFrame(_rows)
    band_half_width = _band
    print(
        f"Panel A (unique pairs, PBS): "
        f"{(classified_pbs['category'] == 'Enhancer').sum()} enhancers, "
        f"{(classified_pbs['category'] == 'Suppressor').sum()} suppressors, "
        f"{len(classified_pbs)} pairs"
    )
    return (
        ENHANCER_THRESHOLD,
        SUPPRESSOR_CEILING,
        band_half_width,
        classified_pbs,
    )


@app.cell
def _(classified_pbs, nz, paths, pd):
    # Charge view for Panels B/C: take the single pair-level classification from
    # Panel A and attach MOE ens_charge (pH 7.4) per arm, then drop pairs lacking
    # charge. No re-classification: one unit across the whole figure.
    _pairs = classified_pbs.copy()

    # Attach MOE ens_charge (pH 7.4) per arm, then drop pairs lacking charge.
    _n4f = pd.read_csv(paths.RAW_IN_SILICO_DIR / "MOE_properties.csv")
    _n4f["parent"] = _n4f["antibody_name"].map(nz.strip_isotype_suffix)
    _charge_map = _n4f.set_index("parent")["ens_charge"].to_dict()

    _pairs["q1"] = _pairs["parent_a"].map(_charge_map)
    _pairs["q2"] = _pairs["parent_b"].map(_charge_map)
    _pairs["xc"] = _pairs[["q1", "q2"]].max(axis=1)
    _pairs["yc"] = _pairs[["q1", "q2"]].min(axis=1)

    pairs_charged = _pairs.dropna(subset=["q1", "q2"]).copy()
    print(
        f"Panels B/C (pair-level, charge subset): "
        f"{(pairs_charged['category'] == 'Enhancer').sum()} enhancers, "
        f"{(pairs_charged['category'] == 'Suppressor').sum()} suppressors, "
        f"{(pairs_charged['category'] == '').sum()} not-classified "
        f"({len(pairs_charged)} of {len(_pairs)} pairs)"
    )
    return (pairs_charged,)


@app.cell
def _(LogisticRegression, np, pairs_charged, pd):
    # Six single-predictor charge-transform logistic horserace (Panel C bar).
    # Mirrors s10/02_charge_mechanism.py.
    TRANSFORM_ORDER = [
        "B1_abs_sum_q", "B2_q_prod", "T1_cancel_eff",
        "T2_cancel_mass", "T3_signed_gmean", "T4_mono_dipole",
    ]
    TRANSFORM_LABELS = {
        "B1_abs_sum_q":    "|q1+q2|\nsum",
        "B2_q_prod":       "q1*q2\nproduct",
        "T1_cancel_eff":   "|Sq|/(|q1|+|q2|)\ncancel. eff.",
        "T2_cancel_mass":  "min*opp\ncancel. mass",
        "T3_signed_gmean": "-sgn(q1q2)*sqrt|q1q2|\nsigned geom.",
        "T4_mono_dipole":  "|Sq|/|Dq|\nmono/dipole",
    }

    def _compute_transforms(df):
        _q1 = df["q1"].values
        _q2 = df["q2"].values
        _aq1 = np.abs(_q1)
        _aq2 = np.abs(_q2)
        return {
            "B1_abs_sum_q": np.abs(_q1 + _q2),
            "B2_q_prod": _q1 * _q2,
            "T1_cancel_eff": np.where(
                (_aq1 + _aq2) > 0, np.abs(_q1 + _q2) / (_aq1 + _aq2), 0.0,
            ),
            "T2_cancel_mass": np.minimum(_aq1, _aq2) * np.where(
                _q1 * _q2 < 0, 1.0, -1.0
            ),
            "T3_signed_gmean": -np.sign(_q1 * _q2) * np.sqrt(np.abs(_q1 * _q2)),
            "T4_mono_dipole": np.where(
                np.abs(_q1 - _q2) > 1e-9, np.abs(_q1 + _q2) / np.abs(_q1 - _q2), 0.0,
            ),
        }

    _nc = pairs_charged[pairs_charged["category"] == ""]
    _agg = pairs_charged[pairs_charged["category"] == "Enhancer"]
    _res = pairs_charged[pairs_charged["category"] == "Suppressor"]

    _rows = []
    for _side, _pos, _side_label in [
        ("enh", _agg, "Enhancer"),
        ("suppress", _res, "Suppressor"),
    ]:
        _all = pd.concat([_pos, _nc], ignore_index=True)
        _y = np.array([1] * len(_pos) + [0] * len(_nc))
        _transforms = _compute_transforms(_all)
        for _tname, _xvals in _transforms.items():
            _X = _xvals.reshape(-1, 1)
            _valid = ~(np.isnan(_X).ravel() | np.isinf(_X).ravel())
            _Xv, _yv = _X[_valid], _y[_valid]
            if len(np.unique(_yv)) < 2 or len(_Xv) < 5:
                continue
            _clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            _clf.fit(_Xv, _yv)
            _proba = _clf.predict_proba(_Xv)
            _ll = float(np.sum(
                _yv * np.log(_proba[:, 1] + 1e-15)
                + (1 - _yv) * np.log(_proba[:, 0] + 1e-15)
            ))
            _aic = 2 * 2 - 2 * _ll
            _rows.append({
                "side": _side, "side_label": _side_label, "transform": _tname,
                "n_pos": int(_yv.sum()), "n_neg": int(len(_yv) - _yv.sum()),
                "aic": _aic,
            })

    horserace = pd.DataFrame(_rows)
    for _side in horserace["side"].unique():
        _mask = horserace["side"] == _side
        horserace.loc[_mask, "delta_aic"] = (
            horserace.loc[_mask, "aic"] - horserace.loc[_mask, "aic"].min()
        )
    print(horserace[["side_label", "transform", "aic", "delta_aic"]].to_string())
    return TRANSFORM_LABELS, TRANSFORM_ORDER, horserace


@app.cell
def _(
    DATAPOINTS_COLORS,
    ENHANCER_THRESHOLD,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    Line2D,
    SUPPRESSOR_CEILING,
    band_half_width,
    classified_pbs,
    equalize_axes,
    np,
    pairs_charged,
    paths,
    plt,
):
    _CORAL = DATAPOINTS_COLORS["coral"]
    _GREEN = DATAPOINTS_COLORS["green"]
    _GRAY = DATAPOINTS_COLORS["gray"]
    _SLATE = DATAPOINTS_COLORS["slate"]

    fig = plt.figure(figsize=(7.5, 8.5), layout="constrained")
    _gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], wspace=0.10, hspace=0.12)
    _axA = fig.add_subplot(_gs[0, :])
    _axB_l = fig.add_subplot(_gs[1, 0])
    _axB_r = fig.add_subplot(_gs[1, 1])

    # ----- Panel A: classification scatter (observed vs expected), PBS ------
    _g = classified_pbs
    _lo = float(min(_g["expected"].min(), _g["observed"].min()) - 2)
    _hi = float(max(_g["expected"].max(), _g["observed"].max()) + 2)
    _diag = np.linspace(_lo, _hi, 200)

    # Backmost region highlights honoring the full classification rule:
    # enhancer = above the upper noise band AND above the 17.51 nm threshold
    # (light red); suppressor = below the lower noise band AND below the 5 nm
    # ceiling (light green).
    _agg_lo = np.maximum(_diag + band_half_width, ENHANCER_THRESHOLD)
    _res_hi = np.minimum(_diag - band_half_width, SUPPRESSOR_CEILING)
    _axA.fill_between(_diag, _agg_lo, _hi, where=_agg_lo < _hi,
                      color=_CORAL, alpha=0.07, zorder=-2)
    _axA.fill_between(_diag, _lo, _res_hi, where=_res_hi > _lo,
                      color=_GREEN, alpha=0.07, zorder=-2)

    _axA.fill_between(_diag, _diag - band_half_width, _diag + band_half_width,
                      alpha=0.12, color="gray", zorder=0)
    _axA.plot(_diag, _diag, "k--", lw=0.8, alpha=0.5, zorder=1)
    _axA.axhline(ENHANCER_THRESHOLD, color=_CORAL, ls=":", lw=1.2, alpha=0.8)
    _axA.axhline(SUPPRESSOR_CEILING, color=_GREEN, ls=":", lw=1.2, alpha=0.8)

    _nc = _g[_g["category"] == ""]
    _agg = _g[_g["category"] == "Enhancer"]
    _res = _g[_g["category"] == "Suppressor"]
    _axA.scatter(_nc["expected"], _nc["observed"], s=18, c=_GRAY, alpha=0.45,
                 edgecolors=_GRAY, linewidths=0.3, zorder=2)
    _axA.scatter(_agg["expected"], _agg["observed"], s=42, c=_CORAL, alpha=0.85,
                 edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA.scatter(_res["expected"], _res["observed"], s=42, c=_GREEN, alpha=0.85,
                 edgecolors=_SLATE, linewidths=0.4, zorder=5)

    _axA.text(_lo + 0.8, ENHANCER_THRESHOLD + 1.0,
              f"high\nself-association\nobs > {ENHANCER_THRESHOLD:.2f} nm",
              ha="left", va="bottom", fontsize=FONT_SIZE_LEGEND, color=_CORAL,
              fontweight="semibold", linespacing=1.1)
    _axA.text(_hi - 0.8, SUPPRESSOR_CEILING - 1.0,
              f"low\nself-association\nobs < {SUPPRESSOR_CEILING:.0f} nm",
              ha="right", va="top", fontsize=FONT_SIZE_LEGEND, color=_GREEN,
              fontweight="semibold", linespacing=1.1)

    _axA.set_xlim(_lo, _hi)
    _axA.set_ylim(_lo, _hi)
    equalize_axes(_axA)
    _axA.set_box_aspect(1)
    _axA.set_xlabel("Expected (parent mean) AC-SINS ΔLmax  [nm]",
                    fontsize=FONT_SIZE_LABEL)
    _axA.set_ylabel("Observed bispecific ΔLmax  [nm]", fontsize=FONT_SIZE_LABEL)
    _axA.set_title(
        "AC-SINS at PBS pH 7.4",
        fontsize=FONT_SIZE_TITLE, fontweight="semibold", loc="left",
    )
    _axA.tick_params(labelsize=FONT_SIZE_TICK)

    # ----- Panel B: arm-charge quadrant scatter, split ----------------------
    _ncc = pairs_charged[pairs_charged["category"] == ""]
    _aggc = pairs_charged[pairs_charged["category"] == "Enhancer"]
    _resc = pairs_charged[pairs_charged["category"] == "Suppressor"]
    _qmax = float(np.ceil(max(pairs_charged["q1"].abs().max(),
                              pairs_charged["q2"].abs().max())) + 1)

    for _ax, _hi_set, _hi_color, _hi_label, _other in [
        (_axB_l, _aggc, _CORAL, "Enhancers", _resc),
        (_axB_r, _resc, _GREEN, "Suppressors", _aggc),
    ]:
        _ax.fill_between([0, _qmax], 0, _qmax, color=_CORAL, alpha=0.04, zorder=0)
        _ax.fill_between([-_qmax, 0], -_qmax, 0, color=_CORAL, alpha=0.04, zorder=0)
        _ax.fill_between([0, _qmax], -_qmax, 0, color=_GREEN, alpha=0.04, zorder=0)
        _ax.axhline(0, color=_GRAY, lw=0.5, alpha=0.4, zorder=1)
        _ax.axvline(0, color=_GRAY, lw=0.5, alpha=0.4, zorder=1)
        _ax.plot([-_qmax, _qmax], [-_qmax, _qmax], "--", color=_GRAY, lw=0.8,
                 alpha=0.5, zorder=1)

        _ax.scatter(_ncc["xc"], _ncc["yc"], s=22, color=_GRAY, alpha=0.45,
                    edgecolors=_GRAY, linewidths=0.3, zorder=2)
        if len(_other):
            _ax.scatter(_other["xc"], _other["yc"], s=32, color=_GRAY,
                        alpha=0.30, edgecolors=_GRAY, linewidths=0.3, zorder=3)
        if len(_hi_set):
            _ax.scatter(_hi_set["xc"], _hi_set["yc"], s=85, color=_hi_color,
                        alpha=0.9, edgecolors=_SLATE, linewidths=0.4, zorder=5)

        _ax.text(_qmax * 0.30, _qmax * 0.70, "++ same sign\nreinforce",
                 ha="center", va="center", fontsize=FONT_SIZE_LEGEND,
                 color=_CORAL, fontweight="semibold", alpha=0.85)
        _ax.text(-_qmax * 0.70, -_qmax * 0.30, "−− same sign\nreinforce",
                 ha="center", va="center", fontsize=FONT_SIZE_LEGEND,
                 color=_CORAL, fontweight="semibold", alpha=0.85)
        _ax.text(_qmax * 0.55, -_qmax * 0.78, "+− opposite sign\ncancel",
                 ha="center", va="center", fontsize=FONT_SIZE_LEGEND,
                 color=_GREEN, fontweight="semibold", alpha=0.85)

        _ax.set_xlim(-_qmax, _qmax)
        _ax.set_ylim(-_qmax, _qmax)
        equalize_axes(_ax)
        _ax.set_box_aspect(1)
        _ax.set_xlabel("max(q1, q2)  stronger arm charge  [e]",
                       fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel("min(q1, q2)  weaker arm charge  [e]",
                       fontsize=FONT_SIZE_LABEL)
        _ax.set_title(_hi_label, fontsize=FONT_SIZE_TITLE, fontweight="semibold",
                      loc="left")
        _ax.tick_params(labelsize=FONT_SIZE_TICK)

    # ===== Panel letters (left of the left-aligned titles, clear of labels) ==
    for _ax, _letter, _dx in [(_axA, "A", -40), (_axB_l, "B", -40)]:
        _ax.annotate(_letter, xy=(0, 1), xycoords="axes fraction",
                     xytext=(_dx, 6), textcoords="offset points",
                     fontsize=16, fontweight="bold", ha="left", va="bottom")

    # ===== Single shared legend below all panels (consistent colors) =========
    _handles = [
        Line2D([], [], color=_GRAY, marker="o", linestyle="None", markersize=8,
               label="Not classified"),
        Line2D([], [], color=_CORAL, marker="o", linestyle="None", markersize=9,
               label="Enhancers"),
        Line2D([], [], color=_GREEN, marker="o", linestyle="None", markersize=9,
               label="Suppressors"),
    ]
    fig.legend(handles=_handles, loc="outside lower center", ncol=3,
               frameon=False, fontsize=FONT_SIZE_LEGEND,
               handletextpad=0.4, columnspacing=2.2)

    _o = paths.FIGURES / "main" / "figure_3.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    Line2D,
    band_half_width,
    classified_pbs,
    equalize_axes,
    np,
    pairs_charged,
    paths,
    plt,
):
    # figure_3_version_2: band-only classification (no ΔLmax cutoffs).
    # Panel B uses the same labels. Does not overwrite figure_3.png.
    _CORAL = DATAPOINTS_COLORS["coral"]
    _GREEN = DATAPOINTS_COLORS["green"]
    _GRAY = DATAPOINTS_COLORS["gray"]
    _SLATE = DATAPOINTS_COLORS["slate"]

    from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable

    _g = classified_pbs.copy()
    _g["category"] = np.where(
        _g["residual"].to_numpy() >= band_half_width,
        "Enhancer",
        np.where(
            _g["residual"].to_numpy() <= -band_half_width,
            "Suppressor",
            "",
        ),
    )
    _pairs = pairs_charged.drop(columns=["category"]).merge(
        _g[["parent_a", "parent_b", "category"]],
        on=["parent_a", "parent_b"],
        how="left",
    )
    _n_enh = int((_g["category"] == "Enhancer").sum())
    _n_sup = int((_g["category"] == "Suppressor").sum())
    print(
        f"figure_3_version_2 (band only): {_n_enh} enhancers, "
        f"{_n_sup} suppressors, {len(_g)} pairs"
    )

    _fig_v2 = plt.figure(figsize=(7.5, 9.2))
    _gs_v2 = _fig_v2.add_gridspec(
        2, 2, height_ratios=[1.35, 1.05],
        left=0.10, right=0.96, top=0.94, bottom=0.12,
        wspace=0.28, hspace=0.28,
    )
    _axA_v2 = _fig_v2.add_subplot(_gs_v2[0, :])
    _axB_l_v2 = _fig_v2.add_subplot(_gs_v2[1, 0])
    _axB_r_v2 = _fig_v2.add_subplot(_gs_v2[1, 1])

    # ----- Panel A: noise-band classification, no ΔLmax cutoffs ------------
    _lo = float(min(_g["expected"].min(), _g["observed"].min()) - 2)
    _hi = float(max(_g["expected"].max(), _g["observed"].max()) + 2)
    _diag = np.linspace(_lo, _hi, 200)
    _agg_lo = _diag + band_half_width
    _res_hi = _diag - band_half_width
    _axA_v2.fill_between(_diag, _agg_lo, _hi, where=_agg_lo < _hi,
                         color=_CORAL, alpha=0.07, zorder=-2)
    _axA_v2.fill_between(_diag, _lo, _res_hi, where=_res_hi > _lo,
                         color=_GREEN, alpha=0.07, zorder=-2)
    _axA_v2.fill_between(
        _diag, _diag - band_half_width, _diag + band_half_width,
        alpha=0.12, color="gray", zorder=0,
    )
    _axA_v2.plot(_diag, _diag, "k--", lw=0.8, alpha=0.5, zorder=1)

    _nc = _g[_g["category"] == ""]
    _agg = _g[_g["category"] == "Enhancer"]
    _res = _g[_g["category"] == "Suppressor"]
    _axA_v2.scatter(_nc["expected"], _nc["observed"], s=18, c=_GRAY, alpha=0.45,
                    edgecolors=_GRAY, linewidths=0.3, zorder=2)
    _axA_v2.scatter(_agg["expected"], _agg["observed"], s=42, c=_CORAL, alpha=0.85,
                    edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA_v2.scatter(_res["expected"], _res["observed"], s=42, c=_GREEN, alpha=0.85,
                    edgecolors=_SLATE, linewidths=0.4, zorder=5)

    _axA_v2.text(_lo + 0.8, _hi - 1.5, f"Enhancers  (n = {_n_enh})",
                 ha="left", va="top", fontsize=FONT_SIZE_LEGEND, color=_CORAL,
                 fontweight="semibold")
    _axA_v2.text(_hi - 0.8, _lo + 1.5, f"Suppressors  (n = {_n_sup})",
                 ha="right", va="bottom", fontsize=FONT_SIZE_LEGEND, color=_GREEN,
                 fontweight="semibold")
    _axA_v2.set_xlim(_lo, _hi)
    _axA_v2.set_ylim(_lo, _hi)
    equalize_axes(_axA_v2)
    _axA_v2.set_box_aspect(1)
    _axA_v2.set_xlabel("Expected (parent mean) AC-SINS ΔLmax  [nm]",
                       fontsize=FONT_SIZE_LABEL)
    _axA_v2.set_ylabel("Observed bispecific ΔLmax  [nm]", fontsize=FONT_SIZE_LABEL)
    _axA_v2.set_title(
        "AC-SINS at PBS pH 7.4",
        fontsize=FONT_SIZE_TITLE, fontweight="semibold", loc="left",
    )
    _axA_v2.tick_params(labelsize=FONT_SIZE_TICK)

    # ----- Panel B: q1 vs q2, size |q1|+|q2|, marginal histograms ----------
    _ncc = _pairs[_pairs["category"] == ""]
    _aggc = _pairs[_pairs["category"] == "Enhancer"]
    _resc = _pairs[_pairs["category"] == "Suppressor"]
    _qmax = float(np.ceil(max(_pairs["q1"].abs().max(),
                              _pairs["q2"].abs().max())) + 1)
    _bins = np.linspace(-_qmax, _qmax, 25)

    for _sc, _hi_set, _hi_color, _hi_label, _other in [
        (_axB_l_v2, _aggc, _CORAL, f"Enhancers  (n = {_n_enh})", _resc),
        (_axB_r_v2, _resc, _GREEN, f"Suppressors  (n = {_n_sup})", _aggc),
    ]:
        _s_nc = 6.0 + 5.0 * (
            np.abs(_ncc["q1"].to_numpy()) + np.abs(_ncc["q2"].to_numpy())
        )
        _sc.fill_between([0, _qmax], 0, _qmax, color=_CORAL, alpha=0.04, zorder=0)
        _sc.fill_between([-_qmax, 0], -_qmax, 0, color=_CORAL, alpha=0.04, zorder=0)
        _sc.fill_between([0, _qmax], -_qmax, 0, color=_GREEN, alpha=0.04, zorder=0)
        _sc.fill_between([-_qmax, 0], 0, _qmax, color=_GREEN, alpha=0.04, zorder=0)
        _sc.axhline(0, color=_GRAY, ls="--", lw=0.8, alpha=0.7, zorder=1)
        _sc.axvline(0, color=_GRAY, ls="--", lw=0.8, alpha=0.7, zorder=1)
        _sc.scatter(
            _ncc["q1"], _ncc["q2"], s=_s_nc, color=_GRAY, alpha=0.45,
            edgecolors=_GRAY, linewidths=0.3, zorder=2,
        )
        if len(_other):
            _s_ot = 6.0 + 5.0 * (
                np.abs(_other["q1"].to_numpy()) + np.abs(_other["q2"].to_numpy())
            )
            _sc.scatter(
                _other["q1"], _other["q2"], s=_s_ot, color=_GRAY,
                alpha=0.30, edgecolors=_GRAY, linewidths=0.3, zorder=3,
            )
        if len(_hi_set):
            _s_hi = 6.0 + 5.0 * (
                np.abs(_hi_set["q1"].to_numpy())
                + np.abs(_hi_set["q2"].to_numpy())
            )
            _sc.scatter(
                _hi_set["q1"], _hi_set["q2"], s=_s_hi,
                color=_hi_color, alpha=0.9, edgecolors=_SLATE, linewidths=0.4,
                zorder=5,
            )
        _sc.set_xlim(-_qmax, _qmax)
        _sc.set_ylim(-_qmax, _qmax)
        equalize_axes(_sc)
        _sc.set_box_aspect(1)
        _sc.set_xlabel("q1  arm charge  [e]", fontsize=FONT_SIZE_LABEL)
        _sc.set_ylabel("q2  arm charge  [e]", fontsize=FONT_SIZE_LABEL)
        _sc.tick_params(labelsize=FONT_SIZE_TICK)

        _div = _make_axes_locatable(_sc)
        _hx = _div.append_axes("top", size="18%", pad=0.06, sharex=_sc)
        _hy = _div.append_axes("right", size="18%", pad=0.06, sharey=_sc)
        _hx.hist(
            _pairs["q1"].to_numpy(), bins=_bins, color=_GRAY, alpha=0.40,
            density=True, orientation="vertical", histtype="stepfilled",
        )
        _hy.hist(
            _pairs["q2"].to_numpy(), bins=_bins, color=_GRAY, alpha=0.40,
            density=True, orientation="horizontal", histtype="stepfilled",
        )
        if len(_hi_set):
            _hx.hist(
                _hi_set["q1"].to_numpy(), bins=_bins, color=_hi_color, alpha=0.70,
                density=True, orientation="vertical", histtype="stepfilled",
            )
            _hy.hist(
                _hi_set["q2"].to_numpy(), bins=_bins, color=_hi_color, alpha=0.70,
                density=True, orientation="horizontal", histtype="stepfilled",
            )
        _hx.tick_params(labelbottom=False, labelleft=False, length=0)
        _hy.tick_params(labelbottom=False, labelleft=False, length=0)
        for _sp in list(_hx.spines.values()) + list(_hy.spines.values()):
            _sp.set_visible(False)
        _hx.set_facecolor("none")
        _hy.set_facecolor("none")
        _hx.set_xlim(-_qmax, _qmax)
        _hy.set_ylim(-_qmax, _qmax)
        _hx.set_title(_hi_label, fontsize=FONT_SIZE_TITLE,
                      fontweight="semibold", loc="left")

    for _let_ax, _letter, _dx in [(_axA_v2, "A", -40), (_axB_l_v2, "B", -40)]:
        _let_ax.annotate(
            _letter, xy=(0, 1), xycoords="axes fraction",
            xytext=(_dx, 6), textcoords="offset points",
            fontsize=16, fontweight="bold", ha="left", va="bottom",
        )

    _handles = [
        Line2D([], [], color=_GRAY, marker="o", linestyle="None", markersize=8,
               label="Not classified"),
        Line2D([], [], color=_CORAL, marker="o", linestyle="None", markersize=9,
               label="Enhancers"),
        Line2D([], [], color=_GREEN, marker="o", linestyle="None", markersize=9,
               label="Suppressors"),
    ]
    _fig_v2.legend(
        handles=_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01),
        frameon=False, fontsize=FONT_SIZE_LEGEND,
        handletextpad=0.4, columnspacing=1.8,
    )

    _o_v2 = paths.FIGURES / "main" / "figure_3_version_2.png"
    _o_v2.parent.mkdir(parents=True, exist_ok=True)
    _fig_v2.savefig(_o_v2, dpi=300, bbox_inches="tight")
    print(f"wrote {_o_v2.relative_to(paths.REPO_ROOT)}")
    _fig_v2
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    TRANSFORM_LABELS,
    TRANSFORM_ORDER,
    horserace,
    mannwhitneyu,
    np,
    pairs_charged,
    paths,
    plt,
):
    _CORAL = DATAPOINTS_COLORS["coral"]
    _GREEN = DATAPOINTS_COLORS["green"]
    _GRAY = DATAPOINTS_COLORS["gray"]
    _SLATE = DATAPOINTS_COLORS["slate"]

    fig_c = plt.figure(figsize=(13.0, 4.5), layout="constrained")
    _gs = fig_c.add_gridspec(1, 3, wspace=0.22)
    _axC_bar = fig_c.add_subplot(_gs[0, 0])
    _axC_a = fig_c.add_subplot(_gs[0, 1])
    _axC_r = fig_c.add_subplot(_gs[0, 2])

    _hr_a = horserace[horserace["side"] == "enh"].set_index("transform")
    _hr_r = horserace[horserace["side"] == "suppress"].set_index("transform")
    _delta_a = [_hr_a.loc[t, "delta_aic"] for t in TRANSFORM_ORDER]
    _delta_r = [_hr_r.loc[t, "delta_aic"] for t in TRANSFORM_ORDER]

    _y = np.arange(len(TRANSFORM_ORDER))[::-1]
    _h = 0.36
    _bars_a = _axC_bar.barh(_y + _h / 2, _delta_a, _h, color=_CORAL,
                            edgecolor=_SLATE, linewidth=0.6, label="Enhancers")
    _bars_r = _axC_bar.barh(_y - _h / 2, _delta_r, _h, color=_GREEN,
                            edgecolor=_SLATE, linewidth=0.6, label="Suppressors")
    for _bars, _deltas in [(_bars_a, _delta_a), (_bars_r, _delta_r)]:
        for _bar, _val in zip(_bars, _deltas):
            if abs(_val) < 1e-6:
                _axC_bar.text(_bar.get_width() + 0.2,
                              _bar.get_y() + _bar.get_height() / 2,
                              "* best", va="center", fontsize=FONT_SIZE_LEGEND,
                              fontweight="bold")
    _axC_bar.set_yticks(_y)
    _axC_bar.set_yticklabels([TRANSFORM_LABELS[t] for t in TRANSFORM_ORDER],
                             fontsize=FONT_SIZE_TICK)
    _axC_bar.set_xlabel("ΔAIC", fontsize=FONT_SIZE_LABEL)
    _axC_bar.set_title(
        "Charge-transform horserace\n(single-predictor logistic\nregression, "
        "ΔAIC)",
        fontsize=FONT_SIZE_TITLE, loc="left",
    )
    _axC_bar.axvline(0, color="black", lw=0.5)
    _axC_bar.axvspan(0, 2, color="gray", alpha=0.10, zorder=0)
    _axC_bar.grid(axis="x", ls=":", alpha=0.4)
    _axC_bar.set_xlim(left=-0.3)
    _axC_bar.set_box_aspect(1)

    def _signed_gmean(q1, q2):
        _a = np.asarray(q1, dtype=float)
        _b = np.asarray(q2, dtype=float)
        return -np.sign(_a * _b) * np.sqrt(np.abs(_a * _b))

    _ncc = pairs_charged[pairs_charged["category"] == ""]
    _aggc = pairs_charged[pairs_charged["category"] == "Enhancer"]
    _resc = pairs_charged[pairs_charged["category"] == "Suppressor"]
    _rng = np.random.default_rng(0)

    def _violin(_ax, _vals_nc, _vals_set, _title, _ylabel, _color, _set_label):
        _u, _p = mannwhitneyu(_vals_set, _vals_nc, alternative="greater")
        _parts = _ax.violinplot([_vals_nc, _vals_set], positions=[0, 1],
                                widths=0.7, showmeans=False, showmedians=False,
                                showextrema=False)
        for _i, _body in enumerate(_parts["bodies"]):
            _body.set_facecolor(_GRAY if _i == 0 else _color)
            _body.set_edgecolor(_SLATE)
            _body.set_linewidth(0.4)
            _body.set_alpha(0.65)
        for _pos, _vals, _c in [(0, _vals_nc, _GRAY), (1, _vals_set, _color)]:
            _jit = _rng.uniform(-0.08, 0.08, size=len(_vals))
            _ax.scatter(np.full(len(_vals), _pos) + _jit, _vals, s=16,
                        alpha=0.55, color=_c, edgecolors=_GRAY, linewidths=0.3)
            _ax.plot([_pos - 0.22, _pos + 0.22], [np.median(_vals)] * 2,
                     color="black", lw=2.2)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["n.c.", _set_label], fontsize=FONT_SIZE_TICK)
        _ax.set_ylabel(_ylabel, fontsize=FONT_SIZE_LABEL)
        _ax.set_title(_title, fontsize=FONT_SIZE_TITLE)
        _ax.text(0.5, 0.98,
                 f"MWU greater: p = {_p:.3g}\n"
                 f"med {np.median(_vals_nc):.2f} vs {np.median(_vals_set):.2f}",
                 transform=_ax.transAxes, ha="center", va="top",
                 fontsize=FONT_SIZE_LEGEND,
                 bbox=dict(facecolor="white", edgecolor=_GRAY, alpha=0.95, pad=3))
        _ax.grid(axis="y", ls=":", alpha=0.4)
        _ax.set_box_aspect(1)

    _violin(_axC_a, np.abs(_ncc["q1"].values + _ncc["q2"].values),
            np.abs(_aggc["q1"].values + _aggc["q2"].values),
            "Enhancer · charge", "|q1 + q2|  [e]", _CORAL, "enh")
    _violin(_axC_r, _signed_gmean(_ncc["q1"].values, _ncc["q2"].values),
            _signed_gmean(_resc["q1"].values, _resc["q2"].values),
            "Suppressor · charge", "−sgn(q1q2)·√|q1q2|", _GREEN, "sup")

    for _ax, _letter, _dx in [(_axC_bar, "A", -62), (_axC_a, "B", -40),
                               (_axC_r, "C", -40)]:
        _ax.annotate(_letter, xy=(0, 1), xycoords="axes fraction",
                     xytext=(_dx, 6), textcoords="offset points",
                     fontsize=16, fontweight="bold", ha="left", va="bottom")

    _o = paths.FIGURES / "main" / "figure_4.png"
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(_o, dpi=300, bbox_inches="tight")
    print(f"wrote {_o.relative_to(paths.REPO_ROOT)}")
    fig_c
    return


if __name__ == "__main__":
    app.run()
