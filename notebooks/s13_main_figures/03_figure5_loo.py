"""s13 -- figure_5 & figure_6: leave-one-bispecific-out predictive modeling.

Builds two manuscript main figures from the s05/s07 LOO pipeline data in the
house datapoints_figures style.

figure_5 (LOO design + model/config performance sweep):
  Panel A  schematic of the leave-one-bispecific-out design (split, featurize,
           sweep classical ML), embedded from the provided diagram image.
  Panel B  Spearman rho distribution by model and feature config across all
           labels (parent-disjoint LOO).

figure_5_version_2 keeps Panel A and replaces the forest with grouped bars:
  best parental combining-function (mean/min/max), best supervised model
  under parent-disjoint LOO (excluding compositional_baseline), and
  in-silico-only LOO. Assays sorted by parental Spearman ρ descending.

figure_5_version_3 keeps Panel A and plots two Δρ bars per assay (best
supervised LOO and in-silico-only LOO, each minus the best parental
operator), same assay order.

figure_6 (feature-group importance + marginal value over baseline):
  Panel A  per-label Spearman rho decomposed into feature-group contributions
           (stacked, three subset labels, best operator config x model).
  Panel B  marginal value of the best supervised model over the compositional
           baseline, per label (dumbbell).

Open interactively:    marimo edit notebooks/s13_main_figures/03_figure5_loo.py
Re-run headless:       python notebooks/s13_main_figures/03_figure5_loo.py
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
    from matplotlib.gridspec import GridSpec

    from prophet_ab import paths, schema
    from prophet_ab.features.naming import (
        display_config,
        display_label,
        display_feature,
        parse_feature_name,
        parse_label_name,
    )
    from prophet_ab.features.transforms import SHORT_LABEL
    from datapoints_figures import (
        set_manuscript_style,
        DATAPOINTS_COLORS,
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
        GridSpec,
        SHORT_LABEL,
        display_config,
        display_label,
        mo,
        np,
        parse_feature_name,
        parse_label_name,
        paths,
        pd,
        plt,
        schema,
        sns,
    )


@app.cell
def _(mo):
    mo.md("""
    # s13 -- figure_5 & figure_6: leave-one-bispecific-out predictive modeling

    Two manuscript main figures built from the s05 CV sweep and the s07 LOO
    analysis.

    **figure_5 -- LOO design + model/config performance sweep**

    - **Panel A**: schematic of the parent-disjoint leave-one-bispecific-out
      design (split / featurize / sweep), embedded from
      `reports/manual_input_materials/loo_diagram.png`.
    - **Panel B**: Spearman rho by model x feature config across all labels
      (parent-disjoint LOO).

    **figure_5_version_2** keeps Panel A and replaces the forest with grouped
    bars (best parental operator / best supervised LOO / in-silico-only LOO).
    Assays ordered by parental Spearman ρ descending.

    **figure_5_version_3** keeps Panel A and plots Δρ vs the best parental
    operator for best supervised LOO and in-silico-only LOO (same order).

    **figure_6 -- feature-group importance + marginal value over baseline**

    - **Panel A**: per-label Spearman rho split into feature-group
      contributions (stacked) for the three subset labels, taking the best
      operator (non-arm) in-silico config x model per label.
    - **Panel B**: dumbbell of the compositional baseline (mean of parents)
      vs the best supervised model per label, with per-label rho gain.

    """)
    return


@app.cell
def _(display_label, paths, pd, schema):
    from prophet_ab.features.naming import parse_label_name as _parse_label

    # Manuscript label set: drop the deprecated raw AC-SINS Lmax (the signed
    # ΔLmax is reported instead) and IntactMS Purity (%), matching the 13
    # labels in the external figure_5 reference.
    _DROP_VALUE_COLS = set(schema.DEPRECATED_VALUE_COLS) | {"purity_pct"}

    def _keep_label(label_col: str) -> bool:
        _vc, _ = _parse_label(label_col)
        return _vc not in _DROP_VALUE_COLS

    # LOO per-(label, config, model) metrics and the permutation-importance
    # long table that drives the stacked feature-group decomposition.
    m = pd.read_parquet(paths.S05 / "cv_metrics_loo.parquet")
    m = m[m["label"].map(_keep_label)].copy()
    m = m.assign(label_short=m["label"].map(display_label))

    imp = pd.read_parquet(paths.S05 / "feature_importance_long.parquet")
    imp = imp.assign(
        label_short=imp["label"].map(display_label),
        abs_perm=imp["perm_importance_mean"].abs(),
    )
    return imp, m


@app.cell
def _(DATAPOINTS_COLORS, parse_feature_name, parse_label_name):
    # Feature-group taxonomy shared with s07: every feature is one of four
    # groups relative to the label being predicted.
    FEATURE_GROUP_COLORS = {
        "Same Assay": DATAPOINTS_COLORS["navy"],
        "Same Assay Diff Condition": DATAPOINTS_COLORS["blue"],
        "Different Assay": DATAPOINTS_COLORS["teal"],
        "In-silico": DATAPOINTS_COLORS["amber"],
    }
    FEATURE_GROUP_ORDER = list(FEATURE_GROUP_COLORS.keys())

    def classify_feature(feature_col: str, label_col: str) -> str:
        _pf = parse_feature_name(feature_col)
        if _pf.kind in ("is", "arm_is"):
            return "In-silico"
        _lbl_vc, _lbl_cond = parse_label_name(label_col)
        if _pf.value_col == _lbl_vc:
            if _pf.condition == _lbl_cond:
                return "Same Assay"
            return "Same Assay Diff Condition"
        return "Different Assay"

    return FEATURE_GROUP_COLORS, FEATURE_GROUP_ORDER, classify_feature


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    display_config,
    m,
    paths,
    plt,
    schema,
    sns,
):
    # ===== figure_5: LOO design + model/config performance sweep =========
    _fig5 = plt.figure(figsize=(9.5, 9.4))
    _gs5 = GridSpec(
        2, 1, height_ratios=[1.0, 1.25], hspace=0.22,
        left=0.085, right=0.78, top=0.95, bottom=0.07, figure=_fig5,
    )

    # ---- Panel A: embedded LOO schematic --------------------------------
    _axA = _fig5.add_subplot(_gs5[0, 0])
    _img = plt.imread(
        paths.REPO_ROOT / "reports" / "manual_input_materials" / "loo_diagram.png"
    )
    _axA.imshow(_img)
    _axA.axis("off")

    # ---- Panel B: boxenplot of rho by model x feature config ------------
    _axC = _fig5.add_subplot(_gs5[1, 0])
    _cfg_order = list(schema.CONFIG_DISPLAY.keys())
    _plotC = m.dropna(subset=["spearman_rho"]).copy()
    _plotC["config_label"] = _plotC["config"].map(display_config)
    _plotC["model_upper"] = _plotC["model"].str.upper()
    _model_order = sorted(_plotC["model_upper"].unique())
    _palette = [
        DATAPOINTS_COLORS["blue"], DATAPOINTS_COLORS["purple"],
        DATAPOINTS_COLORS["pink"], DATAPOINTS_COLORS["red"],
        DATAPOINTS_COLORS["teal"], DATAPOINTS_COLORS["gray"],
        DATAPOINTS_COLORS["amber"], DATAPOINTS_COLORS["navy"],
        DATAPOINTS_COLORS["green"], DATAPOINTS_COLORS["coral"],
        DATAPOINTS_COLORS["slate"],
    ]
    _hue_order = [display_config(c) for c in _cfg_order]
    sns.boxenplot(
        data=_plotC, x="model_upper", y="spearman_rho", hue="config_label",
        order=_model_order, hue_order=_hue_order, palette=_palette,
        linewidth=0.5, ax=_axC,
    )
    _axC.set_xlabel("Model", fontsize=FONT_SIZE_LABEL)
    _axC.set_ylabel("Spearman ρ (LOO)", fontsize=FONT_SIZE_LABEL)
    _axC.axhline(0, color="k", lw=0.5, ls="--")
    _axC.tick_params(labelsize=FONT_SIZE_TICK)
    _axC.legend(title="Feature config", loc="center left",
                bbox_to_anchor=(1.02, 0.5), ncol=1,
                fontsize=FONT_SIZE_LEGEND, title_fontsize=FONT_SIZE_LEGEND,
                frameon=False, handletextpad=0.4,
                labelspacing=0.35, borderpad=0.3)
    _axC.set_title(
        "Spearman ρ by model and feature config\n"
        "(parent-disjoint LOO)",
        fontsize=FONT_SIZE_TITLE,
    )

    # ---- Panel letters --------------------------------------------------
    for _let, _ax in zip("AB", [_axA, _axC]):
        _pos = _ax.get_position()
        _fig5.text(_pos.x0 - 0.045, _pos.y1 + 0.012, _let,
                   fontsize=16, fontweight="bold", va="bottom", ha="left")

    _o5 = paths.FIGURES / "main" / "figure_5_loo.png"
    _o5.parent.mkdir(parents=True, exist_ok=True)
    _fig5.savefig(_o5, dpi=300, bbox_inches="tight")
    print(f"wrote {_o5.relative_to(paths.REPO_ROOT)}")
    _fig5
    return


@app.cell
def _(
    SHORT_LABEL,
    display_label,
    m,
    np,
    parse_label_name,
    paths,
    pd,
    schema,
):
    # Grouped-bar inputs for figure_5_version_2 / version_3. Parental rho
    # from the s03 metrics table (best of mean/min/max). Supervised rho from
    # the LOO parquet, excluding compositional_baseline. Assays sorted by
    # parental-operator Spearman ρ descending.
    _OPS = ("mean", "min", "max")

    _s03 = pd.read_csv(paths.TABLES / "s03_baseline_metrics.csv")
    _s03 = _s03[
        _s03["operator"].isin(_OPS)
        & _s03["value_col"].isin(schema.REPORTED_VALUE_COLS)
    ]
    _parent = (
        _s03.sort_values("spearman_rho", ascending=False)
        .groupby(["value_col", "condition"], as_index=False)
        .head(1)
    )

    _loo = m.dropna(subset=["spearman_rho"]).copy()
    _loo = _loo.assign(
        value_col=_loo["label"].map(lambda c: parse_label_name(c)[0]),
        condition=_loo["label"].map(lambda c: parse_label_name(c)[1]),
    )
    _loo = _loo[
        _loo["value_col"].isin(schema.REPORTED_VALUE_COLS)
        & (_loo["config"] != "compositional_baseline")
    ]
    _sup = (
        _loo.sort_values("spearman_rho", ascending=False)
        .groupby(["value_col", "condition"], as_index=False)
        .head(1)
    )

    _rows = []
    for _, _op_row in _parent.iterrows():
        _vc, _cond, _op = (
            _op_row["value_col"], _op_row["condition"], _op_row["operator"],
        )
        _hit = _sup[(_sup["value_col"] == _vc) & (_sup["condition"] == _cond)]
        if _hit.empty:
            continue
        _hit = _hit.iloc[0]
        _parent_rho = float(_op_row["spearman_rho"])
        _model_rho = float(_hit["spearman_rho"])
        _is = _loo[
            (_loo["value_col"] == _vc)
            & (_loo["condition"] == _cond)
            & (_loo["config"].isin(("in_silico_only", "arm_in_silico_only")))
        ]
        _is_rho = float(_is["spearman_rho"].max()) if len(_is) else np.nan
        _rows.append({
            "value_col": _vc,
            "condition": _cond,
            "label": SHORT_LABEL.get((_vc, _cond), display_label(_hit["label"])),
            "operator": _op,
            "parent_rho": _parent_rho,
            "model_rho": _model_rho,
            "insilico_rho": _is_rho,
            "delta_rho": _model_rho - _parent_rho,
            "config": _hit["config"],
            "model": _hit["model"],
        })

    bars_v2 = (
        pd.DataFrame(_rows)
        .sort_values("parent_rho", ascending=False)
        .reset_index(drop=True)
    )
    bars_v2
    return (bars_v2,)


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    bars_v2,
    np,
    paths,
    plt,
):
    # ===== figure_5_version_2: schematic + grouped bars ==================
    _fig5v2 = plt.figure(figsize=(9.5, 9.6))
    _gs5v2 = GridSpec(
        2, 1, height_ratios=[1.0, 1.25], hspace=0.28,
        left=0.085, right=0.76, top=0.95, bottom=0.16, figure=_fig5v2,
    )

    _axA2 = _fig5v2.add_subplot(_gs5v2[0, 0])
    _img2 = plt.imread(
        paths.REPO_ROOT / "reports" / "manual_input_materials" / "loo_diagram.png"
    )
    _axA2.imshow(_img2)
    _axA2.axis("off")

    _axB2 = _fig5v2.add_subplot(_gs5v2[1, 0])
    _n = len(bars_v2)
    _x = np.arange(_n)
    _w = 0.24
    _c_par = DATAPOINTS_COLORS["navy"]
    _c_mod = DATAPOINTS_COLORS["blue"]
    _c_is = DATAPOINTS_COLORS["pink"]
    _TICK = {
        "AC-SINS PBS pH 7.4": "AC-SINS PBS",
        "AC-SINS His/NaCl pH 6.0": "AC-SINS His/NaCl",
        "AC-SINS His/Arg pH 6.0": "AC-SINS His/Arg",
    }

    _axB2.bar(
        _x - _w, bars_v2["parent_rho"], _w, color=_c_par,
        edgecolor="white", linewidth=0.4, label="Best parental operator", zorder=2,
    )
    _axB2.bar(
        _x, bars_v2["model_rho"], _w, color=_c_mod,
        edgecolor="white", linewidth=0.4,
        label="Best supervised\n(parent-disjoint LOO,\nany featurization)", zorder=2,
    )
    _axB2.bar(
        _x + _w, bars_v2["insilico_rho"], _w, color=_c_is,
        edgecolor="white", linewidth=0.4,
        label="Best supervised\n(parent-disjoint LOO,\nin-silico features only)",
        zorder=2,
    )

    _axB2.axhline(0, color="k", lw=0.5, zorder=1)
    _axB2.set_xticks(_x)
    _axB2.set_xticklabels(
        [_TICK.get(_lab, _lab) for _lab in bars_v2["label"]],
        fontsize=FONT_SIZE_TICK, rotation=35, ha="right",
    )
    _ymax = float(np.nanmax(
        bars_v2[["parent_rho", "model_rho", "insilico_rho"]].to_numpy()
    ))
    _ymin = float(np.nanmin(
        bars_v2[["parent_rho", "model_rho", "insilico_rho"]].to_numpy()
    ))
    _axB2.set_ylim(_ymin - 0.08, _ymax + 0.08)
    _axB2.set_ylabel("Spearman ρ", fontsize=FONT_SIZE_LABEL)
    _axB2.set_xlabel("")
    _axB2.tick_params(labelsize=FONT_SIZE_TICK)
    _axB2.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        fontsize=FONT_SIZE_LEGEND, labelspacing=0.9,
    )
    _axB2.set_title(
        "Parental operator vs supervised LOO",
        fontsize=FONT_SIZE_TITLE,
    )

    for _let, _ax in zip("AB", [_axA2, _axB2]):
        _pos = _ax.get_position()
        _fig5v2.text(
            _pos.x0 - 0.045, _pos.y1 + 0.012, _let,
            fontsize=16, fontweight="bold", va="bottom", ha="left",
        )

    _o5v2 = paths.FIGURES / "main" / "figure_5_version_2.png"
    _o5v2.parent.mkdir(parents=True, exist_ok=True)
    _fig5v2.savefig(_o5v2, dpi=300, bbox_inches="tight")
    print(f"wrote {_o5v2.relative_to(paths.REPO_ROOT)}")
    _fig5v2
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    bars_v2,
    np,
    paths,
    plt,
):
    # ===== figure_5_version_3: Δρ vs best parental operator ==============
    _fig5v3 = plt.figure(figsize=(9.5, 9.6))
    _gs5v3 = GridSpec(
        2, 1, height_ratios=[1.0, 1.25], hspace=0.28,
        left=0.085, right=0.76, top=0.95, bottom=0.16, figure=_fig5v3,
    )

    _axA3 = _fig5v3.add_subplot(_gs5v3[0, 0])
    _img3 = plt.imread(
        paths.REPO_ROOT / "reports" / "manual_input_materials" / "loo_diagram.png"
    )
    _axA3.imshow(_img3)
    _axA3.axis("off")

    _axB3 = _fig5v3.add_subplot(_gs5v3[1, 0])
    _d_mod = bars_v2["model_rho"] - bars_v2["parent_rho"]
    _d_is = bars_v2["insilico_rho"] - bars_v2["parent_rho"]
    _n3 = len(bars_v2)
    _x3 = np.arange(_n3)
    _w3 = 0.32
    _c_mod3 = DATAPOINTS_COLORS["blue"]
    _c_is3 = DATAPOINTS_COLORS["pink"]
    _TICK3 = {
        "AC-SINS PBS pH 7.4": "AC-SINS PBS",
        "AC-SINS His/NaCl pH 6.0": "AC-SINS His/NaCl",
        "AC-SINS His/Arg pH 6.0": "AC-SINS His/Arg",
    }

    _axB3.bar(
        _x3 - 0.5 * _w3, _d_mod, _w3, color=_c_mod3,
        edgecolor="white", linewidth=0.4,
        label="Best supervised\n(parent-disjoint LOO,\nany featurization)", zorder=2,
    )
    _axB3.bar(
        _x3 + 0.5 * _w3, _d_is, _w3, color=_c_is3,
        edgecolor="white", linewidth=0.4,
        label="Best supervised\n(parent-disjoint LOO,\nin-silico features only)",
        zorder=2,
    )
    _axB3.axhline(0, color="k", lw=0.5, zorder=1)
    _axB3.set_xticks(_x3)
    _axB3.set_xticklabels(
        [_TICK3.get(_lab, _lab) for _lab in bars_v2["label"]],
        fontsize=FONT_SIZE_TICK, rotation=35, ha="right",
    )
    _ymax3 = float(np.nanmax(np.concatenate([_d_mod.to_numpy(), _d_is.to_numpy()])))
    _ymin3 = float(np.nanmin(np.concatenate([_d_mod.to_numpy(), _d_is.to_numpy()])))
    _pad3 = 0.08 * max(_ymax3 - _ymin3, 0.2)
    _axB3.set_ylim(_ymin3 - _pad3, _ymax3 + _pad3)
    _axB3.set_ylabel("Δ Spearman ρ  (vs best parental operator)",
                     fontsize=FONT_SIZE_LABEL)
    _axB3.set_xlabel("")
    _axB3.tick_params(labelsize=FONT_SIZE_TICK)
    _axB3.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        fontsize=FONT_SIZE_LEGEND, labelspacing=0.9,
    )
    _axB3.set_title(
        "Supervised LOO minus best parental operator",
        fontsize=FONT_SIZE_TITLE,
    )

    for _let, _ax in zip("AB", [_axA3, _axB3]):
        _pos = _ax.get_position()
        _fig5v3.text(
            _pos.x0 - 0.045, _pos.y1 + 0.012, _let,
            fontsize=16, fontweight="bold", va="bottom", ha="left",
        )

    _o5v3 = paths.FIGURES / "main" / "figure_5_version_3.png"
    _o5v3.parent.mkdir(parents=True, exist_ok=True)
    _fig5v3.savefig(_o5v3, dpi=300, bbox_inches="tight")
    print(f"wrote {_o5v3.relative_to(paths.REPO_ROOT)}")
    _fig5v3
    return


@app.cell
def _(
    DATAPOINTS_COLORS,
    FEATURE_GROUP_COLORS,
    FEATURE_GROUP_ORDER,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    GridSpec,
    classify_feature,
    imp,
    m,
    np,
    paths,
    pd,
    plt,
    schema,
):
    # ===== figure_6: feature-group importance + marginal value ===========
    _SUBSET_LABELS = [
        "HIC RT (norm)",
        "AC-SINS ΔLmax @ His/NaCl pH 6",
        "PR Score @ CHO",
    ]
    _TICK_LABELS = {
        "HIC RT (norm)": "HIC",
        "AC-SINS ΔLmax @ His/NaCl pH 6": "AC-SINS\nHis/NaCl\npH 6",
        "PR Score @ CHO": "PR-CHO",
    }

    _fig6 = plt.figure(figsize=(15.0, 5.6))
    _gs6 = GridSpec(
        1, 2, width_ratios=[1.0, 1.0], wspace=1.55,
        left=0.055, right=0.84, top=0.86, bottom=0.13, figure=_fig6,
    )

    # ---- Panel A: stacked feature-group rho decomposition ---------------
    _axB = _fig6.add_subplot(_gs6[0, 0])
    _NONARM_IS_CONFIGS = (
        "in_silico_only",
        "in_silico_plus_corresponding",
        "in_silico_plus_all_experimental",
    )
    _msB = m[
        m["label_short"].isin(_SUBSET_LABELS)
        & m["config"].isin(_NONARM_IS_CONFIGS)
    ].dropna(subset=["spearman_rho"])
    _bestB = (
        _msB.sort_values("spearman_rho", ascending=False)
        .groupby("label").head(1)
        .sort_values("spearman_rho", ascending=False)
    )

    _rev_order = list(reversed(FEATURE_GROUP_ORDER))
    for _i, (_, _row) in enumerate(_bestB.iterrows()):
        _sl = imp[
            (imp["label"] == _row["label"])
            & (imp["config"] == _row["config"])
            & (imp["model"] == _row["model"])
        ]
        _sl = _sl.assign(
            group=_sl["feature"].map(
                lambda f, lbl=_row["label"]: classify_feature(f, lbl)
            ),
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
            _axB.bar(_i, _h, bottom=_bottom, width=0.62,
                     color=FEATURE_GROUP_COLORS[_g], edgecolor="white",
                     linewidth=0.4)
            if _frac > 0.05:
                _axB.text(_i, _bottom + _h / 2, f"{_frac:.0%}",
                          ha="center", va="center", fontsize=FONT_SIZE_LEGEND - 1,
                          color="white", fontweight="bold")
            _bottom += _h

    _axB.set_xticks(range(len(_bestB)))
    _axB.set_xticklabels(
        [_TICK_LABELS.get(r["label_short"], r["label_short"])
         for _, r in _bestB.iterrows()],
        fontsize=FONT_SIZE_TICK,
    )
    _axB.set_ylabel("Spearman ρ (LOO)", fontsize=FONT_SIZE_LABEL)
    _axB.set_ylim(0, None)
    _axB.tick_params(labelsize=FONT_SIZE_TICK)
    _handlesB = [plt.Rectangle((0, 0), 1, 1, color=FEATURE_GROUP_COLORS[g])
                 for g in FEATURE_GROUP_ORDER]
    _labelsB = [g.replace("Same Assay Diff Condition",
                          "Same Assay\nDiff Condition")
                for g in FEATURE_GROUP_ORDER]
    _axB.legend(_handlesB, _labelsB, loc="center left",
                bbox_to_anchor=(1.02, 0.5), fontsize=FONT_SIZE_LEGEND,
                frameon=False, labelspacing=0.8)
    _axB.set_box_aspect(1.35)
    _axB.set_title(
        "Performance & importance by feature group\n"
        "(best operator config × model, subset labels)",
        fontsize=FONT_SIZE_TITLE,
    )

    # ---- Panel B: supervised vs compositional-baseline dumbbell ---------
    # Restrict to the canonical reported subset (schema.REPORTED_VALUE_COLS):
    # drops BVP, SEC %monomer, and Tonset, which the importance panel includes.
    _axD = _fig6.add_subplot(_gs6[0, 1])
    _reported_vc = set(schema.REPORTED_VALUE_COLS)
    _md = m.dropna(subset=["spearman_rho"])
    _md = _md[_md["label"].str.split("__").str[1].isin(_reported_vc)]
    _base = (
        _md[_md["config"] == "compositional_baseline"]
        .sort_values("spearman_rho", ascending=False)
        .groupby("label").head(1).set_index("label")
    )
    _sup = (
        _md[_md["config"] != "compositional_baseline"]
        .sort_values("spearman_rho", ascending=False)
        .groupby("label").head(1).set_index("label")
    )
    _labels = [_l for _l in _sup.index if _l in _base.index]
    _dfD = (
        pd.DataFrame({
            "label_short": _base.loc[_labels, "label_short"],
            "baseline": _base.loc[_labels, "spearman_rho"],
            "supervised": _sup.loc[_labels, "spearman_rho"],
        })
        .assign(gain=lambda d: d["supervised"] - d["baseline"])
        .sort_values("supervised", ascending=True)
        .reset_index(drop=True)
    )

    _nD = len(_dfD)
    _yD = np.arange(_nD)
    _c_base = DATAPOINTS_COLORS["gray"]
    _c_sup = DATAPOINTS_COLORS["blue"]
    for _i, _row in _dfD.iterrows():
        _axD.plot([_row["baseline"], _row["supervised"]], [_i, _i],
                  color=DATAPOINTS_COLORS["gray"], lw=1.2, alpha=0.5, zorder=1)
    _axD.scatter(_dfD["baseline"], _yD, s=46, color=_c_base, zorder=2,
                 edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                 label="Compositional baseline\n(mean of parents)")
    _axD.scatter(_dfD["supervised"], _yD, s=46, color=_c_sup, zorder=3,
                 edgecolors=DATAPOINTS_COLORS["gray"], linewidths=0.3,
                 label="Best supervised model\n(parent-disjoint LOO)")

    _xmax = float(np.nanmax([_dfD["baseline"].max(), _dfD["supervised"].max()]))
    _xmin = min(0.0, float(_dfD[["baseline", "supervised"]].min().min()))
    for _i, _row in _dfD.iterrows():
        _gain = _row["gain"]
        _x = max(_row["baseline"], _row["supervised"])
        _axD.text(_x + 0.035, _i, f"{_gain:+.2f}",
                  va="center_baseline", ha="left",
                  fontsize=FONT_SIZE_LEGEND - 1, fontweight="bold",
                  color=_c_sup if _gain >= 0 else DATAPOINTS_COLORS["red"])

    _axD.axvline(0, color="k", lw=0.5)
    _axD.set_yticks(_yD)
    _axD.set_yticklabels(_dfD["label_short"], fontsize=FONT_SIZE_TICK)
    _axD.set_ylim(-0.6, _nD - 0.4)
    _axD.set_xlim(_xmin - 0.05, _xmax + 0.20)
    _axD.set_xlabel("Spearman ρ  (out-of-fold, parent-disjoint LOO)",
                    fontsize=FONT_SIZE_LABEL)
    _axD.tick_params(labelsize=FONT_SIZE_TICK)
    _axD.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=False, fontsize=FONT_SIZE_LEGEND, labelspacing=0.8)
    _axD.set_box_aspect(1.35)
    _median_gain = float(_dfD["gain"].median())
    _axD.set_title(
        "Supervised vs. compositional baseline\n"
        f"(median ρ gain across {_nD} labels = {_median_gain:+.2f})",
        fontsize=FONT_SIZE_TITLE,
    )

    # ---- Panel letters --------------------------------------------------
    for _let, _ax in zip("AB", [_axB, _axD]):
        _pos = _ax.get_position()
        _fig6.text(_pos.x0 - 0.05, _pos.y1 + 0.10, _let,
                   fontsize=16, fontweight="bold", va="bottom", ha="left")

    _o6 = paths.FIGURES / "main" / "figure_6_loo.png"
    _o6.parent.mkdir(parents=True, exist_ok=True)
    _fig6.savefig(_o6, dpi=300, bbox_inches="tight")
    print(f"wrote {_o6.relative_to(paths.REPO_ROOT)}")
    _fig6
    return


if __name__ == "__main__":
    app.run()
