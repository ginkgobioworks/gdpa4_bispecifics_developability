"""Generate supplementary Figure S09.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s09.py``
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _source in (_ROOT / 'src', _ROOT / 'datapoints_figures' / 'src'):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
from prophet_ab.source_data import emit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sp_stats
from prophet_ab import paths, schema
from prophet_ab import normalize as nz
from prophet_ab.features.naming import display_value_col
from datapoints_figures import DATAPOINTS_COLORS, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE, FULL_WIDTH, SINGLE_COL_WIDTH, set_manuscript_style, wrap_label, wrap_title, equalize_axes, grid_figsize
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D as _Line2D
from shutil import copy2

def main():
    set_manuscript_style()
    MIN_DEGREE = 3
    EXCLUDED = schema.DEPRECATED_VALUE_COLS | schema.PRODUCTION_VALUE_COLS
    _summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    _components = pd.read_parquet(paths.S02 / 'bispecific_components.parquet')
    _monospecific = _summaries[_summaries['kind'] == schema.KIND_MONOSPECIFIC][['antibody_name', 'value_col', 'condition', 'median']].copy()
    _monospecific['parent'] = _monospecific['antibody_name'].map(nz.strip_isotype_suffix)
    _bispecific = _summaries[_summaries['kind'] == schema.KIND_BISPECIFIC][['antibody_name', 'value_col', 'condition', 'median']].rename(columns={'median': 'bispecific_median'})
    _bispecific = _bispecific[~_bispecific['value_col'].isin(EXCLUDED)]
    _bispecificp = _bispecific.merge(_components[['antibody_name', 'parent_a', 'parent_b']], on='antibody_name')
    _bispecificp = _bispecificp.merge(_monospecific[['parent', 'value_col', 'condition', 'median']].rename(columns={'parent': 'parent_a', 'median': 'pa_median'}), on=['parent_a', 'value_col', 'condition'], how='left').merge(_monospecific[['parent', 'value_col', 'condition', 'median']].rename(columns={'parent': 'parent_b', 'median': 'pb_median'}), on=['parent_b', 'value_col', 'condition'], how='left')
    _bispecificp['parent_mean'] = (_bispecificp['pa_median'] + _bispecificp['pb_median']) / 2
    _bispecificp['residual'] = _bispecificp['bispecific_median'] - _bispecificp['parent_mean']
    residuals_long = _bispecificp.dropna(subset=['residual'])
    components = _components
    G_full = nx.Graph()
    for _, _row in components.iterrows():
        G_full.add_edge(_row['parent_a'], _row['parent_b'], bispecific=_row['antibody_name'])
    _pos = nx.spring_layout(G_full, seed=42, k=1.5)
    _degrees = dict(G_full.degree())
    _sizes = [_degrees[n] * 40 + 20 for n in G_full.nodes()]
    _colors = ['tab:blue' if _degrees[n] >= MIN_DEGREE else 'lightgray' for n in G_full.nodes()]
    fig_graph, _ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH), layout='constrained')
    nx.draw_networkx_edges(G_full, _pos, ax=_ax, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(G_full, _pos, ax=_ax, node_size=_sizes, node_color=_colors, edgecolors='k', linewidths=0.3)
    nx.draw_networkx_labels(G_full, _pos, labels={n: n for n in G_full.nodes() if _degrees[n] >= MIN_DEGREE}, ax=_ax, font_size=5)
    _ax.set_title(f'Bispecific parent graph  |  {G_full.number_of_nodes()} Fvs, {G_full.number_of_edges()} bispecific edges')
    _legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markeredgecolor='k', markersize=8, label=f'degree >= {MIN_DEGREE}'), Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markeredgecolor='k', markersize=8, label=f'degree < {MIN_DEGREE}')]
    _ax.legend(handles=_legend_handles, loc='upper right', fontsize=FONT_SIZE_LEGEND)
    _ax.axis('off')
    _o = paths.FIGURES / 's08_parent_graph.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_graph.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_graph
    _degrees = [d for _, d in G_full.degree()]
    fig_deg, _ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.75), layout='constrained')
    _bins = np.arange(0, max(_degrees) + 2) - 0.5
    _ax.hist(_degrees, bins=_bins, edgecolor='k', linewidth=0.5)
    _ax.axvline(MIN_DEGREE - 0.5, color='red', linestyle='--', linewidth=1.2, label=f'MIN_DEGREE = {MIN_DEGREE}')
    _n_pass = sum((1 for d in _degrees if d >= MIN_DEGREE))
    _ax.set_xlabel('Degree (number of bispecific partners)')
    _ax.set_ylabel('Count (Fvs)')
    _ax.set_title(f'Fv degree distribution\n{_n_pass}/{len(_degrees)} pass threshold')
    _ax.legend()
    _o = paths.FIGURES / 's08_degree_distribution.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_deg.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_deg
    fvs_passing = sorted([n for n, d in G_full.degree() if d >= MIN_DEGREE])
    fvs_set = set(fvs_passing)
    _src = residuals_long[residuals_long['parent_a'].isin(fvs_set) & residuals_long['parent_b'].isin(fvs_set)]
    _panels = _src.groupby(['value_col', 'condition']).size().reset_index(name='n').query('n >= 5')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = min(3, len(_panels))
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols)
    fig_strips, _axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout='constrained')
    _axes_flat = _axes.flatten()
    _rng = np.random.default_rng(0)
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _src[(_src['value_col'] == _vc) & (_src['condition'] == _cond)]
        _a = _g[['parent_a', 'residual']].rename(columns={'parent_a': 'fv'})
        _b = _g[['parent_b', 'residual']].rename(columns={'parent_b': 'fv'})
        _fr = pd.concat([_a, _b], ignore_index=True)
        _order = _fr.groupby('fv')['residual'].median().sort_values().index
        _positions = {fv: i for i, fv in enumerate(_order)}
        _x = _fr['fv'].map(_positions) + _rng.uniform(-0.2, 0.2, len(_fr))
        _ax.scatter(_x, _fr['residual'], s=8, alpha=0.5, edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
        _ax.axhline(0, color='k', lw=0.5, ls='--')
        _ax.set_title(wrap_title(display_value_col(_vc, _cond), _cell_w))
        _ax.set_xticks([])
        _ax.set_ylabel('residual')
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's08_residual_strips.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_strips.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_strips
    _fv_to_idx = {fv: i for i, fv in enumerate(fvs_passing)}
    _n_fv = len(fvs_passing)
    _assays = residuals_long.groupby(['value_col', 'condition']).size().reset_index(name='n')
    _all_rows = []
    for _, _arow in _assays.iterrows():
        _vc, _cond = (_arow['value_col'], _arow['condition'])
        _g = residuals_long[(residuals_long['value_col'] == _vc) & (residuals_long['condition'] == _cond) & residuals_long['parent_a'].isin(fvs_set) & residuals_long['parent_b'].isin(fvs_set)].reset_index(drop=True)
        _n = len(_g)
        if _n < 3:
            continue
        _y = _g['residual'].values
        _X = np.zeros((_n, _n_fv))
        _X[np.arange(_n), _g['parent_a'].map(_fv_to_idx).values] = 1.0
        _X[np.arange(_n), _g['parent_b'].map(_fv_to_idx).values] = 1.0
        _used = _X.sum(axis=0) > 0
        _X_sub = _X[:, _used]
        _fvs_sub = [fv for fv, u in zip(fvs_passing, _used) if u]
        _p = _X_sub.shape[1]
        if _n <= _p or _p == 0:
            continue
        _beta, _, _rank, _ = np.linalg.lstsq(_X_sub, _y, rcond=None)
        _y_hat = _X_sub @ _beta
        _rss = float(np.sum((_y - _y_hat) ** 2))
        _dof = _n - int(_rank)
        if _dof <= 0:
            continue
        _sigma2 = _rss / _dof
        try:
            _xtx_inv = np.linalg.inv(_X_sub.T @ _X_sub)
        except np.linalg.LinAlgError:
            _xtx_inv = np.linalg.pinv(_X_sub.T @ _X_sub)
        _se = np.sqrt(np.maximum(np.diag(_sigma2 * _xtx_inv), 0.0))
        _t_stat = np.where(_se > 0, _beta / _se, 0.0)
        _p_val = 2 * (1 - sp_stats.t.cdf(np.abs(_t_stat), _dof))
        _p_adj = sp_stats.false_discovery_control(_p_val, method='bh')
        _tss = float(np.sum((_y - _y.mean()) ** 2))
        _r2 = 1 - _rss / _tss if _tss > 0 else np.nan
        _f_stat = (_tss - _rss) / _p / (_rss / _dof) if _dof > 0 else np.nan
        _f_pval = 1 - sp_stats.f.cdf(_f_stat, _p, _dof) if np.isfinite(_f_stat) else np.nan
        for _j, _fv in enumerate(_fvs_sub):
            _all_rows.append({'value_col': _vc, 'condition': _cond, 'fv': _fv, 'degree': int(_X_sub[:, _j].sum()), 'coef': float(_beta[_j]), 'se': float(_se[_j]), 't_stat': float(_t_stat[_j]), 'p_value': float(_p_val[_j]), 'p_adj': float(_p_adj[_j]), 'significant': bool(_p_adj[_j] < 0.05), 'n_obs': _n, 'n_fv': _p, 'model_r2': _r2, 'model_f_stat': float(_f_stat) if np.isfinite(_f_stat) else np.nan, 'model_f_pval': float(_f_pval) if np.isfinite(_f_pval) else np.nan})
    coefficients = pd.DataFrame(_all_rows)
    'Intercept analysis: decompose no-intercept coefficients into global\nmean (mu) + per-Fv deviations (delta_k), and test whether mu != 0.\n\nThe no-intercept model r_ij = beta_i + beta_j is identified within each\nconnected component because there is no intercept to absorb a global\nshift.  Each beta_k is an absolute format effect: beta_k = 0 means\nFv k contributes no systematic format shift.\n\nTo answer "is there a global format bias?", we decompose:\n    beta_k = mu + delta_k,  where mu = (1/p) sum(beta_k)\nand test H0: mu = 0 using the delta method on the OLS covariance.\nThe degree-weighted mean (weighting by observation count per Fv) is\na natural alternative that down-weights noisy low-degree Fvs.\n'
    _fv_to_idx = {fv: i for i, fv in enumerate(fvs_passing)}
    _n_fv = len(fvs_passing)
    _assays = residuals_long.groupby(['value_col', 'condition']).size().reset_index(name='n')
    _icpt_rows = []
    for _, _arow in _assays.iterrows():
        _vc, _cond = (_arow['value_col'], _arow['condition'])
        _g = residuals_long[(residuals_long['value_col'] == _vc) & (residuals_long['condition'] == _cond) & residuals_long['parent_a'].isin(fvs_set) & residuals_long['parent_b'].isin(fvs_set)].reset_index(drop=True)
        _n = len(_g)
        if _n < 3:
            continue
        _y = _g['residual'].values
        _X = np.zeros((_n, _n_fv))
        _X[np.arange(_n), _g['parent_a'].map(_fv_to_idx).values] = 1.0
        _X[np.arange(_n), _g['parent_b'].map(_fv_to_idx).values] = 1.0
        _used = _X.sum(axis=0) > 0
        _X_sub = _X[:, _used]
        _fvs_sub = [fv for fv, u in zip(fvs_passing, _used) if u]
        _p = _X_sub.shape[1]
        if _n <= _p or _p == 0:
            continue
        _beta, _, _rank, _ = np.linalg.lstsq(_X_sub, _y, rcond=None)
        _y_hat = _X_sub @ _beta
        _rss = float(np.sum((_y - _y_hat) ** 2))
        _dof = _n - int(_rank)
        if _dof <= 0:
            continue
        _sigma2 = _rss / _dof
        try:
            _xtx_inv = np.linalg.inv(_X_sub.T @ _X_sub)
        except np.linalg.LinAlgError:
            _xtx_inv = np.linalg.pinv(_X_sub.T @ _X_sub)
        _cov_beta = _sigma2 * _xtx_inv
        _w_unif = np.ones(_p) / _p
        _mu = float(_w_unif @ _beta)
        _mu_var = float(_w_unif @ _cov_beta @ _w_unif)
        _mu_se = float(np.sqrt(max(_mu_var, 0.0)))
        _mu_t = _mu / _mu_se if _mu_se > 0 else 0.0
        _mu_p = float(2 * (1 - sp_stats.t.cdf(abs(_mu_t), _dof)))
        _col_degrees = _X_sub.sum(axis=0).astype(float)
        _w_deg = _col_degrees / _col_degrees.sum()
        _dw_mu = float(_w_deg @ _beta)
        _dw_mu_var = float(_w_deg @ _cov_beta @ _w_deg)
        _dw_mu_se = float(np.sqrt(max(_dw_mu_var, 0.0)))
        _dw_mu_t = _dw_mu / _dw_mu_se if _dw_mu_se > 0 else 0.0
        _dw_mu_p = float(2 * (1 - sp_stats.t.cdf(abs(_dw_mu_t), _dof)))
        _deltas = _beta - _mu
        _max_abs_delta = float(np.max(np.abs(_deltas)))
        _icpt_rows.append({'value_col': _vc, 'condition': _cond, 'intercept': _mu, 'intercept_se': _mu_se, 'intercept_pvalue': _mu_p, 'mean_coef_no_intercept': _mu, 'degweighted_mean_coef_no_intercept': _dw_mu, 'degweighted_mean_se': _dw_mu_se, 'degweighted_mean_pvalue': _dw_mu_p, 'coef_max_abs_diff': _max_abs_delta})
    intercept_comparison = pd.DataFrame(_icpt_rows)
    _o = paths.TABLES / 's08_intercept_comparison.csv'
    _o.parent.mkdir(parents=True, exist_ok=True)
    intercept_comparison.to_csv(_o, index=False)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}  rows={len(intercept_comparison)}')
    print()
    print('Intercept analysis summary:')
    print(intercept_comparison.to_string(index=False))
    _sig_unif = intercept_comparison[intercept_comparison['intercept_pvalue'] < 0.05]
    _sig_dw = intercept_comparison[intercept_comparison['degweighted_mean_pvalue'] < 0.05]
    _n_total = len(intercept_comparison)
    _n_sig_unif = len(_sig_unif)
    _n_sig_dw = len(_sig_dw)
    _max_delta = intercept_comparison['coef_max_abs_diff'].max()
    _med_dw = intercept_comparison['degweighted_mean_coef_no_intercept'].median()
    _panels = coefficients.groupby(['value_col', 'condition']).size().reset_index(name='n')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = min(3, len(_panels))
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols, cell_aspect=1.4)
    fig_forest, _axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout='constrained')
    _axes_flat = _axes.flatten()
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = coefficients[(coefficients['value_col'] == _vc) & (coefficients['condition'] == _cond)].sort_values('coef')
        _y = np.arange(len(_g))
        _sig = _g['significant'].values
        _lo = _g['coef'].values - 1.96 * _g['se'].values
        _hi = _g['coef'].values + 1.96 * _g['se'].values
        _ax.hlines(_y[~_sig], _lo[~_sig], _hi[~_sig], colors='tab:gray', linewidth=0.8)
        _ax.scatter(_g['coef'].values[~_sig], _y[~_sig], s=10, color='tab:gray', zorder=5)
        _ax.hlines(_y[_sig], _lo[_sig], _hi[_sig], colors='tab:red', linewidth=0.8)
        _ax.scatter(_g['coef'].values[_sig], _y[_sig], s=10, color='tab:red', zorder=5)
        _ax.axvline(0, color='k', lw=0.5, ls='--')
        _ax.set_yticks(_y)
        _ax.set_yticklabels(_g['fv'].values, fontsize=4)
        _r2 = _g['model_r2'].iloc[0]
        _ax.set_title(wrap_title(f'{display_value_col(_vc, _cond)}\nR²={_r2:.2f}', _cell_w))
        _ax.set_xlabel('bsAb effect')
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's08_coefficient_forest.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_forest.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_forest
    _coef = coefficients.copy()
    _coef['assay_label'] = _coef.apply(lambda r: display_value_col(r['value_col'], r['condition']), axis=1)
    _pivot = _coef.pivot_table(index='fv', columns='assay_label', values='coef', aggfunc='first')
    _sig_pivot = _coef.pivot_table(index='fv', columns='assay_label', values='significant', aggfunc='first')
    _order = _pivot.abs().mean(axis=1).sort_values(ascending=False).index
    _pivot = _pivot.loc[_order]
    _sig_pivot = _sig_pivot.reindex(_order)
    _sig_arr = _sig_pivot.fillna(False).values.astype(bool)
    _vmax = float(np.nanmax(np.abs(_pivot.values)))
    if _vmax == 0:
        _vmax = 1.0
    _cmap = plt.get_cmap('RdBu_r').copy()
    _cmap.set_bad('lightgray')
    _fig_w = FULL_WIDTH
    _fig_h = max(6, len(_pivot) * 0.25 + 2)
    fig_heatmap, _ax = plt.subplots(figsize=(_fig_w, _fig_h), layout='constrained')
    _norm = TwoSlopeNorm(vmin=-_vmax, vcenter=0, vmax=_vmax)
    _im = _ax.imshow(_pivot.values, aspect='auto', cmap=_cmap, norm=_norm, interpolation='nearest')
    plt.colorbar(_im, ax=_ax, label='bs effect', shrink=0.8)
    for _i in range(len(_pivot)):
        for _j in range(len(_pivot.columns)):
            if _sig_arr[_i, _j]:
                _ax.text(_j, _i, '*', ha='center', va='center', fontsize=6, color='k', fontweight='bold')
    _ax.set_xticks(range(len(_pivot.columns)))
    _ax.set_xticklabels(_pivot.columns, rotation=45, ha='right', fontsize=6)
    _ax.set_yticks(range(len(_pivot)))
    _ax.set_yticklabels(_pivot.index, fontsize=5)
    _ax.set_title(wrap_title('Per-Fv bispecific format effects (* = FDR < 0.05)', _fig_w))
    _o = paths.FIGURES / 's08_coefficient_heatmap.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_heatmap.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_heatmap
    _o = paths.TABLES / 's08_format_effect_coefficients.csv'
    _o.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(_o, index=False)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}  rows={len(coefficients)}')
    _src = residuals_long[residuals_long['parent_a'].isin(fvs_set) & residuals_long['parent_b'].isin(fvs_set) & residuals_long['value_col'].isin(schema.REPORTED_VALUE_COLS)]
    _panels = _src.groupby(['value_col', 'condition']).size().reset_index(name='n').query('n >= 5')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = min(3, len(_panels))
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols)
    fig_strips_reported, _axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout='constrained')
    _axes_flat = _axes.flatten() if hasattr(_axes, 'flatten') else [_axes]
    _rng = np.random.default_rng(0)
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _src[(_src['value_col'] == _vc) & (_src['condition'] == _cond)]
        _a = _g[['parent_a', 'residual']].rename(columns={'parent_a': 'fv'})
        _b = _g[['parent_b', 'residual']].rename(columns={'parent_b': 'fv'})
        _fr = pd.concat([_a, _b], ignore_index=True)
        _order = _fr.groupby('fv')['residual'].median().sort_values().index
        _positions = {fv: i for i, fv in enumerate(_order)}
        _x = _fr['fv'].map(_positions) + _rng.uniform(-0.2, 0.2, len(_fr))
        _ax.scatter(_x, _fr['residual'], s=8, alpha=0.5, edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
        _ax.axhline(0, color='k', lw=0.5, ls='--')
        _ax.set_title(wrap_title(display_value_col(_vc, _cond), _cell_w))
        _ax.set_xticks([])
        _ax.set_ylabel('residual')
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's08_residual_strips_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_strips_reported.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_strips_reported
    _coefs_rep = coefficients[coefficients['value_col'].isin(schema.REPORTED_VALUE_COLS)]
    _panels = _coefs_rep.groupby(['value_col', 'condition']).size().reset_index(name='n')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = min(3, len(_panels))
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols, cell_aspect=1.4)
    fig_forest_reported, _axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout='constrained')
    _axes_flat = _axes.flatten() if hasattr(_axes, 'flatten') else [_axes]
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _coefs_rep[(_coefs_rep['value_col'] == _vc) & (_coefs_rep['condition'] == _cond)].sort_values('coef')
        _y = np.arange(len(_g))
        _sig = _g['significant'].values
        _lo = _g['coef'].values - 1.96 * _g['se'].values
        _hi = _g['coef'].values + 1.96 * _g['se'].values
        _ax.hlines(_y[~_sig], _lo[~_sig], _hi[~_sig], colors='tab:gray', linewidth=0.8)
        _ax.scatter(_g['coef'].values[~_sig], _y[~_sig], s=10, color='tab:gray', zorder=5)
        _ax.hlines(_y[_sig], _lo[_sig], _hi[_sig], colors='tab:red', linewidth=0.8)
        _ax.scatter(_g['coef'].values[_sig], _y[_sig], s=10, color='tab:red', zorder=5)
        _ax.axvline(0, color='k', lw=0.5, ls='--')
        _ax.set_yticks(_y)
        _ax.set_yticklabels(_g['fv'].values, fontsize=4)
        _r2 = _g['model_r2'].iloc[0]
        _ax.set_title(wrap_title(f'{display_value_col(_vc, _cond)}\nR²={_r2:.2f}', _cell_w))
        _ax.set_xlabel('bsAb effect')
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's08_coefficient_forest_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_forest_reported.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_forest_reported
    _coefs = coefficients[coefficients['value_col'].isin(schema.REPORTED_VALUE_COLS)].copy()
    _STYLE = [('acsins_delta_Lmax', '1X PBS', DATAPOINTS_COLORS['blue'], 'o'), ('acsins_delta_Lmax', 'His/Arg, pH 6', DATAPOINTS_COLORS['blue'], 's'), ('acsins_delta_Lmax', 'His/NaCl, pH 6', DATAPOINTS_COLORS['blue'], 'D'), ('hihplc_normretentiontime', 'default', DATAPOINTS_COLORS['teal'], 'o'), ('smachplc_retentiontime', 'default', DATAPOINTS_COLORS['teal'], 's'), ('hachplc_retentiontime', 'default', DATAPOINTS_COLORS['teal'], 'D'), ('pr_score', 'CHO', DATAPOINTS_COLORS['coral'], 'o'), ('pr_score', 'Ovalbumin', DATAPOINTS_COLORS['coral'], 's'), ('bvp_score_norm', 'default', DATAPOINTS_COLORS['coral'], 'D'), ('thermostability_tm1', 'Tm1', DATAPOINTS_COLORS['amber'], 'o'), ('thermostability_tm2', 'Tm2', DATAPOINTS_COLORS['amber'], 's')]
    _z_parts = []
    for _vc, _cond, _, _ in _STYLE:
        _g = _coefs[(_coefs['value_col'] == _vc) & (_coefs['condition'] == _cond)]
        if _g.empty:
            continue
        _sd = _g['coef'].std()
        if _sd == 0 or np.isnan(_sd):
            continue
        _gc = _g.copy()
        _gc['z_coef'] = (_gc['coef'] - _gc['coef'].mean()) / _sd
        _gc['z_se'] = _gc['se'] / _sd
        _z_parts.append(_gc)
    _z = pd.concat(_z_parts, ignore_index=True)
    _sig_counts = _z.groupby('fv')['significant'].sum().reindex(_z['fv'].unique(), fill_value=0).sort_values()
    _fv_order = _sig_counts.index.tolist()
    _fv_pos = {fv: i for i, fv in enumerate(_fv_order)}
    _n_fv = len(_fv_order)
    _CLASS_IDX = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    _n_classes = 4
    _class_offsets = np.linspace(-0.25, 0.25, _n_classes)
    _fig_h = max(3, _n_fv * 0.13 + 1.0)
    fig_overlay, _ax = plt.subplots(figsize=(FULL_WIDTH, _fig_h), layout='constrained')
    for _row in range(_n_fv):
        if _row % 2 == 0:
            _ax.axhspan(_row - 0.5, _row + 0.5, color='#f0f0f0', zorder=0)
    _gray = DATAPOINTS_COLORS['gray']
    _legend_handles = []
    for _i, (_vc, _cond, _color, _marker) in enumerate(_STYLE):
        _g = _z[(_z['value_col'] == _vc) & (_z['condition'] == _cond)]
        if _g.empty:
            continue
        _ypos = _g['fv'].map(_fv_pos).values + _class_offsets[_CLASS_IDX[_i]]
        _xerr = 1.96 * _g['z_se'].values
        _LABEL_OVERRIDE = {('pr_score', 'CHO'): 'PR-CHO Score', ('pr_score', 'Ovalbumin'): 'PR-Ovalbumin Score', ('bvp_score_norm', 'default'): 'PR-BVP Score'}
        _label = _LABEL_OVERRIDE.get((_vc, _cond), display_value_col(_vc, _cond).replace(' @ ', ' '))
        _sig = _g['significant'].values
        _ax.errorbar(_g['z_coef'].values[~_sig], _ypos[~_sig], xerr=_xerr[~_sig], fmt='none', ecolor=_gray, elinewidth=0.5, alpha=0.25, capsize=1.5, capthick=0.4, zorder=3)
        _ax.errorbar(_g['z_coef'].values[_sig], _ypos[_sig], xerr=_xerr[_sig], fmt='none', ecolor=_color, elinewidth=0.5, alpha=0.35, capsize=1.5, capthick=0.4, zorder=3)
        _ax.scatter(_g['z_coef'].values[~_sig], _ypos[~_sig], s=12, color=_gray, marker=_marker, edgecolors='white', linewidths=0.3, zorder=4, alpha=0.5)
        _ax.scatter(_g['z_coef'].values[_sig], _ypos[_sig], s=12, color=_color, marker=_marker, edgecolors='white', linewidths=0.3, zorder=5, alpha=0.85)
        _legend_handles.append(_Line2D([0], [0], marker=_marker, color='w', markerfacecolor=_color, markeredgecolor='white', markersize=5, label=_label))
    _ax.axvline(0, color='k', lw=0.5, ls='--')
    _ax.set_yticks(range(_n_fv))
    _ax.set_yticklabels(_fv_order, fontsize=5)
    _ax.set_xlabel('Fv effect (z-score normalized within assay)')
    _ax.legend(handles=_legend_handles, bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, borderaxespad=0)
    _o = paths.FIGURES / 's08_coefficient_overlay_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_overlay.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_overlay
    _coefs_rep = coefficients[coefficients['value_col'].isin(schema.REPORTED_VALUE_COLS)]
    _coefs_rep = _coefs_rep.copy()
    _coefs_rep['assay_label'] = _coefs_rep.apply(lambda r: display_value_col(r['value_col'], r['condition']), axis=1)
    _pivot = _coefs_rep.pivot_table(index='fv', columns='assay_label', values='coef', aggfunc='first')
    _sig_pivot = _coefs_rep.pivot_table(index='fv', columns='assay_label', values='significant', aggfunc='first')
    _order = _pivot.abs().mean(axis=1).sort_values(ascending=False).index
    _pivot = _pivot.loc[_order]
    _sig_pivot = _sig_pivot.reindex(_order)
    _sig_arr = _sig_pivot.fillna(False).values.astype(bool)
    _vmax = float(np.nanmax(np.abs(_pivot.values)))
    if _vmax == 0:
        _vmax = 1.0
    _cmap = plt.get_cmap('RdBu_r').copy()
    _cmap.set_bad('lightgray')
    _fig_w = FULL_WIDTH
    _fig_h = max(4, len(_pivot) * 0.22 + 2)
    fig_heatmap_reported, _ax = plt.subplots(figsize=(_fig_w, _fig_h), layout='constrained')
    _norm = TwoSlopeNorm(vmin=-_vmax, vcenter=0, vmax=_vmax)
    _im = _ax.imshow(_pivot.values, aspect='auto', cmap=_cmap, norm=_norm, interpolation='nearest')
    plt.colorbar(_im, ax=_ax, label='bs effect', shrink=0.8)
    for _i in range(len(_pivot)):
        for _j in range(len(_pivot.columns)):
            if _sig_arr[_i, _j]:
                _ax.text(_j, _i, '*', ha='center', va='center', fontsize=6, color='k', fontweight='bold')
    _ax.set_xticks(range(len(_pivot.columns)))
    _ax.set_xticklabels(_pivot.columns, rotation=45, ha='right', fontsize=6)
    _ax.set_yticks(range(len(_pivot)))
    _ax.set_yticklabels(_pivot.index, fontsize=5)
    _ax.set_title(wrap_title('Per-Fv bispecific format effects (* = FDR < 0.05)', _fig_w))
    _o = paths.FIGURES / 's08_coefficient_heatmap_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_heatmap_reported.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_heatmap_reported
    _src = paths.FIGURES / 's08_coefficient_overlay_reported_subset.png'
    _dst = paths.FIGURES / 'supplementary/figure_s09.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('supp_09')
if __name__ == '__main__':
    main()
