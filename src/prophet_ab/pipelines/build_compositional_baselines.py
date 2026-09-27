import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from prophet_ab import paths, schema
from prophet_ab.features import compositional
from prophet_ab.features.naming import display_value_col
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, equalize_axes, grid_figsize, wrap_title, FULL_WIDTH, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE, wrap_label
from prophet_ab import normalize as nz
from matplotlib.lines import Line2D as _Line2D

def main():
    set_manuscript_style()
    DEPRECATED = schema.DEPRECATED_VALUE_COLS
    summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    components = pd.read_parquet(paths.S02 / 'bispecific_components.parquet')
    monospecific = summaries[summaries['kind'] == schema.KIND_MONOSPECIFIC][['antibody_name', 'value_col', 'condition', 'median']]
    monospecific = monospecific.assign(parent=monospecific['antibody_name'].map(nz.strip_isotype_suffix))
    monospecific_lookup = monospecific[['parent', 'value_col', 'condition', 'median']]
    bispecific = summaries[summaries['kind'] == schema.KIND_BISPECIFIC][['antibody_name', 'value_col', 'condition', 'median']].rename(columns={'median': 'bispecific_median'})
    n3p = bispecific.merge(components[['antibody_name', 'parent_a', 'parent_b']], on='antibody_name')
    n3p = n3p.merge(monospecific_lookup.rename(columns={'parent': 'parent_a', 'median': 'parent_a_median'}), on=['parent_a', 'value_col', 'condition'], how='left').merge(monospecific_lookup.rename(columns={'parent': 'parent_b', 'median': 'parent_b_median'}), on=['parent_b', 'value_col', 'condition'], how='left')
    pred_df = compositional.apply_all(n3p['parent_a_median'], n3p['parent_b_median'])
    out = pd.concat([n3p, pred_df.add_prefix('pred_')], axis=1)
    out
    _o = paths.S04 / 'bispecific_compositional_predictions.parquet'
    _o.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_o, index=False)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}  shape={out.shape}')
    _rows = []
    _src = out[~out['value_col'].isin(DEPRECATED)]
    for (_vc, _cond), _g in _src.groupby(['value_col', 'condition'], sort=False):
        for _op in compositional.OPERATORS:
            _pred = _g[f'pred_{_op}']
            _ok = _pred.notna() & _g['bispecific_median'].notna()
            _n = int(_ok.sum())
            if _n < 3:
                _rows.append(dict(value_col=_vc, condition=_cond, operator=_op, n=_n, spearman_rho=np.nan, r2=np.nan, mae=np.nan))
                continue
            _rho, _ = stats.spearmanr(_g.loc[_ok, 'bispecific_median'], _pred[_ok])
            _r = np.corrcoef(_g.loc[_ok, 'bispecific_median'], _pred[_ok])[0, 1]
            _mae = float((_g.loc[_ok, 'bispecific_median'] - _pred[_ok]).abs().mean())
            _rows.append(dict(value_col=_vc, condition=_cond, operator=_op, n=_n, spearman_rho=_rho, r2=_r ** 2, mae=_mae))
    metrics = pd.DataFrame(_rows)
    metrics
    _o = paths.TABLES / 's03_baseline_metrics.csv'
    _o.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(_o, index=False)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}  rows={len(metrics)}')
    _src = out[~out['value_col'].isin(DEPRECATED)]
    _panels = _src.dropna(subset=['pred_mean', 'bispecific_median']).groupby(['value_col', 'condition']).size().reset_index(name='n').query('n >= 3')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = 4
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fig_w, _fig_h, _cell_w = grid_figsize(_nrows, _ncols)
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(_fig_w, _fig_h), layout='constrained')
    _axes_flat = axes.flatten()
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _src[(_src['value_col'] == _vc) & (_src['condition'] == _cond)].dropna(subset=['pred_mean', 'bispecific_median'])
        _row = metrics[(metrics['value_col'] == _vc) & (metrics['condition'] == _cond) & (metrics['operator'] == 'mean')].iloc[0]
        _ax.scatter(_g['pred_mean'], _g['bispecific_median'], s=14, alpha=0.7, edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
        _lo = float(min(_g['pred_mean'].min(), _g['bispecific_median'].min()))
        _hi = float(max(_g['pred_mean'].max(), _g['bispecific_median'].max()))
        _ax.plot([_lo, _hi], [_lo, _hi], 'k--', lw=0.6, alpha=0.5)
        equalize_axes(_ax)
        _title = f'{display_value_col(_vc, _cond)}\nρ={_row['spearman_rho']:.2f}, R²={_row['r2']:.2f}, n={int(_row['n'])}'
        _ax.set_title(wrap_title(_title, _cell_w), fontsize=FONT_SIZE_TITLE)
        _ax.set_xlabel('mean(parents)', fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel('Bispecific measured', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's03_baseline_scatter.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig
    _src = out[out['value_col'].isin(schema.REPORTED_VALUE_COLS)]
    _panels = _src.dropna(subset=['pred_mean', 'bispecific_median']).groupby(['value_col', 'condition']).size().reset_index(name='n').query('n >= 3')
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = 3
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fig_w, _fig_h, _cell_w = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_fig_w, _fig_h), layout='constrained')
    _axes_flat = _axes.flatten()
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _src[(_src['value_col'] == _vc) & (_src['condition'] == _cond)].dropna(subset=['pred_mean', 'bispecific_median'])
        _row = metrics[(metrics['value_col'] == _vc) & (metrics['condition'] == _cond) & (metrics['operator'] == 'mean')].iloc[0]
        _ax.scatter(_g['pred_mean'], _g['bispecific_median'], s=14, alpha=0.7, edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
        _lo = float(min(_g['pred_mean'].min(), _g['bispecific_median'].min()))
        _hi = float(max(_g['pred_mean'].max(), _g['bispecific_median'].max()))
        _ax.plot([_lo, _hi], [_lo, _hi], 'k--', lw=0.6, alpha=0.5)
        equalize_axes(_ax)
        _title = f'{display_value_col(_vc, _cond)}\nρ={_row['spearman_rho']:.2f}, R²={_row['r2']:.2f}, n={int(_row['n'])}'
        _ax.set_title(wrap_title(_title, _cell_w), fontsize=FONT_SIZE_TITLE)
        _ax.set_xlabel('mean(parents)', fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel('Bispecific measured', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _o = paths.FIGURES / 's03_baseline_scatter_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    _fig
    _OPS = ['mean', 'max', 'min']
    _OP_COLORS = {'mean': DATAPOINTS_COLORS['purple'], 'max': DATAPOINTS_COLORS['teal'], 'min': DATAPOINTS_COLORS['amber']}
    _N_MIN = 10
    _N_STEPS = 15
    _src = out[out['value_col'].isin(schema.REPORTED_VALUE_COLS)].copy()
    _src['abs_diff'] = (_src['parent_a_median'] - _src['parent_b_median']).abs()
    _src = _src.dropna(subset=['abs_diff', 'bispecific_median', 'pred_mean', 'pred_min', 'pred_max'])
    _panels = _src.groupby(['value_col', 'condition']).size().reset_index(name='n')
    _panels = _panels[_panels['n'] >= _N_MIN]
    _panels = list(zip(_panels['value_col'], _panels['condition']))
    _ncols = 4
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fig_w, _fig_h, _cell_w = grid_figsize(_nrows, _ncols)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_fig_w, _fig_h), layout='constrained')
    _axes_flat = _axes.flatten()
    for _ax, (_vc, _cond) in zip(_axes_flat, _panels):
        _g = _src[(_src['value_col'] == _vc) & (_src['condition'] == _cond)]
        _absd = _g['abs_diff'].to_numpy()
        _y_true = _g['bispecific_median'].to_numpy()
        _n = len(_g)
        _max_frac = max(0.0, 1.0 - _N_MIN / _n)
        _thresholds = np.unique(np.quantile(_absd, np.linspace(0.0, _max_frac, _N_STEPS)))
        for _op in _OPS:
            _pred = _g[f'pred_{_op}'].to_numpy()
            _xs, _ys = ([], [])
            for _t in _thresholds:
                _mask = _absd >= _t
                if _mask.sum() < _N_MIN:
                    continue
                _rho, _ = stats.spearmanr(_y_true[_mask], _pred[_mask])
                _xs.append(_t)
                _ys.append(_rho)
            _ax.plot(_xs, _ys, marker='o', markersize=3.5, linewidth=1.0, color=_OP_COLORS[_op], markeredgecolor=DATAPOINTS_COLORS['gray'], markeredgewidth=0.3, alpha=0.9, label=_op)
        _ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.4)
        _ax.set_title(wrap_title(display_value_col(_vc, _cond), _cell_w), fontsize=FONT_SIZE_TITLE)
        _ax.set_xlabel(wrap_label('Parent |Δ| threshold (≥)', _cell_w), fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel(wrap_label('Spearman ρ (subset)', _cell_w), fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
    _legend_handles = [_Line2D([], [], marker='o', color=_OP_COLORS[_op], markersize=5, markeredgecolor=DATAPOINTS_COLORS['gray'], markeredgewidth=0.3, linewidth=1.0, label=f'{_op}(parents)') for _op in _OPS]
    for _i, _ax in enumerate(_axes_flat[len(_panels):]):
        _ax.axis('off')
        if _i == 0:
            _ax.legend(handles=_legend_handles, fontsize=FONT_SIZE_LEGEND, loc='center', title='Operator', frameon=False)
    _o = paths.FIGURES / 's03_operator_vs_absdiff_threshold_reported_subset.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_o, dpi=300)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    _fig
if __name__ == '__main__':
    main()
