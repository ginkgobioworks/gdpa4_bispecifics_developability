"""Generate supplementary Figure S01.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s01.py``
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
from scipy import stats
from prophet_ab import paths, schema
from prophet_ab.features.naming import display_value_col
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FULL_WIDTH, equalize_axes, grid_figsize, wrap_title
import numpy as _np
from matplotlib.ticker import MaxNLocator

def main():
    set_manuscript_style()
    summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    _gdpa1_all = pd.read_parquet(paths.S03 / 'gdpa1_per_antibody.parquet')
    gdpa1 = _gdpa1_all[_gdpa1_all['hc_subtype'] == 'IgG1']
    monospecific_map = pd.read_parquet(paths.S02 / 'monospecific_gdpa1_map.parquet')
    monospecific = summaries[summaries['kind'] == schema.KIND_MONOSPECIFIC].merge(monospecific_map[['monospecific_antibody_name', 'monospecific_stripped']], left_on='antibody_name', right_on='monospecific_antibody_name', how='left')
    pairs = []
    for (this_vc, this_cond), gdpa1_col in schema.PROPHET_TO_GDPA1.items():
        ours = monospecific[(monospecific['value_col'] == this_vc) & (monospecific['condition'] == this_cond)][['monospecific_stripped', 'median']].rename(columns={'median': 'this_median'})
        theirs = gdpa1[gdpa1['value_col'] == gdpa1_col][['antibody_name', 'median']].rename(columns={'antibody_name': 'monospecific_stripped', 'median': 'gdpa1_median'})
        merged = ours.merge(theirs, on='monospecific_stripped', how='inner').dropna()
        merged['this_value_col'] = this_vc
        merged['this_condition'] = this_cond
        merged['gdpa1_col'] = gdpa1_col
        merged['panel'] = display_value_col(this_vc, this_cond)
        pairs.append(merged)
    paired = pd.concat(pairs, ignore_index=True)
    paired['this_plot'] = paired['this_median']
    paired['gdpa1_plot'] = paired['gdpa1_median']
    _acsins = paired['this_value_col'] == 'acsins_delta_Lmax'
    for _src, _dst in (('this_median', 'this_plot'), ('gdpa1_median', 'gdpa1_plot')):
        paired.loc[_acsins, _dst] = paired.loc[_acsins].groupby('panel')[_src].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    _rows = []
    for _panel, _g in paired.groupby('panel', sort=False):
        if len(_g) < 3:
            _rows.append(dict(panel=_panel, n=len(_g), pearson_r=None, p=None))
            continue
        _r, _p = stats.pearsonr(_g['this_median'], _g['gdpa1_median'])
        _rows.append(dict(panel=_panel, n=len(_g), pearson_r=_r, p=_p, this_value_col=_g['this_value_col'].iloc[0], this_condition=_g['this_condition'].iloc[0], gdpa1_col=_g['gdpa1_col'].iloc[0]))
    metrics = pd.DataFrame(_rows).sort_values('pearson_r', ascending=False)

    def _sync_ticks(ax, nticks=3):
        """Make both axes share identical ticks with the range max on a tick."""
        _lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        _hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        _loc = MaxNLocator(nbins=nticks * 2)
        _candidates = [t for t in _loc.tick_values(_lo, _hi) if _lo <= t <= _hi]
        _ticks = [_candidates[0], (_candidates[0] + _candidates[-1]) / 2, _candidates[-1]]
        _span = _ticks[-1] - _ticks[0]
        _pad = _span * 0.3
        ax.set_xticks(_ticks)
        ax.set_yticks(_ticks)
        ax.set_xlim(_ticks[0] - _pad, _ticks[-1] + _pad)
        ax.set_ylim(_ticks[0] - _pad, _ticks[-1] + _pad)
    _panels = list(metrics['panel'])
    _ncols = 4
    _nrows = (len(_panels) + _ncols - 1) // _ncols
    _fw, _fh, _cell_w = grid_figsize(_nrows, _ncols)
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(_fw, _fh), layout='constrained')
    _axes_flat = axes.flatten()
    for _ax, _panel in zip(_axes_flat, _panels):
        _g = paired[paired['panel'] == _panel]
        _r = metrics.loc[metrics['panel'] == _panel, 'pearson_r'].iloc[0]
        _n = metrics.loc[metrics['panel'] == _panel, 'n'].iloc[0]
        _ax.scatter(_g['this_plot'], _g['gdpa1_plot'], s=14, alpha=0.7, edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
        _sync_ticks(_ax)
        _lim = _ax.get_xlim()
        _ax.plot(_lim, _lim, 'k--', lw=0.6, alpha=0.5)
        _label = _panel.replace('AC-SINS ΔLmax', 'AC-SINS ΔLmax (norm)')
        if _label.startswith('PR Score @ '):
            _label = f'PR-{_label.removeprefix('PR Score @ ')} Score'
        else:
            _label = _label.replace(' @ ', ', ')
        _title = _label + f'\nr={_r:.2f}, n={_n}' if _r is not None else _label + f'\nn={_n}'
        _ax.set_title(wrap_title(_title, _cell_w))
        _ax.set_xlabel('GDPa4')
        _ax.set_ylabel('GDPa1')
    for _ax in _axes_flat[len(_panels):]:
        _ax.axis('off')
    _out = paths.FIGURES / 'supplementary/figure_s01.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _out = paths.TABLES / 's02_cross_platform_correlations.csv'
    _out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(_out, index=False)
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}  rows={len(metrics)}')
    emit('supp_01')
if __name__ == '__main__':
    main()
