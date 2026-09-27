"""Generate supplementary Figure S14.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s14.py``
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _source in (_ROOT / 'src', _ROOT / 'datapoints_figures' / 'src'):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
from prophet_ab.source_data import emit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from prophet_ab import paths
from prophet_ab.features.parental_surface import SURFACE_ASSAYS, highlight_table, load_example_parents, load_surface_long, summarize_parental_frontier
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE
from shutil import copy2

def main():
    set_manuscript_style()
    surface = load_surface_long()
    parents = load_example_parents()
    summary = summarize_parental_frontier(surface)
    highlight = highlight_table(surface)
    _table = paths.TABLES / 'brazi_lige_summary.csv'
    _table.parent.mkdir(parents=True, exist_ok=True)
    highlight.to_csv(_table, index=False)
    print(f'wrote {_table.relative_to(paths.REPO_ROOT)}  rows={len(highlight)}')
    print(f'brazikumab-containing: {int(summary['n_brazi_bsabs'])}   ligelizumab-containing: {int(summary['n_lige_bsabs'])}   brazi × lige: {int(summary['n_brazi_lige'])}')
    print(f'parental-mean HIC vs HAC: ρ = {summary['spearman_phic_phac']:.2f}  (n = {int(summary['spearman_n'])})')
    print(f'75th percentiles (Hazen): HIC {summary['hic_q75']:.3f}   HAC {summary['hac_q75']:.3f}   above both: {int(summary['n_double_frontier_all'])} / {int(summary['n_library'])}')
    _titles = {'hic': 'HIC normalized RT', 'hac': 'HAC retention time', 'pr_cho': 'PR-CHO score'}
    _pink = DATAPOINTS_COLORS['pink']
    _purple = DATAPOINTS_COLORS['purple']
    _gray = DATAPOINTS_COLORS['gray']
    _slate = DATAPOINTS_COLORS['slate']

    def _limits(values: np.ndarray, pad_frac: float=0.08) -> tuple[float, float]:
        lo = float(np.min(values))
        hi = float(np.max(values))
        span = hi - lo
        if span == 0:
            span = 1.0
        pad = pad_frac * span
        return (lo - pad, hi + pad)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.85), layout='constrained')
    for _ax, (_key, _value_col, _condition) in zip(axes, SURFACE_ASSAYS):
        _panel = surface[surface['assay'] == _key]
        _parent = parents[parents['assay'] == _key]
        _x = _panel['parental_mean'].to_numpy()
        _y = _panel['observed'].to_numpy()
        _lo, _hi = _limits(np.concatenate([_x, _y, _parent['value'].to_numpy()]))
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.plot([_lo, _hi], [_lo, _hi], ls='--', color=_gray, lw=0.8, zorder=1)
        _ax.scatter(_x, _y, s=22, facecolors='none', edgecolors=_gray, linewidths=0.7, zorder=2)
        _brazi = _panel[_panel['contains_brazikumab'] & ~_panel['is_brazi_lige_pair']]
        _lige = _panel[_panel['contains_ligelizumab'] & ~_panel['is_brazi_lige_pair']]
        _cross = _panel[_panel['is_brazi_lige_pair']]
        _ax.scatter(_brazi['parental_mean'], _brazi['observed'], s=42, color=_pink, edgecolors=_slate, linewidths=0.4, zorder=3)
        _ax.scatter(_lige['parental_mean'], _lige['observed'], s=42, color=_purple, edgecolors='white', linewidths=0.4, zorder=3)
        if len(_cross):
            _ax.scatter(_cross['parental_mean'], _cross['observed'], s=70, marker='*', color=DATAPOINTS_COLORS['navy'], edgecolors='white', linewidths=0.4, zorder=4)
        for _arm, _color in (('brazikumab', _pink), ('ligelizumab', _purple)):
            _value = float(_parent.loc[_parent['arm'] == _arm, 'value'].iloc[0])
            _ax.scatter([_value], [_value], s=160, marker='D', color='white', linewidths=0, zorder=5)
            _ax.scatter([_value], [_value], s=64, marker='D', color=_color, edgecolors=_slate, linewidths=0.6, zorder=6)
        _ax.set_box_aspect(1)
        _ax.set_title(f'{_titles[_key]}   (n = {len(_panel)})', fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
        _ax.set_xlabel('Parental mean', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
    axes[0].set_ylabel('Observed', fontsize=FONT_SIZE_LABEL)
    _handles = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='none', markeredgecolor=_gray, markeredgewidth=0.8, markersize=6, label='Other bispecifics'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=_pink, markeredgecolor=_slate, markeredgewidth=0.4, markersize=7, label='Brazikumab-containing'), Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=_purple, markeredgecolor='white', markeredgewidth=0.4, markersize=7, label='Ligelizumab-containing'), Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=_pink, markeredgecolor=_slate, markeredgewidth=0.6, markersize=7, label='Brazikumab mAb'), Line2D([0], [0], marker='D', linestyle='none', markerfacecolor=_purple, markeredgecolor=_slate, markeredgewidth=0.6, markersize=7, label='Ligelizumab mAb'), Line2D([0], [0], color=_gray, ls='--', lw=0.8, label='y = x')]
    if surface['is_brazi_lige_pair'].any():
        _handles.insert(3, Line2D([0], [0], marker='*', linestyle='none', markerfacecolor=DATAPOINTS_COLORS['navy'], markeredgecolor='white', markersize=12, label='Brazikumab × ligelizumab'))
    fig.legend(handles=_handles, loc='outside lower center', fontsize=FONT_SIZE_LEGEND, frameon=False, ncols=3)
    for _ax, _letter in zip(axes, 'ABC'):
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(-6, 4), textcoords='offset points', fontsize=14, fontweight='bold', ha='right', va='bottom')
    _out = paths.FIGURES / 'main' / 'supplementary_figure_s14.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    fig
    _src = paths.FIGURES / 'main/supplementary_figure_s14.png'
    _dst = paths.FIGURES / 'supplementary/figure_s14.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('supp_14')
if __name__ == '__main__':
    main()
