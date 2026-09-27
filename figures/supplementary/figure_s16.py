"""Generate supplementary Figure S16.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s16.py``
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _source in (_ROOT / 'src', _ROOT / 'datapoints_figures' / 'src'):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
from prophet_ab.source_data import emit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from prophet_ab import paths
from prophet_ab.features.orientation import ORIENTATION_ASSAYS, load_orientation_pairs, orientation_correlations, orientation_pair_index, source_pair_table
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE
from shutil import copy2

def main():
    set_manuscript_style()
    library_pairs = orientation_pair_index()
    pairs = load_orientation_pairs()
    correlations = orientation_correlations(pairs)
    table = source_pair_table(pairs, correlations)
    _out_table = paths.TABLES / 'orientation_pairs_summary.csv'
    _out_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(_out_table, index=False)
    print(f'orientation-paired unordered pairs: {len(library_pairs)}  wrote {_out_table.relative_to(paths.REPO_ROOT)}  rows={len(table)}')
    for _row in correlations.itertuples(index=False):
        print(f'{_row.panel}  {_row.assay}:  n={_row.n}  ρ={_row.spearman_rho:.2f}')
    _blue = DATAPOINTS_COLORS['blue']
    _gray = DATAPOINTS_COLORS['gray']
    _slate = DATAPOINTS_COLORS['slate']
    fig = plt.figure(figsize=(10.5, 7.55), layout='constrained')
    _gs = fig.add_gridspec(2, 6)
    _axes = [fig.add_subplot(_gs[0, 0:2]), fig.add_subplot(_gs[0, 2:4]), fig.add_subplot(_gs[0, 4:6]), fig.add_subplot(_gs[1, 1:3]), fig.add_subplot(_gs[1, 3:5])]
    for _ax, (_panel, _key, _assay, _value_col, _condition) in zip(_axes, ORIENTATION_ASSAYS):
        _block = pairs.loc[pairs['key'] == _key]
        _stats = correlations.loc[correlations['key'] == _key].iloc[0]
        _x = _block['A_B'].to_numpy()
        _y = _block['B_A'].to_numpy()
        _lo = float(min(_x.min(), _y.min()))
        _hi = float(max(_x.max(), _y.max()))
        _span = _hi - _lo
        if _span == 0:
            _span = 1.0
        _pad = 0.06 * _span
        _lo -= _pad
        _hi += _pad
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.plot([_lo, _hi], [_lo, _hi], ls='--', color=_gray, lw=0.8, zorder=1)
        _ax.scatter(_x, _y, s=22, alpha=0.8, color=_blue, edgecolors=_gray, linewidths=0.3, zorder=2)
        _ax.set_box_aspect(1)
        _ax.set_title(_assay, fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
        _ax.set_xlabel('A-B configuration', fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel('B-A configuration', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
        _ax.text(0.96, 0.05, f'ρ = {_stats['spearman_rho']:.2f}\nn = {int(_stats['n'])}', transform=_ax.transAxes, ha='right', va='bottom', fontsize=FONT_SIZE_LEGEND, color=_slate, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5), zorder=3)
    fig.legend(handles=[Line2D([0], [0], color=_gray, ls='--', lw=0.8, label='y = x')], loc='outside lower center', fontsize=FONT_SIZE_LEGEND, frameon=False)
    for _ax, _letter in zip(_axes, 'ABCDE'):
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(-6, 4), textcoords='offset points', fontsize=14, fontweight='bold', ha='right', va='bottom')
    _out = paths.FIGURES / 'main' / 'supplementary_figure_s16.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    fig
    _src = paths.FIGURES / 'main/supplementary_figure_s16.png'
    _dst = paths.FIGURES / 'supplementary/figure_s16.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('supp_16')
if __name__ == '__main__':
    main()
