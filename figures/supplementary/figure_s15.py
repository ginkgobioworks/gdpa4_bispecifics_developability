"""Generate supplementary Figure S15.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s15.py``
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
from prophet_ab.features.production_qc import PRODUCTION_ATTRIBUTES, load_production_pairs, tier_assignments
from prophet_ab.features.transforms import TIER_COLORS
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE
from shutil import copy2

def main():
    set_manuscript_style()
    pairs = load_production_pairs()
    assignments = tier_assignments(pairs)
    _table = paths.TABLES / 'production_qc_tier_assignments.csv'
    _table.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(_table, index=False)
    print(f'wrote {_table.relative_to(paths.REPO_ROOT)}  rows={len(assignments)}')
    for _row in assignments.itertuples(index=False):
        print(f'{_row.panel}  {_row.attribute}:  n={_row.n}  ρ={_row.spearman_rho:.3f}  r={_row.pearson_r:.3f}  median residual={_row.median_residual:+.2f}  σ residual={_row.sigma_residual:.3f}  σ arm={_row.sigma_arm:.3f}  Class {_row.tier}')
    _tier_color = {'I': TIER_COLORS[1], 'II': TIER_COLORS[2], 'III': TIER_COLORS[3]}
    _gray = DATAPOINTS_COLORS['gray']
    _slate = DATAPOINTS_COLORS['slate']
    fig, _axes = plt.subplots(2, 2, figsize=(7.4, 7.5), layout='constrained')
    for _ax, (_panel, _key, _attribute, _value_col, _condition) in zip(_axes.ravel(), PRODUCTION_ATTRIBUTES):
        _block = pairs.loc[pairs['key'] == _key]
        _stats = assignments.loc[assignments['key'] == _key].iloc[0]
        _x = _block['parental_mean'].to_numpy()
        _y = _block['observed'].to_numpy()
        _lo = float(min(_x.min(), _y.min()))
        _hi = float(max(_x.max(), _y.max()))
        _span = _hi - _lo
        if _span == 0:
            _span = 1.0
        _pad = 0.06 * _span
        _lo -= _pad
        _hi += _pad
        _color = _tier_color[str(_stats['tier'])]
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.plot([_lo, _hi], [_lo, _hi], ls='--', color=_gray, lw=0.8, zorder=1)
        _ax.scatter(_x, _y, s=22, alpha=0.8, color=_color, edgecolors=_gray, linewidths=0.3, zorder=2)
        _ax.set_box_aspect(1)
        _ax.set_title(_attribute, fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
        _ax.set_xlabel('Parental mean', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
        _ax.text(0.04, 0.96, f'ρ = {_stats['spearman_rho']:.3f}\nn = {int(_stats['n'])}\nClass {_stats['tier']}', transform=_ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE_LEGEND, color=_slate, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5), zorder=3)
    _axes[0, 0].set_ylabel('Observed', fontsize=FONT_SIZE_LABEL)
    _axes[1, 0].set_ylabel('Observed', fontsize=FONT_SIZE_LABEL)
    fig.legend(handles=[Line2D([0], [0], color=_gray, ls='--', lw=0.8, label='y = x')], loc='outside lower center', fontsize=FONT_SIZE_LEGEND, frameon=False)
    for _ax, _letter in zip(_axes.ravel(), 'ABCD'):
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(-6, 4), textcoords='offset points', fontsize=14, fontweight='bold', ha='right', va='bottom')
    _out = paths.FIGURES / 'main' / 'supplementary_figure_s15.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    fig
    _src = paths.FIGURES / 'main/supplementary_figure_s15.png'
    _dst = paths.FIGURES / 'supplementary/figure_s15.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('supp_15')
if __name__ == '__main__':
    main()
