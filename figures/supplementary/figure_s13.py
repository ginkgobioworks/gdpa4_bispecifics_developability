"""Generate supplementary Figure S13.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s13.py``
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
from prophet_ab.features.thermal import THERMAL_METRICS, load_crossmab_deltas, round_half_away, summarize_delta
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK, FONT_SIZE_TITLE

def main():
    set_manuscript_style()
    deltas = load_crossmab_deltas()
    _table = paths.TABLES / 'crossmab_tm_deltas.csv'
    _table.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(_table, index=False)
    print(f'wrote {_table.relative_to(paths.REPO_ROOT)}  rows={len(deltas)}')
    summaries = {}
    for _key, _value_col, _condition in THERMAL_METRICS:
        _stats = summarize_delta(deltas[f'delta_{_key}'])
        summaries[_key] = _stats
        _iqr = f'[{round_half_away(_stats['q25']):+.2f}, {round_half_away(_stats['q75']):+.2f}]'
        print(f'{_condition}: n={int(_stats['n'])}  median={round_half_away(_stats['median']):+.2f}  IQR {_iqr}  <{'-3'}: {int(_stats['n_below_3'])}  <{'-5'}: {int(_stats['n_below_5'])}  <{'-7'}: {int(_stats['n_below_7'])}  range [{round_half_away(_stats['minimum']):+.2f}, {round_half_away(_stats['maximum']):+.2f}]')
    _edges = np.arange(-20, 11, 1)
    _labels = {'tm1': 'ΔTm1', 'tm2': 'ΔTm2', 'tonset': 'ΔTonset'}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), layout='constrained')
    for _ax, (_key, _value_col, _condition) in zip(axes, THERMAL_METRICS):
        _vals = deltas[f'delta_{_key}'].dropna().to_numpy()
        _stats = summaries[_key]
        _median = float(_stats['median'])
        _ax.hist(_vals, bins=_edges, color=DATAPOINTS_COLORS['blue'], edgecolor='white', linewidth=0.4, alpha=0.9, zorder=2)
        _ax.axvline(0, color=DATAPOINTS_COLORS['gray'], ls='--', lw=0.9, zorder=3)
        _ax.axvline(_median, color='black', ls='-', lw=1.5, zorder=4)
        _ax.set_xlim(_edges[0], _edges[-1])
        _ax.set_title(f'{_labels[_key]}   (n = {int(_stats['n'])})', fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
        _ax.set_xlabel('ΔT (°C)', fontsize=FONT_SIZE_LABEL)
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
        _ax.set_xticks([-20, -10, 0, 10])
        _note = f'median {round_half_away(_median):+.2f} °C\nIQR [{round_half_away(_stats['q25']):+.2f}, {round_half_away(_stats['q75']):+.2f}]\n< -7 °C: {_stats['pct_below_7']:.1f}% ({int(_stats['n_below_7'])})'
        _ax.text(0.03, 0.97, _note, transform=_ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE_LEGEND, color=DATAPOINTS_COLORS['slate'], bbox=dict(facecolor='white', edgecolor='none', alpha=0.92, pad=1.5), zorder=5)
    axes[0].set_ylabel('Bispecifics', fontsize=FONT_SIZE_LABEL)
    _ymax = max((_ax.get_ylim()[1] for _ax in axes))
    for _ax in axes:
        _ax.set_ylim(0, _ymax)
    _handles = [Line2D([0], [0], color=DATAPOINTS_COLORS['gray'], ls='--', lw=0.9, label='no change'), Line2D([0], [0], color='black', ls='-', lw=1.5, label='median')]
    fig.legend(handles=_handles, loc='outside lower center', fontsize=FONT_SIZE_LEGEND, frameon=False, ncols=2)
    for _ax, _letter in zip(axes, 'ABC'):
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(-6, 4), textcoords='offset points', fontsize=14, fontweight='bold', ha='right', va='bottom')
    _out = paths.FIGURES / 'supplementary/figure_s13.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    emit('supp_13')
if __name__ == '__main__':
    main()
