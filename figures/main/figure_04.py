"""Generate main manuscript Figure 04.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/main/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/main/figure_04.py``
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _source in (_ROOT / 'src', _ROOT / 'datapoints_figures' / 'src'):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
from prophet_ab.source_data import emit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from prophet_ab import paths, schema
from prophet_ab import normalize as nz
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, equalize_axes, wrap_title, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_TITLE, FONT_SIZE_LEGEND
from prophet_ab.features.charge import TRANSFORM_LABELS, TRANSFORM_ORDER, horserace as _horserace
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from shutil import copy2

def main():
    set_manuscript_style()
    _summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    _components = pd.read_parquet(paths.S02 / 'bispecific_components.parquet')
    _acsins = _summaries[_summaries['value_col'] == 'acsins_delta_Lmax'].copy()
    _monospecific = _acsins[_acsins['kind'] == schema.KIND_MONOSPECIFIC][['antibody_name', 'condition', 'median', 'std']].copy()
    _monospecific['parent'] = _monospecific['antibody_name'].map(nz.strip_isotype_suffix)
    _bispecific = _acsins[_acsins['kind'] == schema.KIND_BISPECIFIC][['antibody_name', 'condition', 'median']].rename(columns={'median': 'observed'})
    _bispecificp = _bispecific.merge(_components[['antibody_name', 'parent_a', 'parent_b']], on='antibody_name')
    _bispecificp = _bispecificp.merge(_monospecific[['parent', 'condition', 'median']].rename(columns={'parent': 'parent_a', 'median': 'pa_median'}), on=['parent_a', 'condition'], how='left').merge(_monospecific[['parent', 'condition', 'median']].rename(columns={'parent': 'parent_b', 'median': 'pb_median'}), on=['parent_b', 'condition'], how='left')
    _bispecificp['expected'] = (_bispecificp['pa_median'] + _bispecificp['pb_median']) / 2
    _bispecificp['residual'] = _bispecificp['observed'] - _bispecificp['expected']
    _pbs = _acsins[_acsins['condition'] == '1X PBS']
    _sigma_arm = float(np.nanmedian(_pbs[_pbs['kind'] == schema.KIND_MONOSPECIFIC]['std']))
    _sigma_bsab = float(np.nanmedian(_pbs[_pbs['kind'] == schema.KIND_BISPECIFIC]['std']))
    noise_pbs = float(np.sqrt(_sigma_bsab ** 2 + _sigma_arm ** 2 / 2))
    bispecific_acsins_pbs = _bispecificp[_bispecificp['condition'] == '1X PBS'].dropna(subset=['residual']).copy()
    NOISE_MULTIPLIER = 5
    ENHANCER_THRESHOLD = 17.51
    SUPPRESSOR_CEILING = 5.0
    _band = NOISE_MULTIPLIER * noise_pbs
    _g = bispecific_acsins_pbs.copy()
    _g['pair'] = _g.apply(lambda r: tuple(sorted([r['parent_a'], r['parent_b']])), axis=1)
    _rows = []
    for _pair, _grp in _g.groupby('pair'):
        _med = {}
        for _, _rr in _grp.iterrows():
            _med[_rr['parent_a']] = _rr['pa_median']
            _med[_rr['parent_b']] = _rr['pb_median']
        _observed = float(_grp['observed'].mean())
        _expected = float(_grp['expected'].mean())
        _residual = _observed - _expected
        if _residual >= _band and _observed >= ENHANCER_THRESHOLD:
            _cat = 'Enhancer'
        elif _residual <= -_band and _observed <= SUPPRESSOR_CEILING:
            _cat = 'Suppressor'
        else:
            _cat = ''
        _rows.append({'pair': _pair, 'parent_a': _pair[0], 'parent_b': _pair[1], 'pa_median': _med[_pair[0]], 'pb_median': _med[_pair[1]], 'observed': _observed, 'expected': _expected, 'residual': _residual, 'category': _cat, 'n_orientations': len(_grp)})
    classified_pbs = pd.DataFrame(_rows)
    band_half_width = _band
    print(f'Panel A (unique pairs, PBS): {(classified_pbs['category'] == 'Enhancer').sum()} enhancers, {(classified_pbs['category'] == 'Suppressor').sum()} suppressors, {len(classified_pbs)} pairs')
    _pairs = classified_pbs.copy()
    _monospecificf = pd.read_csv(paths.RAW_IN_SILICO_DIR / 'MOE_properties.csv')
    _monospecificf['parent'] = _monospecificf['antibody_name'].map(nz.strip_isotype_suffix)
    _charge_map = _monospecificf.set_index('parent')['ens_charge'].to_dict()
    _pairs['q1'] = _pairs['parent_a'].map(_charge_map)
    _pairs['q2'] = _pairs['parent_b'].map(_charge_map)
    _pairs['xc'] = _pairs[['q1', 'q2']].max(axis=1)
    _pairs['yc'] = _pairs[['q1', 'q2']].min(axis=1)
    pairs_charged = _pairs.dropna(subset=['q1', 'q2']).copy()
    print(f'Panels B/C (pair-level, charge subset): {(pairs_charged['category'] == 'Enhancer').sum()} enhancers, {(pairs_charged['category'] == 'Suppressor').sum()} suppressors, {(pairs_charged['category'] == '').sum()} not-classified ({len(pairs_charged)} of {len(_pairs)} pairs)')
    pairs_ungated = pairs_charged.copy()
    pairs_ungated['category'] = ''
    pairs_ungated.loc[pairs_ungated['residual'] >= band_half_width, 'category'] = 'Enhancer'
    pairs_ungated.loc[pairs_ungated['residual'] <= -band_half_width, 'category'] = 'Suppressor'
    horserace = _horserace(pairs_ungated, 'category')
    print(f'S2 ungated: {(pairs_ungated['category'] == 'Enhancer').sum()} enhancers, {(pairs_ungated['category'] == 'Suppressor').sum()} suppressors, {(pairs_ungated['category'] == '').sum()} on-parental')
    print(horserace[['side_label', 'transform', 'coef', 'wald_p', 'delta_aic', 'auc']].to_string(index=False))
    _CORAL = DATAPOINTS_COLORS['coral']
    _GREEN = DATAPOINTS_COLORS['green']
    _GRAY = DATAPOINTS_COLORS['gray']
    _SLATE = DATAPOINTS_COLORS['slate']
    fig = plt.figure(figsize=(7.5, 8.5), layout='constrained')
    _gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], wspace=0.1, hspace=0.12)
    _axA = fig.add_subplot(_gs[0, :])
    _axB_l = fig.add_subplot(_gs[1, 0])
    _axB_r = fig.add_subplot(_gs[1, 1])
    _g = classified_pbs
    _lo = float(min(_g['expected'].min(), _g['observed'].min()) - 2)
    _hi = float(max(_g['expected'].max(), _g['observed'].max()) + 2)
    _diag = np.linspace(_lo, _hi, 200)
    _agg_lo = np.maximum(_diag + band_half_width, ENHANCER_THRESHOLD)
    _res_hi = np.minimum(_diag - band_half_width, SUPPRESSOR_CEILING)
    _axA.fill_between(_diag, _agg_lo, _hi, where=_agg_lo < _hi, color=_CORAL, alpha=0.07, zorder=-2)
    _axA.fill_between(_diag, _lo, _res_hi, where=_res_hi > _lo, color=_GREEN, alpha=0.07, zorder=-2)
    _axA.fill_between(_diag, _diag - band_half_width, _diag + band_half_width, alpha=0.12, color='gray', zorder=0)
    _axA.plot(_diag, _diag, 'k--', lw=0.8, alpha=0.5, zorder=1)
    _axA.axhline(ENHANCER_THRESHOLD, color=_CORAL, ls=':', lw=1.2, alpha=0.8)
    _axA.axhline(SUPPRESSOR_CEILING, color=_GREEN, ls=':', lw=1.2, alpha=0.8)
    _nc = _g[_g['category'] == '']
    _agg = _g[_g['category'] == 'Enhancer']
    _res = _g[_g['category'] == 'Suppressor']
    _axA.scatter(_nc['expected'], _nc['observed'], s=18, c=_GRAY, alpha=0.45, edgecolors=_GRAY, linewidths=0.3, zorder=2)
    _axA.scatter(_agg['expected'], _agg['observed'], s=42, c=_CORAL, alpha=0.85, edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA.scatter(_res['expected'], _res['observed'], s=42, c=_GREEN, alpha=0.85, edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA.text(_lo + 0.8, ENHANCER_THRESHOLD + 1.0, f'high\nself-association\nobs > {ENHANCER_THRESHOLD:.2f} nm', ha='left', va='bottom', fontsize=FONT_SIZE_LEGEND, color=_CORAL, fontweight='semibold', linespacing=1.1)
    _axA.text(_hi - 0.8, SUPPRESSOR_CEILING - 1.0, f'low\nself-association\nobs < {SUPPRESSOR_CEILING:.0f} nm', ha='right', va='top', fontsize=FONT_SIZE_LEGEND, color=_GREEN, fontweight='semibold', linespacing=1.1)
    _axA.set_xlim(_lo, _hi)
    _axA.set_ylim(_lo, _hi)
    equalize_axes(_axA)
    _axA.set_box_aspect(1)
    _axA.set_xlabel('Expected (parent mean) AC-SINS ΔLmax  [nm]', fontsize=FONT_SIZE_LABEL)
    _axA.set_ylabel('Observed bispecific ΔLmax  [nm]', fontsize=FONT_SIZE_LABEL)
    _axA.set_title('AC-SINS at PBS pH 7.4', fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
    _axA.tick_params(labelsize=FONT_SIZE_TICK)
    _ncc = pairs_charged[pairs_charged['category'] == '']
    _aggc = pairs_charged[pairs_charged['category'] == 'Enhancer']
    _resc = pairs_charged[pairs_charged['category'] == 'Suppressor']
    _qmax = float(np.ceil(max(pairs_charged['q1'].abs().max(), pairs_charged['q2'].abs().max())) + 1)
    for _ax, _hi_set, _hi_color, _hi_label, _other in [(_axB_l, _aggc, _CORAL, 'Enhancers', _resc), (_axB_r, _resc, _GREEN, 'Suppressors', _aggc)]:
        _ax.fill_between([0, _qmax], 0, _qmax, color=_CORAL, alpha=0.04, zorder=0)
        _ax.fill_between([-_qmax, 0], -_qmax, 0, color=_CORAL, alpha=0.04, zorder=0)
        _ax.fill_between([0, _qmax], -_qmax, 0, color=_GREEN, alpha=0.04, zorder=0)
        _ax.axhline(0, color=_GRAY, lw=0.5, alpha=0.4, zorder=1)
        _ax.axvline(0, color=_GRAY, lw=0.5, alpha=0.4, zorder=1)
        _ax.plot([-_qmax, _qmax], [-_qmax, _qmax], '--', color=_GRAY, lw=0.8, alpha=0.5, zorder=1)
        _ax.scatter(_ncc['xc'], _ncc['yc'], s=22, color=_GRAY, alpha=0.45, edgecolors=_GRAY, linewidths=0.3, zorder=2)
        if len(_other):
            _ax.scatter(_other['xc'], _other['yc'], s=32, color=_GRAY, alpha=0.3, edgecolors=_GRAY, linewidths=0.3, zorder=3)
        if len(_hi_set):
            _ax.scatter(_hi_set['xc'], _hi_set['yc'], s=85, color=_hi_color, alpha=0.9, edgecolors=_SLATE, linewidths=0.4, zorder=5)
        _ax.text(_qmax * 0.3, _qmax * 0.7, '++ same sign\nreinforce', ha='center', va='center', fontsize=FONT_SIZE_LEGEND, color=_CORAL, fontweight='semibold', alpha=0.85)
        _ax.text(-_qmax * 0.7, -_qmax * 0.3, '−− same sign\nreinforce', ha='center', va='center', fontsize=FONT_SIZE_LEGEND, color=_CORAL, fontweight='semibold', alpha=0.85)
        _ax.text(_qmax * 0.55, -_qmax * 0.78, '+− opposite sign\ncancel', ha='center', va='center', fontsize=FONT_SIZE_LEGEND, color=_GREEN, fontweight='semibold', alpha=0.85)
        _ax.set_xlim(-_qmax, _qmax)
        _ax.set_ylim(-_qmax, _qmax)
        equalize_axes(_ax)
        _ax.set_box_aspect(1)
        _ax.set_xlabel('max(q1, q2)  stronger arm charge  [e]', fontsize=FONT_SIZE_LABEL)
        _ax.set_ylabel('min(q1, q2)  weaker arm charge  [e]', fontsize=FONT_SIZE_LABEL)
        _ax.set_title(_hi_label, fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
        _ax.tick_params(labelsize=FONT_SIZE_TICK)
    for _ax, _letter, _dx in [(_axA, 'A', -40), (_axB_l, 'B', -40)]:
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(_dx, 6), textcoords='offset points', fontsize=16, fontweight='bold', ha='left', va='bottom')
    _handles = [Line2D([], [], color=_GRAY, marker='o', linestyle='None', markersize=8, label='Not classified'), Line2D([], [], color=_CORAL, marker='o', linestyle='None', markersize=9, label='Enhancers'), Line2D([], [], color=_GREEN, marker='o', linestyle='None', markersize=9, label='Suppressors')]
    fig.legend(handles=_handles, loc='outside lower center', ncol=3, frameon=False, fontsize=FONT_SIZE_LEGEND, handletextpad=0.4, columnspacing=2.2)
    _o = paths.FIGURES / 'main' / 'figure_3.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig
    _CORAL = DATAPOINTS_COLORS['coral']
    _GREEN = DATAPOINTS_COLORS['green']
    _GRAY = DATAPOINTS_COLORS['gray']
    _SLATE = DATAPOINTS_COLORS['slate']
    _g = classified_pbs.copy()
    _g['category'] = np.where(_g['residual'].to_numpy() >= band_half_width, 'Enhancer', np.where(_g['residual'].to_numpy() <= -band_half_width, 'Suppressor', ''))
    _pairs = pairs_charged.drop(columns=['category']).merge(_g[['parent_a', 'parent_b', 'category']], on=['parent_a', 'parent_b'], how='left')
    _n_enh = int((_g['category'] == 'Enhancer').sum())
    _n_sup = int((_g['category'] == 'Suppressor').sum())
    print(f'figure_3_version_2 (band only): {_n_enh} enhancers, {_n_sup} suppressors, {len(_g)} pairs')
    _fig_v2 = plt.figure(figsize=(7.5, 9.2))
    _gs_v2 = _fig_v2.add_gridspec(2, 2, height_ratios=[1.35, 1.05], left=0.1, right=0.96, top=0.94, bottom=0.12, wspace=0.28, hspace=0.28)
    _axA_v2 = _fig_v2.add_subplot(_gs_v2[0, :])
    _axB_l_v2 = _fig_v2.add_subplot(_gs_v2[1, 0])
    _axB_r_v2 = _fig_v2.add_subplot(_gs_v2[1, 1])
    _lo = float(min(_g['expected'].min(), _g['observed'].min()) - 2)
    _hi = float(max(_g['expected'].max(), _g['observed'].max()) + 2)
    _diag = np.linspace(_lo, _hi, 200)
    _agg_lo = _diag + band_half_width
    _res_hi = _diag - band_half_width
    _axA_v2.fill_between(_diag, _agg_lo, _hi, where=_agg_lo < _hi, color=_CORAL, alpha=0.07, zorder=-2)
    _axA_v2.fill_between(_diag, _lo, _res_hi, where=_res_hi > _lo, color=_GREEN, alpha=0.07, zorder=-2)
    _axA_v2.fill_between(_diag, _diag - band_half_width, _diag + band_half_width, alpha=0.12, color='gray', zorder=0)
    _axA_v2.plot(_diag, _diag, 'k--', lw=0.8, alpha=0.5, zorder=1)
    _nc = _g[_g['category'] == '']
    _agg = _g[_g['category'] == 'Enhancer']
    _res = _g[_g['category'] == 'Suppressor']
    _axA_v2.scatter(_nc['expected'], _nc['observed'], s=18, c=_GRAY, alpha=0.45, edgecolors=_GRAY, linewidths=0.3, zorder=2)
    _axA_v2.scatter(_agg['expected'], _agg['observed'], s=42, c=_CORAL, alpha=0.85, edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA_v2.scatter(_res['expected'], _res['observed'], s=42, c=_GREEN, alpha=0.85, edgecolors=_SLATE, linewidths=0.4, zorder=5)
    _axA_v2.text(_lo + 0.8, _hi - 1.5, f'Enhancers  (n = {_n_enh})', ha='left', va='top', fontsize=FONT_SIZE_LEGEND, color=_CORAL, fontweight='semibold')
    _axA_v2.text(_hi - 0.8, _lo + 1.5, f'Suppressors  (n = {_n_sup})', ha='right', va='bottom', fontsize=FONT_SIZE_LEGEND, color=_GREEN, fontweight='semibold')
    _axA_v2.set_xlim(_lo, _hi)
    _axA_v2.set_ylim(_lo, _hi)
    equalize_axes(_axA_v2)
    _axA_v2.set_box_aspect(1)
    _axA_v2.set_xlabel('Expected (parent mean) AC-SINS ΔLmax  [nm]', fontsize=FONT_SIZE_LABEL)
    _axA_v2.set_ylabel('Observed bispecific ΔLmax  [nm]', fontsize=FONT_SIZE_LABEL)
    _axA_v2.set_title('AC-SINS at PBS pH 7.4', fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
    _axA_v2.tick_params(labelsize=FONT_SIZE_TICK)
    _ncc = _pairs[_pairs['category'] == '']
    _aggc = _pairs[_pairs['category'] == 'Enhancer']
    _resc = _pairs[_pairs['category'] == 'Suppressor']
    _qmax = float(np.ceil(max(_pairs['q1'].abs().max(), _pairs['q2'].abs().max())) + 1)
    _bins = np.linspace(-_qmax, _qmax, 25)
    for _sc, _hi_set, _hi_color, _hi_label, _other in [(_axB_l_v2, _aggc, _CORAL, f'Enhancers  (n = {_n_enh})', _resc), (_axB_r_v2, _resc, _GREEN, f'Suppressors  (n = {_n_sup})', _aggc)]:
        _s_nc = 6.0 + 5.0 * (np.abs(_ncc['q1'].to_numpy()) + np.abs(_ncc['q2'].to_numpy()))
        _sc.fill_between([0, _qmax], 0, _qmax, color=_CORAL, alpha=0.04, zorder=0)
        _sc.fill_between([-_qmax, 0], -_qmax, 0, color=_CORAL, alpha=0.04, zorder=0)
        _sc.fill_between([0, _qmax], -_qmax, 0, color=_GREEN, alpha=0.04, zorder=0)
        _sc.fill_between([-_qmax, 0], 0, _qmax, color=_GREEN, alpha=0.04, zorder=0)
        _sc.axhline(0, color=_GRAY, ls='--', lw=0.8, alpha=0.7, zorder=1)
        _sc.axvline(0, color=_GRAY, ls='--', lw=0.8, alpha=0.7, zorder=1)
        _sc.scatter(_ncc['q1'], _ncc['q2'], s=_s_nc, color=_GRAY, alpha=0.45, edgecolors=_GRAY, linewidths=0.3, zorder=2)
        if len(_other):
            _s_ot = 6.0 + 5.0 * (np.abs(_other['q1'].to_numpy()) + np.abs(_other['q2'].to_numpy()))
            _sc.scatter(_other['q1'], _other['q2'], s=_s_ot, color=_GRAY, alpha=0.3, edgecolors=_GRAY, linewidths=0.3, zorder=3)
        if len(_hi_set):
            _s_hi = 6.0 + 5.0 * (np.abs(_hi_set['q1'].to_numpy()) + np.abs(_hi_set['q2'].to_numpy()))
            _sc.scatter(_hi_set['q1'], _hi_set['q2'], s=_s_hi, color=_hi_color, alpha=0.9, edgecolors=_SLATE, linewidths=0.4, zorder=5)
        _sc.set_xlim(-_qmax, _qmax)
        _sc.set_ylim(-_qmax, _qmax)
        equalize_axes(_sc)
        _sc.set_box_aspect(1)
        _sc.set_xlabel('q1  arm charge  [e]', fontsize=FONT_SIZE_LABEL)
        _sc.set_ylabel('q2  arm charge  [e]', fontsize=FONT_SIZE_LABEL)
        _sc.tick_params(labelsize=FONT_SIZE_TICK)
        _div = _make_axes_locatable(_sc)
        _hx = _div.append_axes('top', size='18%', pad=0.06, sharex=_sc)
        _hy = _div.append_axes('right', size='18%', pad=0.06, sharey=_sc)
        _hx.hist(_pairs['q1'].to_numpy(), bins=_bins, color=_GRAY, alpha=0.4, density=True, orientation='vertical', histtype='stepfilled')
        _hy.hist(_pairs['q2'].to_numpy(), bins=_bins, color=_GRAY, alpha=0.4, density=True, orientation='horizontal', histtype='stepfilled')
        if len(_hi_set):
            _hx.hist(_hi_set['q1'].to_numpy(), bins=_bins, color=_hi_color, alpha=0.7, density=True, orientation='vertical', histtype='stepfilled')
            _hy.hist(_hi_set['q2'].to_numpy(), bins=_bins, color=_hi_color, alpha=0.7, density=True, orientation='horizontal', histtype='stepfilled')
        _hx.tick_params(labelbottom=False, labelleft=False, length=0)
        _hy.tick_params(labelbottom=False, labelleft=False, length=0)
        for _sp in list(_hx.spines.values()) + list(_hy.spines.values()):
            _sp.set_visible(False)
        _hx.set_facecolor('none')
        _hy.set_facecolor('none')
        _hx.set_xlim(-_qmax, _qmax)
        _hy.set_ylim(-_qmax, _qmax)
        _hx.set_title(_hi_label, fontsize=FONT_SIZE_TITLE, fontweight='semibold', loc='left')
    for _let_ax, _letter, _dx in [(_axA_v2, 'A', -40), (_axB_l_v2, 'B', -40)]:
        _let_ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(_dx, 6), textcoords='offset points', fontsize=16, fontweight='bold', ha='left', va='bottom')
    _handles = [Line2D([], [], color=_GRAY, marker='o', linestyle='None', markersize=8, label='Not classified'), Line2D([], [], color=_CORAL, marker='o', linestyle='None', markersize=9, label='Enhancers'), Line2D([], [], color=_GREEN, marker='o', linestyle='None', markersize=9, label='Suppressors')]
    _fig_v2.legend(handles=_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=FONT_SIZE_LEGEND, handletextpad=0.4, columnspacing=1.8)
    _o_v2 = paths.FIGURES / 'main' / 'figure_3_version_2.png'
    _o_v2.parent.mkdir(parents=True, exist_ok=True)
    _fig_v2.savefig(_o_v2, dpi=300, bbox_inches='tight')
    print(f'wrote {_o_v2.relative_to(paths.REPO_ROOT)}')
    _fig_v2
    _CORAL = DATAPOINTS_COLORS['coral']
    _GREEN = DATAPOINTS_COLORS['green']
    _GRAY = DATAPOINTS_COLORS['gray']
    _SLATE = DATAPOINTS_COLORS['slate']
    fig_c = plt.figure(figsize=(13.0, 4.5), layout='constrained')
    _gs = fig_c.add_gridspec(1, 3, wspace=0.22)
    _axC_bar = fig_c.add_subplot(_gs[0, 0])
    _axC_a = fig_c.add_subplot(_gs[0, 1])
    _axC_r = fig_c.add_subplot(_gs[0, 2])
    _hr_a = horserace[horserace['side'] == 'enh'].set_index('transform')
    _hr_r = horserace[horserace['side'] == 'suppress'].set_index('transform')
    _delta_a = [_hr_a.loc[t, 'delta_aic'] for t in TRANSFORM_ORDER]
    _delta_r = [_hr_r.loc[t, 'delta_aic'] for t in TRANSFORM_ORDER]
    _p_a = [_hr_a.loc[t, 'wald_p'] for t in TRANSFORM_ORDER]
    _p_r = [_hr_r.loc[t, 'wald_p'] for t in TRANSFORM_ORDER]
    _y = np.arange(len(TRANSFORM_ORDER))[::-1]
    _h = 0.36
    _bars_a = _axC_bar.barh(_y + _h / 2, _delta_a, _h, color=_CORAL, edgecolor=_SLATE, linewidth=0.6, label='Enhancers')
    _bars_r = _axC_bar.barh(_y - _h / 2, _delta_r, _h, color=_GREEN, edgecolor=_SLATE, linewidth=0.6, label='Suppressors')
    for _bars, _deltas, _pvals in [(_bars_a, _delta_a, _p_a), (_bars_r, _delta_r, _p_r)]:
        for _bar, _val, _pval in zip(_bars, _deltas, _pvals):
            if abs(_val) < 1e-06 and _pval < 0.05:
                _axC_bar.text(_bar.get_width() + 0.2, _bar.get_y() + _bar.get_height() / 2, '*', va='center', fontsize=FONT_SIZE_LEGEND, fontweight='bold')
    _axC_bar.set_yticks(_y)
    _axC_bar.set_yticklabels([TRANSFORM_LABELS[t] for t in TRANSFORM_ORDER], fontsize=FONT_SIZE_TICK)
    _axC_bar.set_xlabel('ΔAIC', fontsize=FONT_SIZE_LABEL)
    _axC_bar.set_title('Charge-transform horserace\n(single-predictor logistic\nregression, ΔAIC)', fontsize=FONT_SIZE_TITLE, loc='left')
    _axC_bar.axvline(0, color='black', lw=0.5)
    _axC_bar.axvspan(0, 2, color='gray', alpha=0.1, zorder=0)
    _axC_bar.grid(axis='x', ls=':', alpha=0.4)
    _axC_bar.set_xlim(left=-0.3)
    _axC_bar.set_box_aspect(1)

    def _signed_gmean(q1, q2):
        _a = np.asarray(q1, dtype=float)
        _b = np.asarray(q2, dtype=float)
        return -np.sign(_a * _b) * np.sqrt(np.abs(_a * _b))
    _ncc = pairs_ungated[pairs_ungated['category'] == '']
    _aggc = pairs_ungated[pairs_ungated['category'] == 'Enhancer']
    _resc = pairs_ungated[pairs_ungated['category'] == 'Suppressor']
    _rng = np.random.default_rng(0)

    def _violin(_ax, _vals_nc, _vals_set, _title, _ylabel, _color, _set_label):
        _u, _p = mannwhitneyu(_vals_set, _vals_nc, alternative='greater')
        _parts = _ax.violinplot([_vals_nc, _vals_set], positions=[0, 1], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        for _i, _body in enumerate(_parts['bodies']):
            _body.set_facecolor(_GRAY if _i == 0 else _color)
            _body.set_edgecolor(_SLATE)
            _body.set_linewidth(0.4)
            _body.set_alpha(0.65)
        for _pos, _vals, _c in [(0, _vals_nc, _GRAY), (1, _vals_set, _color)]:
            _jit = _rng.uniform(-0.08, 0.08, size=len(_vals))
            _ax.scatter(np.full(len(_vals), _pos) + _jit, _vals, s=16, alpha=0.55, color=_c, edgecolors=_GRAY, linewidths=0.3)
            _ax.plot([_pos - 0.22, _pos + 0.22], [np.median(_vals)] * 2, color='black', lw=2.2)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(['n.c.', _set_label], fontsize=FONT_SIZE_TICK)
        _ax.set_ylabel(_ylabel, fontsize=FONT_SIZE_LABEL)
        _ax.set_title(_title, fontsize=FONT_SIZE_TITLE)
        _ax.text(0.5, 0.98, f'MWU greater: p = {_p:.3g}\nmed {np.median(_vals_nc):.2f} vs {np.median(_vals_set):.2f}', transform=_ax.transAxes, ha='center', va='top', fontsize=FONT_SIZE_LEGEND, bbox=dict(facecolor='white', edgecolor=_GRAY, alpha=0.95, pad=3))
        _ax.grid(axis='y', ls=':', alpha=0.4)
        _ax.set_box_aspect(1)
    _violin(_axC_a, np.abs(_ncc['q1'].values + _ncc['q2'].values), np.abs(_aggc['q1'].values + _aggc['q2'].values), 'Enhancer · charge', '|q1 + q2|  [e]', _CORAL, 'enh')
    _violin(_axC_r, _signed_gmean(_ncc['q1'].values, _ncc['q2'].values), _signed_gmean(_resc['q1'].values, _resc['q2'].values), 'Suppressor · charge', '−sgn(q1q2)·√|q1q2|', _GREEN, 'sup')
    for _ax, _letter, _dx in [(_axC_bar, 'A', -62), (_axC_a, 'B', -40), (_axC_r, 'C', -40)]:
        _ax.annotate(_letter, xy=(0, 1), xycoords='axes fraction', xytext=(_dx, 6), textcoords='offset points', fontsize=16, fontweight='bold', ha='left', va='bottom')
    _o = paths.FIGURES / 'main' / 'figure_4.png'
    _o.parent.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(_o, dpi=300, bbox_inches='tight')
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}')
    fig_c
    _src = paths.FIGURES / 'main/figure_3_version_2.png'
    _dst = paths.FIGURES / 'main/figure_04.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('main_04')
if __name__ == '__main__':
    main()
