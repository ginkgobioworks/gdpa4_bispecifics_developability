"""Generate supplementary Figure S11.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s11.py``
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _source in (_ROOT / 'src', _ROOT / 'datapoints_figures' / 'src'):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
from prophet_ab.source_data import emit
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from itertools import combinations
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from prophet_ab import paths
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FILL_ALPHA, SINGLE_COL_WIDTH, FULL_WIDTH, FONT_SIZE_TICK, FONT_SIZE_LABEL, FONT_SIZE_TITLE, FONT_SIZE_LEGEND, equalize_axes, grid_figsize, wrap_label, wrap_title
from scipy.ndimage import gaussian_filter
from shutil import copy2

def main():
    set_manuscript_style()
    DESIGN_DATA = paths.RAW / 'bsab_design' / 'bsabs_dec_2025' / 'data'
    _raw = pd.read_csv(DESIGN_DATA / 'GDPa1_v1.2_20250814.csv')

    def _seq(row):
        vh = row.get('vh_protein_sequence')
        vl = row.get('vl_protein_sequence')
        parts = []
        if pd.notna(vh):
            parts.append(str(vh).strip())
        if pd.notna(vl):
            parts.append(str(vl).strip())
        return ''.join(parts).replace(' ', '').upper()

    def _net_charge_at_pH(seq, pH=7.4):
        if not seq:
            return 0.0
        nK = seq.count('K')
        nR = seq.count('R')
        nH = seq.count('H')
        nD = seq.count('D')
        nE = seq.count('E')
        q = 0.0
        q += nK / (1 + 10 ** (pH - 10.53))
        q += nR / (1 + 10 ** (pH - 12.48))
        q += nH / (1 + 10 ** (pH - 6.0))
        q += 1.0 / (1 + 10 ** (pH - 8.0))
        q -= nD / (1 + 10 ** (3.65 - pH))
        q -= nE / (1 + 10 ** (4.25 - pH))
        q -= 1.0 / (1 + 10 ** (3.1 - pH))
        return q

    def _props(row):
        s = _seq(row)
        vh = row.get('vh_protein_sequence')
        vl = row.get('vl_protein_sequence')
        vh_str = str(vh).strip().upper() if pd.notna(vh) else ''
        vl_str = str(vl).strip().upper() if pd.notna(vl) else ''
        nans = {'pI': np.nan, 'hydrophobicity': np.nan, 'aromaticity': np.nan, 'instability_index': np.nan, 'length': np.nan, 'net_charge_vh': np.nan, 'net_charge_vl': np.nan, 'net_charge': np.nan}
        if not s:
            return pd.Series(nans)
        try:
            pa = ProteinAnalysis(s)
            vh_q = _net_charge_at_pH(vh_str)
            vl_q = _net_charge_at_pH(vl_str)
            return pd.Series({'pI': pa.isoelectric_point(), 'hydrophobicity': pa.gravy(), 'aromaticity': pa.aromaticity(), 'instability_index': pa.instability_index(), 'length': float(len(s)), 'net_charge_vh': vh_q, 'net_charge_vl': vl_q, 'net_charge': vh_q + vl_q})
        except Exception:
            return pd.Series(nans)
    _computed = _raw.apply(_props, axis=1)
    _raw = pd.concat([_raw, _computed], axis=1)
    _moe = pd.read_csv(DESIGN_DATA / 'p739_moe_properties.csv')
    df_mabs = _raw.merge(_moe, on='antibody_name', how='left', suffixes=('', '_moe'))
    print(f'GDPa1 mAbs: {len(df_mabs)}')
    df_mabs
    assays = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2']
    computed_props = ['pI', 'hydrophobicity', 'aromaticity', 'instability_index', 'length']
    moe_props = ['patch_cdr_hyd', 'ens_charge', 'dipole_moment', 'affinity_VL_VH']
    charge_props = ['net_charge_vh', 'net_charge_vl', 'net_charge']
    cols_of_interest = assays + computed_props + moe_props
    df_exported = pd.read_csv(DESIGN_DATA / 'exported_bsabs.csv')
    N_STRAT1 = 95
    df_strat1_raw = df_exported.iloc[:N_STRAT1].copy()
    df_strat2_raw = df_exported.iloc[N_STRAT1:].copy()
    mab_subsample_names = sorted(set(df_strat1_raw['antibody_name-1'].tolist() + df_strat1_raw['antibody_name-2'].tolist()))
    print(f'Total exported bsAbs: {len(df_exported)}')
    print(f'Strategy 1 bsAbs: {len(df_strat1_raw)}')
    print(f'Strategy 2 bsAbs: {len(df_strat2_raw)}')
    print(f'Strategy 1 unique parents (FPS-selected mAbs): {len(mab_subsample_names)}')
    df_mab_subsample = df_mabs[df_mabs['antibody_name'].isin(mab_subsample_names)].copy()
    print(f'mAb subsample: {len(df_mab_subsample)} of {len(mab_subsample_names)} found')
    df_mab_subsample
    _n = len(df_mabs)
    _idx = np.array(list(combinations(range(_n), 2)))
    _names = df_mabs['antibody_name'].values
    _data = {'antibody_name-1': _names[_idx[:, 0]], 'antibody_name-2': _names[_idx[:, 1]]}
    for _col in cols_of_interest:
        _v = df_mabs[_col].values.astype(float)
        _data[f'{_col}_avg'] = (_v[_idx[:, 0]] + _v[_idx[:, 1]]) / 2
        _data[f'{_col}_diff'] = _v[_idx[:, 0]] - _v[_idx[:, 1]]
    df_all_bsabs = pd.DataFrame(_data)
    print(f'All possible bsAbs: {len(df_all_bsabs):,}')
    df_all_bsabs
    _lookup = df_mabs.set_index('antibody_name')

    def _compute_bsab_stats(df_raw):
        result = pd.DataFrame()
        result['antibody_name-1'] = df_raw['antibody_name-1'].values
        result['antibody_name-2'] = df_raw['antibody_name-2'].values
        for col in cols_of_interest:
            v1 = df_raw['antibody_name-1'].map(_lookup[col]).astype(float)
            v2 = df_raw['antibody_name-2'].map(_lookup[col]).astype(float)
            result[f'{col}_avg'] = (v1.values + v2.values) / 2
            result[f'{col}_diff'] = v1.values - v2.values
        return result
    df_strat1 = _compute_bsab_stats(df_strat1_raw)
    df_strat2 = _compute_bsab_stats(df_strat2_raw)
    print(f'Strategy 1 stats: {len(df_strat1)}, Strategy 2 stats: {len(df_strat2)}')

    def make_coverage_pairplot(df_full, df_sub, columns, title='', mode='scatter', save_path=None):
        full_c = DATAPOINTS_COLORS['gray']
        sel_c = DATAPOINTS_COLORS['blue']
        n = len(columns)
        _df_f = df_full[columns].dropna().assign(_dataset='full')
        _df_s = df_sub[columns].dropna().assign(_dataset='selected')
        _combined = pd.concat([_df_f, _df_s], ignore_index=True)
        g = sns.PairGrid(_combined, vars=columns, hue='_dataset', palette={'full': full_c, 'selected': sel_c}, height=1.1, aspect=1, diag_sharey=False)

        def _diag_hist(data, color, label, **kw):
            ax = plt.gca()
            alpha = FILL_ALPHA if label == 'full' else 0.6
            all_vals = _combined[data.name].dropna()
            bins = np.histogram_bin_edges(all_vals, bins=30)
            ax.hist(data.dropna(), bins=bins, density=True, alpha=alpha, color=color)
        g.map_diag(_diag_hist)
        if mode == 'scatter':

            def _offdiag_scatter(x, y, color, label, **kw):
                ax = plt.gca()
                if label == 'full':
                    ax.scatter(x, y, color=color, alpha=0.3, s=10, edgecolors=full_c, linewidths=0.3)
                else:
                    ax.scatter(x, y, color=color, alpha=0.8, s=25, edgecolors=full_c, linewidths=0.3)
            g.map_offdiag(_offdiag_scatter)
        elif mode == 'hex':

            def _offdiag_hex(x, y, color, label, **kw):
                ax = plt.gca()
                mask = np.isfinite(x) & np.isfinite(y)
                if label == 'full':
                    if mask.sum() > 0:
                        ax.hexbin(x[mask], y[mask], gridsize=15, cmap='Greys', alpha=0.6, mincnt=1)
                else:
                    ax.scatter(x[mask], y[mask], color=color, alpha=0.7, s=10, edgecolors=full_c, linewidths=0.3)
            g.map_offdiag(_offdiag_hex)
        elif mode == 'contour':

            def _offdiag_contour(x, y, color, label, **kw):
                ax = plt.gca()
                xv = np.asarray(x, dtype=float)
                yv = np.asarray(y, dtype=float)
                mask = np.isfinite(xv) & np.isfinite(yv)
                if label == 'full':
                    if mask.sum() > 10:
                        try:
                            H, xe, ye = np.histogram2d(xv[mask], yv[mask], bins=20)
                            Xm, Ym = np.meshgrid(xe[:-1], ye[:-1])
                            ax.contourf(Xm, Ym, H.T, levels=10, cmap='Greys', alpha=FILL_ALPHA)
                            ax.contour(Xm, Ym, H.T, levels=10, colors=full_c, alpha=0.4, linewidths=0.5)
                        except Exception:
                            ax.scatter(xv[mask], yv[mask], color=color, alpha=0.2, s=1, edgecolors=full_c, linewidths=0.3)
                elif mask.sum() > 15:
                    try:
                        Hs, xes, yes = np.histogram2d(xv[mask], yv[mask], bins=12)
                        Xms, Yms = np.meshgrid(xes[:-1], yes[:-1])
                        ax.contourf(Xms, Yms, Hs.T, levels=8, cmap='Blues', alpha=FILL_ALPHA)
                        ax.contour(Xms, Yms, Hs.T, levels=8, colors=color, alpha=0.8, linewidths=1.0)
                    except Exception:
                        ax.scatter(xv[mask], yv[mask], color=color, alpha=0.8, s=15, edgecolors=full_c, linewidths=0.3)
                elif mask.sum() > 0:
                    ax.scatter(xv[mask], yv[mask], color=color, alpha=0.8, s=15, edgecolors=full_c, linewidths=0.3)
            g.map_offdiag(_offdiag_contour)
        for ax in g.axes.flat:
            ax.tick_params(labelsize=FONT_SIZE_TICK - 4)
        for ax in g.axes[:, 0]:
            ax.yaxis.label.set_size(FONT_SIZE_LABEL - 4)
        for ax in g.axes[-1, :]:
            ax.xaxis.label.set_size(FONT_SIZE_LABEL - 4)
        g._legend_data = {}
        legend_handles = [Line2D([0], [0], color=full_c, marker='o', linestyle='None', markersize=4, label='Full dataset'), Line2D([0], [0], color=sel_c, marker='o', linestyle='None', markersize=4, label='Selected')]
        g.figure.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=FONT_SIZE_LEGEND)
        if title:
            g.figure.suptitle(title, y=1.01, fontsize=FONT_SIZE_TITLE)
        g.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            g.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        return g.figure
    _groups = [(assays, 'assays', 'Experimental assays'), (computed_props, 'computed', 'Computed properties'), (moe_props, 'moe', 'MOE properties'), (charge_props, 'charge', 'Net charge (VH + VL)')]
    _figs = []
    for _cols, _slug, _label in _groups:
        _path = paths.FIGURES / f's00_mab_coverage_{_slug}.png'
        _fig = make_coverage_pairplot(df_mabs, df_mab_subsample, _cols, title=f'GDPa1 mAbs — {_label}', mode='scatter', save_path=_path)
        _figs.append(_fig)
        print(f'wrote {_path.relative_to(paths.REPO_ROOT)}')
    _charge = df_mabs.set_index('antibody_name')['net_charge']
    _q1_bg = df_all_bsabs['antibody_name-1'].map(_charge)
    _q2_bg = df_all_bsabs['antibody_name-2'].map(_charge)
    _min_bg = np.minimum(_q1_bg, _q2_bg)
    _max_bg = np.maximum(_q1_bg, _q2_bg)
    _q1_sel = df_exported['antibody_name-1'].map(_charge)
    _q2_sel = df_exported['antibody_name-2'].map(_charge)
    _min_sel = np.minimum(_q1_sel, _q2_sel)
    _max_sel = np.maximum(_q1_sel, _q2_sel)
    _ok_bg = np.isfinite(_min_bg) & np.isfinite(_max_bg)
    _ok_sel = np.isfinite(_min_sel) & np.isfinite(_max_sel)
    _fig, _ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
    _ax.scatter(_max_bg[_ok_bg], _min_bg[_ok_bg], s=4, alpha=0.15, color=DATAPOINTS_COLORS['gray'], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3, rasterized=True)
    _ax.scatter(_max_sel[_ok_sel], _min_sel[_ok_sel], s=25, alpha=0.8, color=DATAPOINTS_COLORS['blue'], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3)
    _lims = [min(_ax.get_xlim()[0], _ax.get_ylim()[0]), max(_ax.get_xlim()[1], _ax.get_ylim()[1])]
    _ax.plot(_lims, _lims, '--', color=DATAPOINTS_COLORS['gray'], linewidth=0.5, alpha=0.5)
    _ax.set_xlabel('max(q_arm1, q_arm2)', fontsize=FONT_SIZE_LABEL)
    _ax.set_ylabel('min(q_arm1, q_arm2)', fontsize=FONT_SIZE_LABEL)
    _ax.set_title('Arm net charge coverage (pH 7.4)', fontsize=FONT_SIZE_TITLE)
    equalize_axes(_ax)
    _ax.tick_params(labelsize=FONT_SIZE_TICK)
    _handles = [Line2D([], [], color=DATAPOINTS_COLORS['gray'], marker='o', linestyle='None', markersize=4, label='All possible pairs'), Line2D([], [], color=DATAPOINTS_COLORS['blue'], marker='o', linestyle='None', markersize=4, label='Selected (160)')]
    _ax.legend(handles=_handles, fontsize=FONT_SIZE_LEGEND, loc='upper left')
    _fig.tight_layout()
    _out = paths.FIGURES / 's00_arm_charge_scatter.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _fig
    _groups = [(assays, 'assays', 'Experimental assays'), (computed_props, 'computed', 'Computed properties'), (moe_props, 'moe', 'MOE properties')]
    _figs = []
    for _cols, _slug, _label in _groups:
        for _stat, _mode in [('avg', 'hex'), ('diff', 'contour')]:
            _plot_cols = [f'{c}_{_stat}' for c in _cols]
            _path = paths.FIGURES / f's00_strat1_{_stat}_{_slug}.png'
            _fig = make_coverage_pairplot(df_all_bsabs, df_strat1, _plot_cols, title=f'Strategy 1 — {_label} ({_stat})', mode=_mode, save_path=_path)
            _figs.append(_fig)
            print(f'wrote {_path.relative_to(paths.REPO_ROOT)}')
    _groups = [(assays, 'assays', 'Experimental assays'), (computed_props, 'computed', 'Computed properties'), (moe_props, 'moe', 'MOE properties')]
    _figs = []
    for _cols, _slug, _label in _groups:
        for _stat, _mode in [('avg', 'hex'), ('diff', 'contour')]:
            _plot_cols = [f'{c}_{_stat}' for c in _cols]
            _path = paths.FIGURES / f's00_strat2_{_stat}_{_slug}.png'
            _fig = make_coverage_pairplot(df_all_bsabs, df_strat2, _plot_cols, title=f'Strategy 2 — {_label} ({_stat})', mode=_mode, save_path=_path)
            _figs.append(_fig)
            print(f'wrote {_path.relative_to(paths.REPO_ROOT)}')
    _all_parents = sorted(set(df_exported['antibody_name-1'].tolist() + df_exported['antibody_name-2'].tolist()))
    _df_mab_sel = df_mabs[df_mabs['antibody_name'].isin(_all_parents)]
    _df_bsab_sel = pd.concat([df_strat1, df_strat2], ignore_index=True)
    _pretty = {'HIC': 'HIC', 'PR_CHO': 'PR CHO', 'AC-SINS_pH7.4': 'AC-SINS\npH 7.4', 'Tm2': 'Tm₂', 'pI': 'pI', 'hydrophobicity': 'Hydro-\nphobicity', 'aromaticity': 'Aromaticity', 'instability_index': 'Instability\nindex', 'length': 'Seq. length', 'patch_cdr_hyd': 'CDR hyd.\npatch', 'ens_charge': 'Ens. charge', 'dipole_moment': 'Dipole\nmoment', 'affinity_VL_VH': 'VL/VH\naffinity'}
    _ordered = ['HIC', 'PR_CHO', 'AC-SINS_pH7.4', 'Tm2', None, 'pI', 'hydrophobicity', 'aromaticity', 'instability_index', 'length', 'patch_cdr_hyd', 'ens_charge', 'dipole_moment', 'affinity_VL_VH', None]
    _nrows, _ncols = (3, 5)
    _w, _h, _cw = grid_figsize(_nrows, _ncols, cell_aspect=1.2)
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_w, _h), layout='constrained')
    _axes_flat = _axes.flatten()
    _blue = DATAPOINTS_COLORS['blue']
    _purple = DATAPOINTS_COLORS['purple']
    _gray = DATAPOINTS_COLORS['gray']
    _rng = np.random.default_rng(42)
    for _i, _col in enumerate(_ordered):
        _ax = _axes_flat[_i]
        if _col is None:
            _ax.set_visible(False)
            continue
        _mab_vals = df_mabs[_col].dropna().values
        _mab_sel_vals = _df_mab_sel[_col].dropna().values
        if len(_mab_vals) > 1:
            _vp = _ax.violinplot(_mab_vals, positions=[0], showmedians=True, showextrema=False, widths=0.65)
            for _b in _vp['bodies']:
                _b.set_facecolor(_blue)
                _b.set_alpha(0.15)
                _b.set_edgecolor(_blue)
                _b.set_linewidth(0.5)
            _vp['cmedians'].set_color(_blue)
            _vp['cmedians'].set_linewidth(0.8)
        if len(_mab_sel_vals) > 0:
            _j = _rng.uniform(-0.13, 0.13, len(_mab_sel_vals))
            _ax.scatter(_j, _mab_sel_vals, s=10, alpha=0.7, color=_blue, edgecolors=_gray, linewidths=0.3, zorder=3)
        _bsab_col = f'{_col}_avg'
        _bsab_vals = df_all_bsabs[_bsab_col].dropna().values
        _bsab_sel_vals = _df_bsab_sel[_bsab_col].dropna().values
        if len(_bsab_vals) > 1:
            _vp2 = _ax.violinplot(_bsab_vals, positions=[1], showmedians=True, showextrema=False, widths=0.65)
            for _b in _vp2['bodies']:
                _b.set_facecolor(_purple)
                _b.set_alpha(0.15)
                _b.set_edgecolor(_purple)
                _b.set_linewidth(0.5)
            _vp2['cmedians'].set_color(_purple)
            _vp2['cmedians'].set_linewidth(0.8)
        if len(_bsab_sel_vals) > 0:
            _j2 = _rng.uniform(0.87, 1.13, len(_bsab_sel_vals))
            _ax.scatter(_j2, _bsab_sel_vals, s=4, alpha=0.4, color=_purple, edgecolors=_gray, linewidths=0.2, zorder=3)
        _ax.set_title(_pretty.get(_col, _col), fontsize=FONT_SIZE_TITLE - 3, pad=3)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(['mAbs', 'Pairs'], fontsize=FONT_SIZE_TICK - 3)
        _ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK - 3, length=2)
        _ax.tick_params(axis='x', length=0)
    _leg_ax = _axes_flat[14]
    _leg_ax.set_visible(True)
    _leg_ax.axis('off')
    _handles = [Line2D([], [], color=_blue, marker='o', linestyle='None', markersize=4, markeredgecolor=_gray, markeredgewidth=0.3, label=f'mAbs ({len(_df_mab_sel)} / {len(df_mabs)})'), Line2D([], [], color=_purple, marker='o', linestyle='None', markersize=4, markeredgecolor=_gray, markeredgewidth=0.3, label=f'Pairs ({len(_df_bsab_sel)} / {len(df_all_bsabs):,})')]
    _leg_ax.legend(handles=_handles, loc='center', fontsize=FONT_SIZE_LEGEND, frameon=False)
    _out = paths.FIGURES / 's00_coverage_consolidated.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _fig
    _w, _h, _cw = grid_figsize(2, 2)
    _fig, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(2, 2, figsize=(_w, _h), layout='constrained')
    _blue = DATAPOINTS_COLORS['blue']
    _purple = DATAPOINTS_COLORS['purple']
    _red = DATAPOINTS_COLORS['red']
    _heavy = df_exported['mw_heavy_chain_diff'].dropna()
    _light = df_exported['mw_light_chain_diff'].dropna()
    _whole = df_exported['mw_whole_bsab'].dropna()
    _ax1.hist(_heavy, bins=30, alpha=0.7, edgecolor='white', color=_blue)
    _ax1.axvline(x=2, color=_red, linestyle='--', linewidth=1, label='2 Da')
    _ax1.set_xlabel('MW diff (Da)', fontsize=FONT_SIZE_LABEL)
    _ax1.set_ylabel('Freq', fontsize=FONT_SIZE_LABEL)
    _ax1.set_title(wrap_title('Heavy chain MW diffs', _cw), fontsize=FONT_SIZE_TITLE)
    _ax1.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND)
    _ax1.tick_params(labelsize=FONT_SIZE_TICK)
    _n_heavy = (_heavy <= 2).sum()
    _ax1.text(0.97, 0.55, f'{_n_heavy}/{len(_heavy)}\n≤2 Da', transform=_ax1.transAxes, ha='right', va='center', fontsize=FONT_SIZE_LEGEND, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    _ax2.hist(_light, bins=30, alpha=0.7, edgecolor='white', color=_purple)
    _ax2.axvline(x=2, color=_red, linestyle='--', linewidth=1, label='2 Da')
    _ax2.set_xlabel('MW diff (Da)', fontsize=FONT_SIZE_LABEL)
    _ax2.set_ylabel('Freq', fontsize=FONT_SIZE_LABEL)
    _ax2.set_title(wrap_title('Light chain MW diffs', _cw), fontsize=FONT_SIZE_TITLE)
    _ax2.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND)
    _ax2.tick_params(labelsize=FONT_SIZE_TICK)
    _n_light = (_light <= 2).sum()
    _ax2.text(0.97, 0.55, f'{_n_light}/{len(_light)}\n≤2 Da', transform=_ax2.transAxes, ha='right', va='center', fontsize=FONT_SIZE_LEGEND, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    _ax3.hist(_whole, bins=30, alpha=0.7, edgecolor='white', color=_blue)
    _ax3.set_xlabel('Whole bsAb MW (Da)', fontsize=FONT_SIZE_LABEL)
    _ax3.set_ylabel('Freq', fontsize=FONT_SIZE_LABEL)
    _ax3.set_title(wrap_title('Whole bsAb MWs', _cw), fontsize=FONT_SIZE_TITLE)
    _ax3.tick_params(labelsize=FONT_SIZE_TICK)
    _ax4.hist(_heavy, bins=30, alpha=0.5, edgecolor='white', color=_blue, label='Heavy')
    _ax4.hist(_light, bins=30, alpha=0.5, edgecolor='white', color=_purple, label='Light')
    _ax4.axvline(x=2, color=_red, linestyle='--', linewidth=1, label='2 Da')
    _ax4.set_xlabel('MW diff (Da)', fontsize=FONT_SIZE_LABEL)
    _ax4.set_ylabel('Freq', fontsize=FONT_SIZE_LABEL)
    _ax4.set_title(wrap_title('Heavy vs light diffs', _cw), fontsize=FONT_SIZE_TITLE)
    _ax4.legend(loc='upper right', fontsize=FONT_SIZE_LEGEND)
    _ax4.tick_params(labelsize=FONT_SIZE_TICK)
    _n_both = ((_heavy <= 2) & (_light <= 2)).sum()
    print(f'Heavy ≤2 Da: {_n_heavy}/{len(_heavy)}')
    print(f'Light ≤2 Da: {_n_light}/{len(_light)}')
    print(f'Both ≤2 Da: {_n_both}/{min(len(_heavy), len(_light))}')
    print(f'Whole bsAb MW: {_whole.min():.0f}–{_whole.max():.0f} Da')
    _out = paths.FIGURES / 's00_mw_differences.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _fig
    _all_arms = pd.concat([df_exported['antibody_name-1'], df_exported['antibody_name-2']])
    _counts = _all_arms.value_counts().sort_values(ascending=False)
    _n = len(_counts)
    _half = math.ceil(_n / 2)
    _names = _counts.index.tolist()
    _vals = _counts.values.tolist()
    _left_names = _names[:_half]
    _left_vals = _vals[:_half]
    _right_names = _names[_half:]
    _right_vals = _vals[_half:]
    while len(_right_names) < _half:
        _right_names.append('')
        _right_vals.append('')
    _cell_text = []
    for _i in range(_half):
        _cell_text.append([_left_names[_i], str(_left_vals[_i]), _right_names[_i], str(_right_vals[_i]) if _right_vals[_i] != '' else ''])
    _row_h = 0.18
    _fig_h = _half * _row_h + 0.8
    _fig, _ax = plt.subplots(figsize=(SINGLE_COL_WIDTH + 1.5, _fig_h))
    _ax.axis('off')
    _fig.suptitle(f'Parent mAb usage across 160 bsAbs (n = {_n})', fontsize=FONT_SIZE_TITLE, y=1.02)
    _col_widths = [0.35, 0.12, 0.35, 0.12]
    _tbl = _ax.table(cellText=_cell_text, colLabels=['Parent mAb', 'Count', 'Parent mAb', 'Count'], colWidths=_col_widths, loc='center', cellLoc='left')
    _tbl.auto_set_font_size(False)
    _tbl.set_fontsize(FONT_SIZE_TICK - 2)
    _tbl.scale(1, 1.15)
    for (_r, _c), _cell in _tbl.get_celld().items():
        _cell.set_edgecolor('#cccccc')
        _cell.set_linewidth(0.5)
        if _r == 0:
            _cell.set_facecolor(DATAPOINTS_COLORS['blue'])
            _cell.set_text_props(color='white', fontweight='bold')
        else:
            _cell.set_facecolor('white')
        if _c in (1, 3):
            _cell.set_text_props(ha='center')
    _out = paths.FIGURES / 's00_arm_distribution.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    print(f'Total unique parents: {_n}')
    _fig
    _coords = pd.read_csv(paths.RAW_UMAP_COORDS)
    _bg = _coords[_coords['category'] == 'All possible pairs']
    _sel = _coords[_coords['category'] == 'Selected (160)']
    umap_x = _bg['umap_1'].to_numpy()
    umap_y = _bg['umap_2'].to_numpy()
    sel_umap_x = _sel['umap_1'].to_numpy()
    sel_umap_y = _sel['umap_2'].to_numpy()
    print(f'UMAP (frozen): {len(umap_x)} background pairs, {len(sel_umap_x)} selected')
    _valid = np.isfinite(umap_x)
    _fig, _ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
    _ax.scatter(umap_x[_valid], umap_y[_valid], s=2, alpha=0.15, color=DATAPOINTS_COLORS['gray'], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3, rasterized=True)
    _ax.set_xlabel('UMAP 1', fontsize=FONT_SIZE_LABEL)
    _ax.set_ylabel('UMAP 2', fontsize=FONT_SIZE_LABEL)
    _ax.set_title(wrap_title('Bispecific design space (all possible pairs)', SINGLE_COL_WIDTH), fontsize=FONT_SIZE_TITLE)
    _ax.tick_params(labelsize=FONT_SIZE_TICK)
    _fig.tight_layout()
    _out = paths.FIGURES / 's00_umap_design_space.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _fig
    _bg = np.isfinite(umap_x)
    _sel = np.isfinite(sel_umap_x)
    _fig, _ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH))
    _xb, _yb = (umap_x[_bg], umap_y[_bg])
    _pad = 0.5
    _xmin, _xmax = (_xb.min() - _pad, _xb.max() + _pad)
    _ymin, _ymax = (_yb.min() - _pad, _yb.max() + _pad)
    _H, _xe, _ye = np.histogram2d(_xb, _yb, bins=120, range=[[_xmin, _xmax], [_ymin, _ymax]])
    _H = gaussian_filter(_H, sigma=2.0)
    _xc = 0.5 * (_xe[:-1] + _xe[1:])
    _yc = 0.5 * (_ye[:-1] + _ye[1:])
    _levels = [np.percentile(_H[_H > 0], 25), np.percentile(_H[_H > 0], 65), _H.max()]
    _greys = ['#E0E0E0', '#C0C0C0']
    _ax.contourf(_xc, _yc, _H.T, levels=_levels, colors=_greys)
    _ax.contour(_xc, _yc, _H.T, levels=_levels, colors=['#AAAAAA'], linewidths=0.4)
    _ax.scatter(sel_umap_x[_sel], sel_umap_y[_sel], s=18, alpha=0.8, color=DATAPOINTS_COLORS['blue'], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3, zorder=3)
    _density_handles = [Line2D([], [], color='#BBBBBB', marker='s', linestyle='None', markersize=5, markeredgecolor='none', label='All possible pairs'), Line2D([], [], color=DATAPOINTS_COLORS['blue'], marker='o', linestyle='None', markersize=5, label='Selected (160)')]
    _ax.legend(handles=_density_handles, fontsize=FONT_SIZE_LEGEND, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    _ax.set_xlabel('UMAP 1', fontsize=FONT_SIZE_LABEL)
    _ax.set_ylabel('UMAP 2', fontsize=FONT_SIZE_LABEL)
    _ax.set_title(wrap_title('Selected bispecifics in design space', SINGLE_COL_WIDTH), fontsize=FONT_SIZE_TITLE)
    _ax.tick_params(labelsize=FONT_SIZE_TICK)
    _fig.tight_layout()
    _out = paths.FIGURES / 's00_umap_selected.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _fig
    _src = paths.FIGURES / 's00_coverage_consolidated.png'
    _dst = paths.FIGURES / 'supplementary/figure_s11.png'
    _dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(_src, _dst)
    print(f'wrote {_dst.relative_to(paths.REPO_ROOT)}')
    emit('supp_11')
if __name__ == '__main__':
    main()
