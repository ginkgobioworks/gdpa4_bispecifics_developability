"""Generate supplementary Figure S04.

This is a standalone, non-interactive script. It reads the canonical processed
data and committed model caches, writes the final-numbered PNG under
``reports/figures/supplementary/``, and emits the panel source-data CSVs.

Example
-------
``uv run python figures/supplementary/figure_s04.py``
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve, roc_auc_score
from prophet_ab import paths, schema
from prophet_ab import normalize as nz
from datapoints_figures import set_manuscript_style, DATAPOINTS_COLORS, FULL_WIDTH, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_LEGEND_TITLE, FONT_SIZE_TICK, FONT_SIZE_TITLE, equalize_axes, wrap_label, wrap_title

def main():
    set_manuscript_style()
    _summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    _components = pd.read_parquet(paths.S02 / 'bispecific_components.parquet')
    _metrics = [('hihplc_normretentiontime', 'default', 'hic'), ('hachplc_retentiontime', 'default', 'hac'), ('pr_score', 'Ovalbumin', 'pr_ova'), ('pr_score', 'CHO', 'pr_cho'), ('bvp_score_norm', 'default', 'bvp')]

    def _wide(kind):
        _d = _summaries[_summaries['kind'] == kind]
        _out = None
        for _vc, _cond, _alias in _metrics:
            _sub = _d[(_d['value_col'] == _vc) & (_d['condition'] == _cond)][['antibody_name', 'median']].rename(columns={'median': _alias})
            _out = _sub if _out is None else _out.merge(_sub, on='antibody_name', how='outer')
        return _out
    wide_monospecific = _wide(schema.KIND_MONOSPECIFIC)
    wide_monospecific['parent'] = wide_monospecific['antibody_name'].map(nz.strip_isotype_suffix)
    wide_bispecific = _wide(schema.KIND_BISPECIFIC)
    components = _components[['antibody_name', 'parent_a', 'parent_b']]
    print(f'monospecific: {len(wide_monospecific)}  bispecific: {len(wide_bispecific)}  components: {len(components)}')
    _core = wide_monospecific.dropna(subset=['hic', 'hac']).copy().reset_index(drop=True)
    _pts = _core[['hic', 'hac']].values
    _n = len(_pts)
    _is_pareto = np.ones(_n, dtype=bool)
    for _i in range(_n):
        if not _is_pareto[_i]:
            continue
        for _j in range(_n):
            if _i == _j:
                continue
            if _pts[_j, 0] >= _pts[_i, 0] and _pts[_j, 1] >= _pts[_i, 1] and (_pts[_j, 0] > _pts[_i, 0] or _pts[_j, 1] > _pts[_i, 1]):
                _is_pareto[_i] = False
                break
    _core['on_frontier'] = _is_pareto
    _fpts = _core.loc[_is_pareto, ['hic', 'hac']].values
    frontier_pts = _fpts[np.argsort(_fpts[:, 0])]
    frontier_parent_set = set(_core.loc[_is_pareto, 'parent'])
    print(f'frontier mAbs: {int(_is_pareto.sum())}')
    _bispecific = wide_bispecific.merge(components, on='antibody_name', how='left')
    _bispecific = _bispecific.dropna(subset=['hic', 'hac', 'parent_a', 'parent_b']).copy()
    _bispecific['pair_key'] = _bispecific.apply(lambda r: tuple(sorted([r['parent_a'], r['parent_b']])), axis=1)
    crosser_df = _bispecific.groupby('pair_key', as_index=False).agg(parent_a=('parent_a', 'first'), parent_b=('parent_b', 'first'), antibody_name=('antibody_name', 'first'), hic=('hic', 'mean'), hac=('hac', 'mean'), pr_ova=('pr_ova', 'mean'), pr_cho=('pr_cho', 'mean'), bvp=('bvp', 'mean'))
    _fx, _fy = (frontier_pts[:, 0], frontier_pts[:, 1])
    crosser_df['frontier_dist'] = crosser_df['hac'] - np.interp(crosser_df['hic'], _fx, _fy, left=_fy[0], right=_fy[-1])
    crosser_df['crossed'] = crosser_df['frontier_dist'] > 0

    def _categorize(row):
        if not row['crossed']:
            return 'Within'
        _a = row['parent_a'] in frontier_parent_set
        _b = row['parent_b'] in frontier_parent_set
        if _a and _b:
            return 'Additive'
        if _a or _b:
            return 'Partial'
        return 'Emergent'
    crosser_df['category'] = crosser_df.apply(_categorize, axis=1)
    _counts = crosser_df['category'].value_counts()
    print(f'unique pairs: {len(crosser_df)}  crossers: {int(crosser_df['crossed'].sum())}')
    print(_counts.to_dict())
    PR_ASSAYS = [('pr_ova', 'PR-OVA'), ('pr_cho', 'PR-CHO'), ('bvp', 'PR-BVP')]

    def _short(name):
        return nz.strip_isotype_suffix(str(name))
    roc_data = {}
    _auc_rows = []
    for _col, _label in PR_ASSAYS:
        _valid = crosser_df[['frontier_dist', _col]].dropna()
        _x = _valid['frontier_dist'].values
        _y = _valid[_col].values
        _poly = (_y > np.percentile(_y, 80)).astype(int)
        _fpr, _tpr, _ = roc_curve(_poly, _x)
        _auc = roc_auc_score(_poly, _x)
        _rng = np.random.default_rng(0)
        _boot = []
        for _ in range(2000):
            _idx = _rng.integers(0, len(_poly), len(_poly))
            if len(np.unique(_poly[_idx])) < 2:
                continue
            _boot.append(roc_auc_score(_poly[_idx], _x[_idx]))
        _lo, _hi = np.percentile(_boot, [2.5, 97.5])
        roc_data[_label] = dict(fpr=_fpr, tpr=_tpr, auc=_auc, lo=_lo, hi=_hi, n=len(_valid))
        _auc_rows.append(dict(assay=_col, label=_label, n=len(_valid), auc=_auc, ci_lo_95=_lo, ci_hi_95=_hi))
    auc_table = pd.DataFrame(_auc_rows)
    top_crossers = crosser_df[crosser_df['crossed']].nlargest(5, 'pr_cho').reset_index(drop=True)
    top_crossers['pair_label'] = top_crossers.apply(lambda r: f'{_short(r['parent_a'])} x {_short(r['parent_b'])}', axis=1)
    top_keys = set(zip(top_crossers['parent_a'], top_crossers['parent_b']))
    print(auc_table.to_string(index=False))
    _valid = crosser_df[['frontier_dist', 'pr_cho']].dropna()
    _x = _valid['frontier_dist'].values
    _y = _valid['pr_cho'].values
    _pred = (_x > 0).astype(int)
    _true = (_y > np.percentile(_y, 80)).astype(int)
    _tp = int(((_pred == 1) & (_true == 1)).sum())
    _fp = int(((_pred == 1) & (_true == 0)).sum())
    _fn = int(((_pred == 0) & (_true == 1)).sum())
    _tn = int(((_pred == 0) & (_true == 0)).sum())
    _prec = _tp / (_tp + _fp) if _tp + _fp else 0.0
    _rec = _tp / (_tp + _fn) if _tp + _fn else 0.0
    _spec = _tn / (_tn + _fp) if _tn + _fp else 0.0
    _f1 = 2 * _prec * _rec / (_prec + _rec) if _prec + _rec else 0.0
    _or, _p = fisher_exact([[_tp, _fn], [_fp, _tn]], alternative='greater')
    confusion = dict(mat=np.array([[_tp, _fp], [_fn, _tn]]), precision=_prec, recall=_rec, specificity=_spec, f1=_f1, odds_ratio=_or, fisher_p=_p)
    print(f'PR-CHO CM rows[crosser,non] cols[poly,non]: {confusion['mat'].tolist()}  prec={_prec:.2f} rec={_rec:.2f} spec={_spec:.2f} F1={_f1:.2f} OR={_or:.1f} p={_p:.4f}')
    _CAT_COLORS = {'Within': DATAPOINTS_COLORS['gray'], 'Additive': DATAPOINTS_COLORS['red'], 'Partial': DATAPOINTS_COLORS['amber'], 'Emergent': DATAPOINTS_COLORS['purple']}
    _CAT_LABELS = {'Within': 'within hull', 'Additive': 'additive (both arms on frontier)', 'Partial': 'partial (one arm on frontier)', 'Emergent': 'emergent (neither arm on frontier)'}
    _PR_COLORS = {'PR-OVA': DATAPOINTS_COLORS['blue'], 'PR-CHO': DATAPOINTS_COLORS['coral'], 'PR-BVP': DATAPOINTS_COLORS['amber']}
    _TOP_MARKERS = ['H', '^', 's', 'D', 'p']
    fig = plt.figure(figsize=(FULL_WIDTH, 7.4), layout='constrained')
    _outer = GridSpec(2, 1, figure=fig, height_ratios=[1.12, 1.0])
    _gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=_outer[0], width_ratios=[3.0, 1.35], wspace=0.04)
    _gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=_outer[1], width_ratios=[1.0, 1.0], wspace=0.28)
    _axA = fig.add_subplot(_gs_top[0, 0])
    _axLeg = fig.add_subplot(_gs_top[0, 1])
    _axLeg.axis('off')
    _axB = fig.add_subplot(_gs_bot[0, 0])
    _axC = fig.add_subplot(_gs_bot[0, 1])
    _axA.plot(frontier_pts[:, 0], frontier_pts[:, 1], ls='--', color=DATAPOINTS_COLORS['coral'], lw=1.4, zorder=3)
    _axA.scatter(frontier_pts[:, 0], frontier_pts[:, 1], marker='D', s=42, facecolor=DATAPOINTS_COLORS['pink'], edgecolor=DATAPOINTS_COLORS['purple'], linewidths=0.9, zorder=4)
    for _cat in ['Within', 'Additive', 'Partial', 'Emergent']:
        _sub = crosser_df[crosser_df['category'] == _cat]
        if _sub.empty:
            continue
        _is_top = _sub.apply(lambda r: (r['parent_a'], r['parent_b']) in top_keys, axis=1)
        _rest = _sub[~_is_top]
        _axA.scatter(_rest['hic'], _rest['hac'], s=22 if _cat == 'Within' else 46, c=_CAT_COLORS[_cat], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.3, alpha=0.55 if _cat == 'Within' else 0.9, zorder=2 if _cat == 'Within' else 5)
    for _i, (_, _r) in enumerate(top_crossers.iterrows()):
        _axA.scatter(_r['hic'], _r['hac'], marker=_TOP_MARKERS[_i], s=120, c=_CAT_COLORS[_r['category']], edgecolors=DATAPOINTS_COLORS['gray'], linewidths=0.8, alpha=0.95, zorder=8)
    _axA.set_xlabel(wrap_label('HIC RT (functional hydrophobicity)', 5.0), fontsize=FONT_SIZE_LABEL)
    _axA.set_ylabel(wrap_label('HAC RT (functional charge)', 4.0), fontsize=FONT_SIZE_LABEL)
    _axA.set_title(wrap_title('bsAbs crossing the mAb biophysical ceiling, by parent composition', 5.0), fontsize=FONT_SIZE_TITLE, fontweight='semibold')
    _axA.tick_params(labelsize=FONT_SIZE_TICK)
    _cat_handles = [Line2D([], [], color=DATAPOINTS_COLORS['coral'], ls='--', lw=1.4, label='mAb ceiling (Pareto frontier)'), Line2D([], [], marker='D', lw=0, markersize=6, markerfacecolor=DATAPOINTS_COLORS['pink'], markeredgecolor=DATAPOINTS_COLORS['purple'], label='frontier mAbs')]
    for _cat in ['Within', 'Additive', 'Partial', 'Emergent']:
        _cat_handles.append(Line2D([], [], marker='o', lw=0, markersize=6, markerfacecolor=_CAT_COLORS[_cat], markeredgecolor=DATAPOINTS_COLORS['gray'], label=_CAT_LABELS[_cat]))
    _leg1 = _axLeg.legend(handles=_cat_handles, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FONT_SIZE_LEGEND, title='Categories', title_fontsize=FONT_SIZE_LEGEND_TITLE, framealpha=0.95, borderpad=0.6, handletextpad=0.5)
    _leg1.get_title().set_fontweight('semibold')
    _axLeg.add_artist(_leg1)
    _top_handles = [Line2D([], [], marker=_TOP_MARKERS[_i], lw=0, markersize=9, markerfacecolor=_CAT_COLORS[_r['category']], markeredgecolor=DATAPOINTS_COLORS['gray'], markeredgewidth=0.8, label=f'{_r['pair_label']} (PR-CHO {_r['pr_cho']:.2f})') for _i, (_, _r) in enumerate(top_crossers.iterrows())]
    _leg2 = _axLeg.legend(handles=_top_handles, loc='upper left', bbox_to_anchor=(0.0, 0.4), fontsize=FONT_SIZE_LEGEND - 0.5, title='Top 5 crossers by PR-CHO', title_fontsize=FONT_SIZE_LEGEND_TITLE, framealpha=0.95, borderpad=0.6, handletextpad=0.5, handlelength=1.2)
    _leg2.get_title().set_fontweight('semibold')
    for _col, _label in PR_ASSAYS:
        _d = roc_data[_label]
        _axB.plot(_d['fpr'], _d['tpr'], lw=2.0, color=_PR_COLORS[_label], label=f'{_label}: AUC {_d['auc']:.2f}')
    _axB.plot([0, 1], [0, 1], ls='--', color=DATAPOINTS_COLORS['gray'], lw=0.8, alpha=0.6)
    _axB.set_xlim(-0.02, 1.02)
    _axB.set_ylim(-0.02, 1.02)
    equalize_axes(_axB)
    _axB.set_box_aspect(1)
    _axB.set_xlabel(wrap_label('False positive rate (1 - specificity)', 3.2), fontsize=FONT_SIZE_LABEL)
    _axB.set_ylabel(wrap_label('True positive rate (recall)', 3.2), fontsize=FONT_SIZE_LABEL)
    _axB.set_title(wrap_title('ROC: frontier distance to polyreactive (top 20%)', 3.2), fontsize=FONT_SIZE_TITLE, fontweight='semibold')
    _axB.tick_params(labelsize=FONT_SIZE_TICK)
    _axB.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND - 0.5, framealpha=0.95)
    _mat = confusion['mat']
    _im = _axC.imshow(_mat, cmap='Purples', aspect='auto', vmin=0, vmax=_mat.max())
    for _i in range(2):
        for _j in range(2):
            _v = int(_mat[_i, _j])
            _axC.text(_j, _i, str(_v), ha='center', va='center', fontsize=20, fontweight='bold', color='white' if _v > _mat.max() * 0.55 else 'black')
    _axC.set_xticks([0, 1])
    _axC.set_yticks([0, 1])
    _axC.set_xticklabels(['polyreactive', 'non-polyreactive'], fontsize=FONT_SIZE_TICK)
    _axC.set_yticklabels(['crosser', 'non-crosser'], fontsize=FONT_SIZE_TICK, rotation=90, va='center')
    _axC.tick_params(length=0)
    _axC.set_title(wrap_title(f'PR-CHO at top-20% threshold (predictor: frontier_dist > 0)\nprec {confusion['precision']:.2f}, rec {confusion['recall']:.2f}, spec {confusion['specificity']:.2f}, F1 {confusion['f1']:.2f}; OR {confusion['odds_ratio']:.1f}, Fisher p {confusion['fisher_p']:.4f}', 3.2), fontsize=FONT_SIZE_TITLE, fontweight='semibold')
    _axA.text(-0.1, 1.04, 'A', transform=_axA.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
    _axB.text(-0.22, 1.04, 'B', transform=_axB.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
    _axC.text(-0.18, 1.04, 'C', transform=_axC.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
    _out = paths.FIGURES / 'supplementary/figure_s04.png'
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=300, bbox_inches='tight')
    print(f'wrote {_out.relative_to(paths.REPO_ROOT)}')
    _t1 = paths.TABLES / 's13_ceiling_crossers.csv'
    _t1.parent.mkdir(parents=True, exist_ok=True)
    crosser_df.to_csv(_t1, index=False)
    print(f'wrote {_t1.relative_to(paths.REPO_ROOT)}  rows={len(crosser_df)}')
    _t2 = paths.TABLES / 's13_ceiling_roc_auc.csv'
    auc_table.to_csv(_t2, index=False)
    print(f'wrote {_t2.relative_to(paths.REPO_ROOT)}')
    emit('supp_04')
if __name__ == '__main__':
    main()
