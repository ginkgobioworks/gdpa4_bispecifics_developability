import pandas as pd
import numpy as np
from scipy import stats
from prophet_ab import paths, schema
from prophet_ab.features import compositional
from prophet_ab import normalize as nz

def main():
    DEPRECATED = schema.DEPRECATED_VALUE_COLS
    summaries = pd.read_parquet(paths.S03 / 'gdpa4_per_antibody.parquet')
    components = pd.read_parquet(paths.S02 / 'bispecific_components.parquet')
    monospecific = summaries[summaries['kind'] == schema.KIND_MONOSPECIFIC][['antibody_name', 'value_col', 'condition', 'median']]
    monospecific = monospecific.assign(parent=monospecific['antibody_name'].map(nz.strip_isotype_suffix))
    monospecific_lookup = monospecific[['parent', 'value_col', 'condition', 'median']]
    bispecific = summaries[summaries['kind'] == schema.KIND_BISPECIFIC][['antibody_name', 'value_col', 'condition', 'median']].rename(columns={'median': 'bispecific_median'})
    paired = bispecific.merge(components[['antibody_name', 'parent_a', 'parent_b']], on='antibody_name')
    paired = paired.merge(monospecific_lookup.rename(columns={'parent': 'parent_a', 'median': 'parent_a_median'}), on=['parent_a', 'value_col', 'condition'], how='left').merge(monospecific_lookup.rename(columns={'parent': 'parent_b', 'median': 'parent_b_median'}), on=['parent_b', 'value_col', 'condition'], how='left')
    pred_df = compositional.apply_all(paired['parent_a_median'], paired['parent_b_median'])
    out = pd.concat([paired, pred_df.add_prefix('pred_')], axis=1)
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
    _o = paths.TABLES / 's03_baseline_metrics.csv'
    _o.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(_o, index=False)
    print(f'wrote {_o.relative_to(paths.REPO_ROOT)}  rows={len(metrics)}')

if __name__ == '__main__':
    main()
