"""Read-only isolation of the historical IJF grid-model sign mismatch."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate import sha

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT/'submission_IJF'
OUT = PROJECT/'artifacts/ijf_sign_audit_20260910'


def main():
    OUT.mkdir(exist_ok=True)
    table_path = ARCHIVE/'Quantlets/CO_full_evaluation/results/all_results.csv'
    table = pd.read_csv(table_path)
    table = table[table.alpha == .01]
    rows = []; inputs = {str(table_path.relative_to(PROJECT)): sha(table_path)}
    for model, directory in [('TimesFM-2.5', 'timesfm25'), ('Moirai-2.0', 'moirai2')]:
        for _, stored in table[table.model == model].iterrows():
            path = ARCHIVE/f'cfp_ijf_data/{directory}/{stored.symbol}.parquet'
            returns_path = ARCHIVE/f'cfp_ijf_data/returns/{stored.symbol}.csv'
            forecast = pd.read_parquet(path)['VaR_0.01']
            returns = pd.read_csv(returns_path, index_col='date', parse_dates=True).log_return
            merged = pd.concat([returns.rename('r'), forecast.rename('stored_var')], axis=1).dropna()
            n_cal = int(.7*len(merged)); test = merged.iloc[n_cal:]
            assert n_cal == stored.n_cal and len(test) == stored.n_test
            wrong = int((test.r < test.stored_var).sum())
            correct = int((test.r < -test.stored_var).sum())
            assert wrong == stored.viol_raw
            assert abs(wrong/len(test)-stored.pihat_raw) < 1e-14
            current = pd.read_parquet(PROJECT/f'cfp_ijf_data/{directory}/{stored.symbol}.parquet')['VaR_0.01']
            pd.testing.assert_index_equal(current.index, forecast.index)
            exact_negative = np.array_equal(current.to_numpy(), -forecast.to_numpy(), equal_nan=True)
            finite = np.isfinite(current.to_numpy()) & np.isfinite(forecast.to_numpy())
            max_sum = float(np.max(np.abs(current.to_numpy()[finite]+forecast.to_numpy()[finite])))
            rows.append({'model': model, 'asset': stored.symbol, 'n_cal': n_cal, 'n_test': len(test),
                'test_start': str(test.index[0].date()), 'test_end': str(test.index[-1].date()),
                'published_violations': int(stored.viol_raw), 'reproduced_old_violations': wrong,
                'correct_sign_violations': correct, 'old_rate': wrong/len(test), 'correct_sign_rate': correct/len(test),
                'current_stored_q_is_exact_negative_archived_var': bool(exact_negative),
                'current_plus_archived_max_absolute': max_sum})
            for p in (path, returns_path, PROJECT/f'cfp_ijf_data/{directory}/{stored.symbol}.parquet'):
                inputs[str(p.relative_to(PROJECT))] = sha(p)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'per_pair.csv', index=False)
    summary = frame.groupby('model').agg(pairs=('asset','size'), old_rate=('old_rate','mean'),
        correct_sign_rate=('correct_sign_rate','mean'), old_violations=('reproduced_old_violations','sum'),
        correct_sign_violations=('correct_sign_violations','sum'), test_observations=('n_test','sum'))
    summary.to_csv(OUT/'summary.csv')
    for name in ['pipeline/CFP_TimesFM_Forecasts.ipynb','pipeline/CFP_Moirai_Forecasts.ipynb',
                 'Quantlets/CO_full_evaluation/CO_full_evaluation.ipynb','Recalibrating_Tail_Risk_Forecasts.tex']:
        p = ARCHIVE/name; inputs[str(p.relative_to(PROJECT))] = sha(p)
    receipt = {'status': 'passed', 'pairs': len(frame), 'original_counts_exact': True,
        'only_counterfactual_change': 'Compare return to negative stored positive-loss VaR',
        'old_artifacts_modified': False, 'producer_sha256': sha(__file__), 'inputs': inputs,
        'outputs': {p.name: sha(p) for p in OUT.glob('*.csv')}}
    (OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(summary.to_string())


if __name__ == '__main__': main()
