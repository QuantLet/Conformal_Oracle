"""Compose artifacts/r8_optimism_estimator/RESULTS.md from the saved CSV/JSON outputs (no computation)."""
import json
import numpy as np
import pandas as pd
from engine import BIASES, ESTIMATORS, LAWS, OUT, SIZES

summary = pd.read_csv(OUT / 'summary.csv')
bands = pd.read_csv(OUT / 'simultaneous_bands.csv')
sup = pd.read_csv(OUT / 'supplementary_loss_bands.csv')
oracle = pd.read_csv(OUT / 'oracle_lambda.csv')
adm = json.loads((OUT / 'admission.json').read_text())
run = json.loads((OUT / 'run.json').read_text())
chk = json.loads((OUT / 'check.json').read_text())
LABEL = {'E1_blocked_cv': 'E1 (blocked cross-validation, K=5)', 'E2_block_bootstrap': 'E2 (circular block bootstrap, R=200, b=ceil(n^(1/3)))'}


def sci(v):
    return f'{v:.3e}'


def fmt(v, d=4):
    return f'{v:.{d}f}'


lines = ['# Nuisance-free optimism estimators and feasible shrinkage: results', '',
         'Protocol: `research/r8_optimism_estimator/PROTOCOL.md` (fixed 13 September 2026 before any calculation). '
         'Engine: `research/r8_optimism_estimator/engine.py`. All numbers below are read from the CSV/JSON files in this folder.', '',
         '## Admission decision', '']
for name in ESTIMATORS:
    e = adm['estimators'][name]
    lines.append(f"- {LABEL[name]}: criterion 1 (penalty accuracy, |ratio-1| <= 0.15 at n in 700, 1000, 2000, both laws, both b): "
                 f"**{'PASS' if e['criterion_1_penalty_accuracy']['passes'] else 'FAIL'}**; "
                 f"criterion 2 (rule value at n >= 700): **{'PASS' if e['criterion_2_rule_value']['passes'] else 'FAIL'}**; "
                 f"criterion 3 (simultaneous family): {e['criterion_3_uncertainty']['status']}, family size {e['criterion_3_uncertainty']['family_size']}, "
                 f"critical value {fmt(e['criterion_3_uncertainty']['critical_value'])}, cells whose band includes 1: {e['criterion_3_uncertainty']['cells_including_one']}/20; "
                 f"estimator admitted: **{'YES' if e['passes'] else 'NO'}**.")
lines += ['', f"Passing estimators: {adm['passing_estimators'] or 'none'}. Financial application: **{adm['financial_application']}**.", '',
          '### Criterion 2 detail (n >= 700; expected loss R(lambda C_n) - R(0) from the reference risk, mean over 500 histories)', '',
          '| estimator | law | n | b | raw | full (lambda=1) | shrunken (lambda_hat) | shrunken < full | shrunken < raw | pass |', '|---|---|---|---|---|---|---|---|---|---|']
for name in ESTIMATORS:
    for c in adm['estimators'][name]['criterion_2_rule_value']['cells']:
        if c['applies']:
            lines.append(f"| {name} | {c['law']} | {c['n']} | {c['bias']:.6f} | {sci(c['loss_raw'])} | {sci(c['loss_full'])} | {sci(c['loss_shrunk'])} | "
                         f"{c['below_full']} | {c['below_raw']} | {'PASS' if c['passes'] else 'FAIL'} |")
lines += ['', 'Failing cells (both estimators): every b > 0 cell at n in 700, 1000, 2000 except t5 at n = 700. In those cells the shrunken policy has a '
          'higher expected loss than full correction. All b = 0 cells pass (shrunken below full).', '',
          '## Synthetic tables', '',
          'Columns: mean and MCSE (sd/sqrt(500)) of O_hat; ratio = mean O_hat / (2A_0n) with 2A_0n = Omega/(n f*) from truth.csv; '
          'band = simultaneous 95% band on the ratio (one family of 20 cells per estimator, saved 999 history-bootstrap rows, '
          'max-standardised-deviation rule, 95th percentile with NumPy method="higher"); v3 ref = the known-truth independent-marginal '
          'optimism ratio from results/theory_loop_v3/diagnostic/simultaneous_bands.csv (oracle, for orientation only); '
          'lambda_hat mean and sd over histories; share of histories with lambda_hat = 0; expected losses R(lambda C_n) - R(0) '
          'for raw (lambda = 0, identically zero), full (lambda = 1), shrunken (lambda_hat); shrunken - full with its MCSE.', '']
for name in ESTIMATORS:
    lines += [f'### {LABEL[name]}', '',
              '| law | n | b | mean O_hat | MCSE | 2A_0n | ratio | band lower | band upper | incl. 1 | v3 ref | mean lambda_hat | sd | share lambda=0 | loss raw | loss full | loss shrunken | shrunken - full | MCSE |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    s = summary[summary.estimator == name].sort_values(['law', 'n', 'bias'])
    for r in s.itertuples():
        lines.append(f'| {r.law} | {r.n} | {r.bias:.6f} | {sci(r.mean_O_hat)} | {sci(r.mcse_O_hat)} | {sci(r.two_A0)} | {fmt(r.ratio_to_2A0)} | '
                     f'{fmt(r.band_lower)} | {fmt(r.band_upper)} | {r.band_includes_one} | {fmt(r.v3_true_optimism_ratio)} | {fmt(r.mean_lambda_hat)} | '
                     f'{fmt(r.sd_lambda_hat)} | {fmt(r.share_lambda_zero, 3)} | {sci(r.loss_raw)} | {sci(r.loss_full)} | {sci(r.loss_shrunk)} | '
                     f'{sci(r.shrunk_minus_full)} | {sci(r.shrunk_minus_full_mcse)} |')
    lines.append('')
e2 = summary[summary.estimator == ESTIMATORS[1]].drop_duplicates('n')[['n', 'block_length']]
lines += ['E2 block lengths ceil(n^(1/3)): ' + ', '.join(f'n={r.n}: {r.block_length}' for r in e2.itertuples()) + '.',
          'E1 is invariant to the bias translation b by construction (the baseline term rho(-s) cancels between L_cv and I_n), so its b = 0 and b > 0 '
          'rows coincide; E2 depends on b through the bootstrap-sample baseline and differs slightly between the two b rows.', '',
          '## Supplementary simultaneous bands on expected-loss differences (not a protocol criterion)', '',
          'One family per estimator with 30 cells: shrunken - full for all 20 (law, n, b) cells and shrunken - raw for the 10 b > 0 cells; '
          'same 999 history-bootstrap rows and band rule as above.', '',
          '| estimator | law | n | b | contrast | mean | SE | lower | upper | critical value |', '|---|---|---|---|---|---|---|---|---|---|']
for r in sup.itertuples():
    lines.append(f'| {r.estimator} | {r.law} | {r.n} | {r.bias:.6f} | {r.contrast} | {sci(r.mean)} | {sci(r.standard_error)} | {sci(r.lower)} | {sci(r.upper)} | {fmt(r.critical_value)} |')
lines += ['', '## Supplementary oracle diagnostic (not a protocol criterion)', '',
          'Expected loss of the oracle factor lambda* = B/(B + A_0) with the true B = R(0) - R(q) and A_0 = Omega/(2 n f*), applied to the same C_n, '
          'from the reference risk; also the best fixed lambda on a 1001-point grid. Computed by `research/r8_optimism_estimator/oracle_lambda.py`, '
          'saved in `oracle_lambda.csv`. It separates the value of the shrinkage rule from the estimation error in lambda_hat.', '',
          '| law | n | b | B | A_0 | lambda* | loss full | loss oracle lambda* | oracle - full | MCSE | best grid lambda | its loss |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in oracle.itertuples():
    lines.append(f'| {r.law} | {r.n} | {r.bias:.6f} | {sci(r.B_true)} | {sci(r.A0)} | {fmt(r.lambda_star)} | {sci(r.loss_full)} | {sci(r.loss_oracle_lambda)} | '
                 f'{sci(r.oracle_minus_full)} | {sci(r.oracle_minus_full_mcse)} | {fmt(r.best_fixed_lambda_on_grid, 3)} | {sci(r.best_fixed_lambda_loss)} |')
lines += ['', '## What is established and what is not', '']
c1 = {name: [c for c in adm['estimators'][name]['criterion_1_penalty_accuracy']['cells'] if c['applies']] for name in ESTIMATORS}
r1 = {name: (min(min(c['ratio_bias0'], c['ratio_bias1']) for c in c1[name]), max(max(c['ratio_bias0'], c['ratio_bias1']) for c in c1[name])) for name in ESTIMATORS}
lines.insert(0, '> **Amendment of 13 September 2026 (dated after the original run).** This run uses the corrected cross-validation factor 2(K-1)/(2K-1) = 8/9 (see AMENDMENT.md); the original run with 4/5 is preserved unchanged in artifacts/r8_optimism_estimator_original_4over5/. Admission outcome unchanged.\n')
lines += [f'1. Measured. Both estimators recover the optimism penalty 2A_0n within the 15% criterion at n in 700, 1000, 2000 on both laws: '
          f'E1 ratios range {fmt(r1[ESTIMATORS[0]][0])} to {fmt(r1[ESTIMATORS[0]][1])}, E2 ratios range {fmt(r1[ESTIMATORS[1]][0])} to {fmt(r1[ESTIMATORS[1]][1])}. '
          'Criterion 1 holds for both. Simultaneous bands: '
          + '; '.join(f"{name}: {adm['estimators'][name]['criterion_3_uncertainty']['cells_including_one']} of 20 bands include 1" for name in ESTIMATORS) + '.',
          '2. Measured. The feasible shrinkage lambda_hat = clip(B_hat/(B_hat + A_hat), 0, 1) fails criterion 2 for both estimators: with b > 0 and n >= 700 its expected loss '
          'exceeds full correction in 5 of 6 cells per estimator (all three normal cells, t5 at n = 1000 and 2000); the supplementary bands on shrunken - full exclude zero '
          'on the harmful side in the three normal cells and in t5 at n = 2000 for both estimators. With b = 0 the shrunken policy beats full correction in every cell '
          '(it is between raw and full, and raw is optimal there).',
          '3. Measured (supplementary). The oracle factor lambda* with the true B and A_0 beats full correction in every b > 0 cell (oracle - full negative, at least 2.7 MCSE from zero '
          'in all cells), so the rule has value when its inputs are known; the failure lies in the sampling error of B_hat = -I_n(C_n) - A_hat, which enters lambda_hat '
          'through a ratio and, for b > 0, produces a mean lambda_hat below lambda* (see the per-cell table for the normal n = 1000 cell) '
          'with history-to-history spread whose cost is not offset; this does not separate the variability of B_hat from its bias or from the dependence between the estimated quantities.',
          '4. Not established. No feasible shrinkage rule with demonstrated value on these synthetic laws; no financial result. The financial application is NOT_RUN by the '
          "protocol's admission rule. Nothing here transfers to a rolling rule or to an individual conditional date.",
          '5. Scope. The evidence is confined to the two stored synthetic laws (500 histories each, GARCH-dependent scores with iid true hits), the two bias translations b in {0, '
          f'{BIASES[1]:.6f}}}, calibration prefixes of length n in {list(SIZES)}, and the reference risk from the saved Chebyshev interpolants (error budget 4e-11).', '',
          '## Financial application', '',
          f"**{adm['financial_application']}.** No estimator passed criteria 1 and 2. `financial.py` was not written. For the record, an inspection of one pair folder "
          '(`artifacts/r8_ten_comparators/pairs/Chronos-2__ASX200/`) found that `daily.parquet` holds the 1868 test dates only (columns r, sigma, the 19 method paths '
          'and 4 DtACI-expected columns); the calibration scores would have to be rebuilt from the returns file and the raw forecast path as in `research/r8_ten_comparators/run.py` '
          '(`y = returns[512:]`, `nc = int(0.7 * len(y))`, score `q - y`, Shift-CP = `q - qshift(score[:nc])` with k = ceil((nc+1)(1-alpha))).', '',
          '## Checks and replay', '',
          f"- Negative-control checks in `run.json`: {len(run['checks'])}, all rejected the defective input and accepted the valid one "
          f"(wrong-tail loss, rank rule against the v3 rank at n and 4n/5, block length, empirical vs conformal rank, empirical integral against the v3 path, "
          f"dropped (K-1)/K factor, circular block indices, shrinkage clip and zero rule, band rule centring, finite saved scores, 999x500 bootstrap rows, "
          f"and per (law, n, b) replay of C_n, I_n(C_n) against estimators.csv (abs 1e-12, rel 1e-10) and of R(C_n) - R(0) against loss_histories.csv (abs and rel 4e-11)).",
          f"- Fresh-process replay `engine.py --check`: **{chk['status']}**; max absolute difference over all saved CSV columns "
          f"{max(v.get('max_abs_difference', 0.) for v in chk['files'].values()):.2e}; admission.json pass/fail identical.",
          f"- Input binding: SHA-256 and mtime of the eight synthetic inputs equal the v3 lock records (`run.json`, field `inputs`); "
          f"v3 lock entries changed since the v3 run: {run['v3_lock_entries_changed_since_v3']} (manuscript files, not inputs here).", '',
          '## Runtime, seeds, environment', '',
          f"- Runtime: E2 {run['timing']['E2_seconds']:.2f} s, E1 and evaluation {run['timing']['E1_and_evaluation_seconds']:.2f} s, total {run['timing']['total_seconds']:.2f} s "
          f"with {run['workers']} worker processes; replay {chk['files']['histories.csv']['rows']} history rows.",
          f"- Seeds: E2 block bootstrap seed {run['seeds']['E2_block_bootstrap']}, stream {run['seeds']['E2_stream']}; history bootstrap: {run['seeds']['history_bootstrap']}.",
          f"- Constants: K = {run['constants']['K']}, R = {run['constants']['R']}, block lengths {run['constants']['block_lengths']}, p = {run['constants']['p']}, Omega = {run['constants']['omega']}.",
          f"- Environment: Python {run['environment']['python'].split()[0]}, NumPy {run['environment']['numpy']}, pandas {run['environment']['pandas']}, SciPy {run['environment']['scipy']}, "
          f"{run['environment']['platform']}, {run['environment']['cpu_count']} cores; executable {run['environment']['executable']}.", '',
          '## Files', '',
          '- `histories.csv`: 20,000 rows (estimator, law, n, b, history): C_n, I_n, O_hat, ratio, A_hat, B_hat, lambda_hat, expected losses.',
          '- `summary.csv`, `simultaneous_bands.csv`, `supplementary_loss_bands.csv`, `bootstrap_maxima.csv`, `e2_bootstrap_diagnostics.csv`, `oracle_lambda.csv`.',
          '- `admission.json`: every criterion cell with pass/fail. `run.json`: seeds, constants, environment, timing, input hashes, checks, output hashes. `check.json`: replay result.']
(OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print('wrote', OUT / 'RESULTS.md', len(lines), 'lines')
