"""Write the LaTeX table, runtime record and RESULTS.md from validated Developed ex-US outputs."""
from datetime import datetime,timezone
import importlib.metadata as md
import json
import platform
import sys
import numpy as np
import pandas as pd
from prepare import PROJECT,OUT,ASSETS,END,END_LABEL,sha
from corrections import MODELS,METHODS
RESULTS=OUT/'results'
LABELS={'DtACI-projected-expected':'Projected DtACI','State-L1':'Regularised state','Rolling500':'Rolling 500',
        'Selected-rolling':'Selected rolling','Gate-selected-rolling':'Coverage-gated rolling','Loss-gate':'Loss gate',
        'Past-minimum':'Past-loss minimum','State-L1-clipped':'Regularised state (clipped)','DtACI-projected-seed':'Projected DtACI (seeded path)'}
TABLE=['Raw','Shift-CP','Vol-ERM','State-L1','POT-Shift','POT-Vol','DtACI-projected-expected','Rolling500',
       'Selected-rolling','Gate-selected-rolling','Loss-gate','Past-minimum']


def stage_times():
    rows=[l.split(' ',1) for l in (OUT/'stage_times.log').read_text().splitlines() if ' ' in l]
    marks={}
    for t,label in rows:
        if label.endswith(' start') or label.endswith(' end'):
            name,kind=label.rsplit(' ',1);marks.setdefault(name,{})[kind]=t
    out={}
    for name,d in marks.items():
        if 'start' in d and 'end' in d:
            a=datetime.fromisoformat(d['start'].replace('Z','+00:00'));b=datetime.fromisoformat(d['end'].replace('Z','+00:00'))
            out[name]=dict(start=d['start'],end=d['end'],seconds=int((b-a).total_seconds()))
    return out


def main():
    checks={n:json.loads((OUT/n).read_text()) for n in ['admission.json','base_validation.json','correction_validation.json','aggregation_validation.json']}
    assert all(c['status']=='passed' for c in checks.values())
    done=json.loads((RESULTS/'complete.json').read_text());assert done['status']=='complete' and done['pairs']==100
    assert checks['aggregation_validation.json']['result_complete_sha256']==sha(RESULTS/'complete.json')
    summary=pd.read_csv(RESULTS/'summary.csv').set_index('method');intervals=pd.read_csv(RESULTS/'intervals.csv')
    primary=json.loads((RESULTS/'primary_decisions.json').read_text());by_model=pd.read_csv(RESULTS/'by_model.csv')
    periods=pd.read_csv(RESULTS/'period_summary.csv');decisions=pd.read_csv(RESULTS/'decisions.csv');pairs=pd.read_csv(RESULTS/'pairs.csv')
    adm=checks['admission.json']
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Method & QS & Normalised QS & Viol. (\%) & UC & Worse \\',r'\midrule']
    for name in TABLE:
        r=summary.loc[name];uc=str(int(r.kupiec_rejections)) if r.kupiec_available else '--'
        lines.append(f'{LABELS.get(name,name)} & {r.QS_x10000:.4f} & {r.normalised_QS:.6f} & {100*r.violation_rate:.3f} & {uc} & {int(r.worse_than_raw)}'+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}'];(RESULTS/'tab_external.tex').write_text('\n'.join(lines)+'\n')
    times=stage_times();total=sum(v['seconds'] for v in times.values())
    runtime=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),executable=sys.executable,
                 platform=platform.platform(),machine=platform.machine(),cores_available=18,workers=12,
                 libraries={n:md.version(n) for n in ['numpy','pandas','scipy','numba','arch','pyarrow','pytest']},
                 stages=times,total_staged_seconds=total,
                 scope='same-environment fresh replay; no claim of cross-platform bitwise identity')
    (OUT/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    md_=[]
    w=md_.append
    w('# Developed ex-US 25 size/value portfolios: external test results\n')
    w('Protocol: `research/r8_external2/PROTOCOL.md` (SHA-256 '+adm['protocol_sha256']+'), fixed 13 September 2026 before download. Universe: 25 value-weighted Developed ex-US ME x BE-ME daily portfolios (USD, dividends included). Models: HS, GJR-GARCH-t, CAViaR-AS, GAS-t. Pairs: 100 (all completed).\n')
    w('## Source and admission\n')
    w(f"- Data file: {adm['raw_source']}\n- Retrieved (UTC): {adm['raw_retrieved_utc']}\n- SHA-256 of the archived zip: {adm['raw_sha256']}\n- Zip member: {adm['source']['zip_member']}; header: \"{adm['source']['source_header'][0]}\"")
    w(f"- Details page used: {adm['details_page']} (SHA-256 {adm['details_page_sha256']}); advertised daily returns July 1, 1990 - July 31, 2026. The protocol's guessed details page {adm['protocol_details_page']['url']} returned HTTP {adm['protocol_details_page']['http_status']}; the library index links the page above as \"Details\" for this file.")
    w(f"- Value-weighted table: \"{adm['source']['value_table_title']}\", file lines {adm['source']['value_table_start_line']}-{adm['source']['value_table_end_line']}; the equal-weighted table ({adm['source']['other_tables']}) is excluded.")
    w(f"- Sentinels: file header states \"{adm['source']['missing_statement'][0]}\"; -99.99 is mapped to missing; -999 (documented only in the CRSP industry file) is counted and never mapped. Counts in the value-weighted table: {adm['sentinel_handling']['counts']}. Missing cells in the admitted window: {adm['missing_in_admitted_window']}.")
    w(f"- Rows in the file: {adm['source']['rows']}, {adm['source']['first']} to {adm['source']['last']}. The file begins 2 July 1990; the protocol text said November 1990. Pre-2000 history enters only as the 1,250-observation initial context (first context date {adm['first_context']}).")
    w(f"- Endpoint: last date in the file = {adm['endpoint']['value']} (protocol minimum {adm['endpoint']['minimum_required']}). First forecast {adm['first_forecast']}; calibration 2000-2014 = {adm['calibration_dates']} dates; test {adm['first_test']} to {adm['last_forecast']} = {adm['test_dates']} dates.")
    w(f"- Calendar: {adm['calendar']}. Every admitted date carries all 25 portfolios. Compared with the archived S&P 500 calendar over 2000-07/2026: {adm['calendar_sp500_comparison']['dates_here']} dates here versus {adm['calendar_sp500_comparison']['dates_sp500']}; {adm['calendar_sp500_comparison']['here_not_in_sp500']} weekday dates here are US holidays absent from the S&P calendar; {adm['calendar_sp500_comparison']['sp500_not_here']} S&P dates are absent here. The first test date is 2015-01-01 (a weekday in this file).")
    w('- Transformation: log(1+R/100). Checks passed: unique increasing dates, finite values, all returns above -100%, 25 named columns, complete weekday calendar.\n')
    w('## Base forecasts and validation\n')
    b=checks['base_validation.json']
    w(f"- HS and GJR-GARCH-t: 250-observation daily fits (`source/scripts/extension_20260831/classical.py`, unchanged). Classical forecast rows reconstructed independently: {b['classical_forecast_rows_reconstructed']}; fixed-parameter GJR reconstructions: {b['fixed_parameter_gjr_reconstructions']}; window-sd fallbacks: {b['gjr_window_sd_fallbacks']}.")
    w(f"- CAViaR-AS and GAS-t: refitted each January 2000-{END[:4]} on the preceding 1,250 returns. Yearly fits replayed: {b['dynamic_yearly_fits_replayed']}; selected fits with a non-success optimiser flag: {b['dynamic_selected_nonconvergence_flags']} (retained under the minimum-objective rule).")
    w(f"- Base series: {b['base_series']}; full fresh replays: {b['full_fresh_replays']}; base validation status: {b['status']}.\n")
    c=checks['correction_validation.json']
    w('## Corrections and validation\n')
    w(f"- Pairs: {c['pairs']}; fresh replays: {c['full_fresh_replays']}; independent scalar checks: {c['independent_scalar_checks']}; exact array values compared: {c['exact_array_values']}; inner gate bootstrap draws recomputed: {c['independent_gate_bootstrap_draws']}; LP objectives: {c['LP_objectives']}; POT inversions: {c['POT_inversions']}; future-outcome perturbation pairs: {c['future_outcome_perturbations']}; status: {c['status']}.")
    w(f"- Loss-gate selections over 100 pairs: {decisions.loss_gate.value_counts().to_dict()}; past-loss minimum selections: {decisions.past_minimum.value_counts().to_dict()}; selected rolling windows: {decisions.selected_window.value_counts().to_dict()}; coverage gate active in {int(decisions.coverage_gate.sum())} pairs; POT final fallbacks: {int(decisions.pot_final_fallbacks.sum())}.\n")
    w('## Aggregate table (100 pairs, test 2015-01-01 to '+END+')\n')
    w('QS in return units x 10^4; normalised QS = per-pair pinball loss divided by the pair\'s 2000-2014 calibration-return standard deviation, averaged over pairs; Viol. = mean violation rate in percent; UC = number of pairs with Kupiec p < 0.05 (-- when the mixture loss has no integer hit count); Worse = pairs with QS above Raw.\n')
    w('| Method | QS x1e4 | Normalised QS | Viol. (%) | UC | Worse |\n|---|---|---|---|---|---|')
    for name in TABLE+['State-L1-clipped','DtACI-projected-seed']:
        r=summary.loc[name];uc=str(int(r.kupiec_rejections)) if r.kupiec_available else '--'
        w(f"| {LABELS.get(name,name)} | {r.QS_x10000:.4f} | {r.normalised_QS:.6f} | {100*r.violation_rate:.3f} | {uc} | {int(r.worse_than_raw)} |")
    w('\nThe last two rows are the clipped-L1 and seeded-DtACI sensitivities. LaTeX version: `results/tab_external.tex`.\n')
    w('## Simultaneous 95% bands (999 calendar circular-bootstrap draws, seed prefix 20260913)\n')
    w('Normalised units; difference = method minus reference in mean normalised QS; simultaneous bands are sup-t over the family. Return-unit bands (x 10^4) are in `intervals.csv`.\n')
    w('| Family | Method | Reference | Block | Difference | Pointwise 95% | Simultaneous 95% |\n|---|---|---|---|---|---|---|')
    for _,r in intervals[intervals.units=='normalised'].iterrows():
        w(f"| {r.family} | {LABELS.get(r.method,r.method)} | {LABELS.get(r.reference,r.reference)} | {int(r.block_calendar_days)} | {r.difference:.6f} | [{r.lower:.6f}, {r.upper:.6f}] | [{r.simultaneous_lower:.6f}, {r.simultaneous_upper:.6f}] |")
    w('\n## Transfer decision\n')
    w('Criterion (protocol): a loss advantage transfers only if the normalised difference is negative and its simultaneous upper endpoint is below zero at both block lengths (20 and 60 calendar days).\n')
    for d in primary:
        g=intervals[(intervals.units=='normalised')&(intervals.family==d['family'])&(intervals.method==d['method'])]
        verdict='TRANSFERS' if d['passes_primary_transfer_criterion'] else ('HIGHER LOSS at both bands' if d['higher_loss_both_bands'] else 'unresolved')
        w(f"- {d['family']}: {LABELS.get(d['method'],d['method'])} vs {LABELS.get(d['reference'],d['reference'])}: difference {g.difference.iloc[0]:.6f}; simultaneous upper endpoints {', '.join(f'{u:.6f} ({int(b)} d)' for u,b in zip(g.simultaneous_upper,g.block_calendar_days))}: **{verdict}**.")
    w('\n## By model\n')
    w('| Model | Method | Pairs | QS | Normalised QS | Violation rate |\n|---|---|---|---|---|---|')
    for _,r in by_model.iterrows():w(f"| {r.model} | {LABELS.get(r.method,r.method)} | {int(r.pairs)} | {r.QS:.8f} | {r.normalised_QS:.6f} | {r.violation_rate:.5f} |")
    w('\n## Descriptive periods (mean normalised QS over 100 pairs)\n')
    w('| Period | Method | QS | Normalised QS |\n|---|---|---|---|')
    for _,r in periods.iterrows():w(f"| {r.period} | {LABELS.get(r.method,r.method)} | {r.QS:.8f} | {r.normalised_QS:.6f} |")
    a=checks['aggregation_validation.json']
    w(f"\n## Aggregate validation\n\nStatus {a['status']}; daily losses rebuilt: {a['daily_losses_rebuilt']}; independent calendar draws: {a['independent_calendar_draws']}; bootstrap means verified: {a['bootstrap_means_verified']}; period rows: {a['period_pair_method_rows']}; interval rows: {a['simultaneous_interval_rows']}.\n")
    w('## Implementation notes and departures\n')
    w('- Scripts: `research/r8_external2/` copies of `research/r8_external/*.py`, changed only in root path, source URLs, the 25-column parser (whitespace-padded fields, one documented sentinel), the endpoint constant, the 2015 boundary assertion (first test date is 2015-01-01 here, 2015-01-02 in the industry file), the pair count, the calendar-bootstrap seed prefix (20260913) and the added third family (Shift-CP against Raw). Fitting, correction, selection and aggregation definitions are unchanged. No file outside `research/r8_external2/` and `artifacts/r8_external2/` was modified.')
    w('- The inner loss-gate bootstrap (499 draws) and the seeded DtACI paths use the seed construction inside `research/r8_decision/methods.py` (prefix 20260909 with the pair key), carried over unchanged as part of the correction definitions. The protocol seed 20260913 applies to the 999-draw calendar bootstrap.')
    w('- One parser check added relative to the industry parser: all 25 columns must parse as numeric and are cast to float. The archived file is all-float; the check changes no value.')
    w('- No foundation-model inference was run. The manuscript was not edited.\n')
    w('## Runtime and environment\n')
    w(f"- Python {runtime['python']} at `{runtime['executable']}`; {runtime['platform']}; libraries: {runtime['libraries']}; 18 cores available, 12 workers.")
    for k,v in times.items():w(f"- {k}: {v['seconds']} s ({v['start']} to {v['end']})")
    w(f"- Total of staged wall-clock seconds: {total} s.")
    (RESULTS/'RESULTS.md').write_text('\n'.join(md_)+'\n')
    print('report written',RESULTS/'RESULTS.md',flush=True)


if __name__=='__main__':main()
