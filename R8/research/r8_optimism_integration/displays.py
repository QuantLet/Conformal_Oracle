"""Render existing, immutable numerical evidence; no estimation or simulation."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/r8_optimism_integration'
TARGET = ROOT / 'source/sections_r8'
FILES = ['results/optimism_ar/primary/bands.csv',
         'results/optimism_ar/primary/dependence.csv',
         'results/optimism_ar/completion.json',
         'results/theory_loop/synthetic/validation_summary.csv']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


def render():
    bands = rows(FILES[0])
    dep = rows(FILES[1])
    validation = rows(FILES[3])
    d = next(r for r in dep if Decimal(r['phi']) == Decimal('.8') and r['n'] == '1000')
    c = next(r for r in bands if Decimal(r['phi']) == Decimal('.8') and r['evaluation'] == 'contiguous')
    v = {r['law']: r for r in validation if r['n'] == '1000'}
    completion = json.loads((ROOT / FILES[2]).read_text())
    values = {
        'nOptInflation': f"{Decimal(d['long_run_inflation']):.2f}",
        'nOptOmegaDep': f"{Decimal(d['omega']):.8f}",
        'nOptIidReference': f"{Decimal(c['iid_reference']):.4f}",
        'nOptContMean': f"{Decimal(c['mean_ratio']):.4f}",
        'nOptContLo': f"{Decimal(c['lower']):.4f}",
        'nOptContHi': f"{Decimal(c['upper']):.4f}",
        'nOptDensityNormal': f"{100*Decimal(v['normal']['median_density_relative_error']):.1f}",
        'nOptDensityT': f"{100*Decimal(v['t5']['median_density_relative_error']):.1f}",
        'nOptHistories': str(completion['independent_latent_histories']),
        'nOptBootstrap': str(completion['bootstrap_replicates']),
        'nOptFamily': str(completion['cells']),
    }
    macros = '% Generated from immutable study CSVs by research/r8_optimism_integration/displays.py.\n'
    macros += ''.join('\\newcommand{\\' + k + '}{' + value + '}\n' for k, value in sorted(values.items()))
    table = '\\begin{tabular}{rrlrr}\n\\toprule\n$\\phi$ & $n$ & Evaluation & Mean & Simultaneous band \\\\\n\\midrule\n'
    for r in bands:
        label = 'Contiguous' if r['evaluation'] == 'contiguous' else 'Independent marginal'
        table += f"{Decimal(r['phi']):g} & {int(r['n']):,} & {label} & {Decimal(r['mean_ratio']):.4f} & $[{Decimal(r['lower']):.4f}, {Decimal(r['upper']):.4f}]$ \\\\\n".replace('1,000', '1{,}000')
    table += '\\bottomrule\n\\end{tabular}\n'
    return {'numbers_optimism.tex': macros, 'tab_optimism.tex': table}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); a = p.parse_args()
    rendered = render()
    if a.check:
        for name, s in rendered.items():
            assert (TARGET / name).read_text() == s, name
    else:
        for name, s in rendered.items():
            (TARGET / name).write_text(s)
    record = {'status': 'passed', 'mode': 'check' if a.check else 'render',
              'producer_sha256': sha(__file__), 'inputs': {n: sha(ROOT / n) for n in FILES},
              'outputs': {str((TARGET / n).relative_to(ROOT)): sha(TARGET / n) for n in rendered}}
    (OUT / ('display_check.json' if a.check else 'display_build.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
