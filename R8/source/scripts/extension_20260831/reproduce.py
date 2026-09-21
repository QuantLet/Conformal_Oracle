#!/usr/bin/env python3
"""Entry point for the frozen August 2026 release. Run with its locked environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
ROOT=REPO/'artifacts/extension_20260831'


def run(script,*args,python=None):
    print('RUN',script,*args,flush=True)
    subprocess.run([python or sys.executable,str(CODE/script),*map(str,args)],cwd=REPO,check=True)


def hashes():
    manifest=REPO/'RELEASE_MANIFEST.json'
    if not manifest.exists():
        raise FileNotFoundError('Use the packaged release, which contains RELEASE_MANIFEST.json')
    records=json.loads(manifest.read_text())['files']
    for name,want in records.items():
        p=REPO/name
        assert p.is_file(),('missing release input',name)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==want,('changed release file',name)
    print('PASS',len(records),'release file hashes',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['hashes','verify','downstream','native','refit','research'])
    ap.add_argument('--tsfm-python',type=Path)
    ap.add_argument('--destination',type=Path)
    a=ap.parse_args()
    if a.mode=='hashes':hashes();return
    if a.mode=='research':
        if a.destination is None:ap.error('--destination is required for a fresh research replay')
        for scope in ['empirical','simulation']:
            subprocess.run([sys.executable,str(REPO/'research/r8_review/verify_extensions.py'),
                            '--scope',scope,'--destination',str(a.destination)],cwd=REPO,check=True)
        subprocess.run([sys.executable,str(REPO/'research/r8_review/validate_review.py')],cwd=REPO,check=True)
        return
    if a.mode=='refit':
        if a.destination is None:ap.error('--destination is required for a fresh refit')
        dest=a.destination.resolve()
        if dest.exists():ap.error('Refit destination must not exist; existing results are never erased')
        deps=['source/scripts/extension_20260831',
              'research/r8_review',
              'source/analysis/phase3_dynamic/run_dynamic_var.py',
              'source/analysis/ae_point4/run_ae_point4.py',
              'source/Quantlets/cfp_config.py',
              'source/Quantlets/CO_gamlss/baseline_gamlss.py']
        for name in deps:
            p=REPO/name;q=dest/name;q.parent.mkdir(parents=True,exist_ok=True)
            if p.is_dir():shutil.copytree(p,q,ignore=shutil.ignore_patterns('__pycache__'))
            else:shutil.copy2(p,q)
        root=dest/'artifacts/extension_20260831';root.mkdir(parents=True)
        for name in ['raw_responses','native']:
            shutil.copytree(ROOT/name,root/name)
        if (ROOT/'calendar_primary.json').exists():
            calendar='artifacts/review_20260909/calendar'
            shutil.copytree(REPO/calendar,dest/calendar,ignore=shutil.ignore_patterns('before_adoption','draws'))
        (root/'quality').mkdir();(root/'results').mkdir()
        shutil.copy2(ROOT/'quality/primary_returns_sources.json',root/'quality/primary_returns_sources.json')
        code=dest/'source/scripts/extension_20260831'
        for script in ['prepare_returns.py','classical.py','reduce_native.py','dynamic.py','evt_fhs.py','posthoc.py']:
            print('REFIT',script,flush=True)
            subprocess.run([sys.executable,str(code/script)],cwd=dest,check=True)
            if script=='reduce_native.py' and (ROOT/'calendar_primary.json').exists():
                subprocess.run([sys.executable,str(dest/'research/r8_review/promote_calendar.py'),
                                '--destination',str(root)],cwd=dest,check=True)
        run('compare_reconstruction.py',root,'--report',root/'quality/fresh_fit_comparison.json','--require-exact')
        return
    if a.mode=='native':
        if a.tsfm_python is None:ap.error('--tsfm-python is required for native replay')
        for model in ['moirai','moirai2','timesfm25','lagllama']:
            run('check_native.py','--model',model,'--device','cpu' if model=='moirai' else 'mps',
                '--all-batches',python=str(a.tsfm_python))
        for model in ['moirai','lagllama']:
            run('check_pools.py','--model',model,python=str(a.tsfm_python))
        return
    if a.mode=='verify':
        hashes()
        for script,args in [
            ('check_returns.py',()),('check_classical.py',()),('check_reduction.py',()),
            ('check_fitted_outputs.py',('--scope','posthoc')),
            ('check_fitted_outputs.py',('--scope','dynamic')),('check_evt_fhs.py',()),
            ('check_monte_carlo.py',()),('validate_r8.py',())]:run(script,*args)
        subprocess.run([sys.executable,'-m','pytest','-q',str(CODE/'test_statistics.py'),
                        str(CODE.parent/'test_reproduction_compare.py'),
                        str(REPO/'source/Quantlets/CO_gamlss/test_sst_inverse.py'),
                        str(REPO/'research/r8_review/test_controlled_comparisons.py')],cwd=REPO,check=True)
        subprocess.run([sys.executable,str(REPO/'research/r8_review/validate_review.py')],cwd=REPO,check=True)
    else:
        for script in ['aggregate_extensions.py','export_results.py']:
            subprocess.run([sys.executable,str(REPO/'research/r8_review'/script)],cwd=REPO,check=True)
        # Reaggregate archived fits, then regenerate all reported sensitivities,
        # Monte Carlo cells, tables and figures. Native inference is separate.
        for script in ['posthoc.py','dynamic.py','evt_fhs.py','analyse_panel.py',
                       'closure_sensitivity.py','analyse_pools.py','additional_diagnostics.py',
                       'monte_carlo.py','build_paper_outputs.py']:
            run(script)
        for doc in ['main_R8','supplement_R8','main_R8','supplement_R8']:
            subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',doc+'.tex'],
                           cwd=REPO/'source',check=True)
        run('validate_r8.py')
    print('PASS requested reconstruction mode:',a.mode,flush=True)


if __name__=='__main__':main()
