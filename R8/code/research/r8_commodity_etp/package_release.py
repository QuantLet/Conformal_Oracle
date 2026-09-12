#!/usr/bin/env python3
"""Package only the R8 dependency closure, with a complete file-hash manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from panel_scope import PROJECT, ROOT as PAPER_ROOT, ART

REPO=PROJECT
ROOT=REPO/'artifacts/extension_20260831'
EXCLUDED={'__pycache__','.DS_Store','.cache','.pytest_cache','mc_replay'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_tree(source,target):
    shutil.copytree(source,target,ignore=lambda p,n:[x for x in n if x in EXCLUDED])


def zip_tree(folder,archive):
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                compression=zipfile.ZIP_STORED if p.suffix in ['.npz','.png','.pdf','.parquet'] else zipfile.ZIP_DEFLATED
                z.write(p,p.relative_to(folder.parent),compress_type=compression,compresslevel=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=REPO/'release/R8_20260910_commodity_funds');a=ap.parse_args()
    assert json.loads((ROOT/'quality/full_reconstruction.json').read_text())['complete']
    assert json.loads((PAPER_ROOT/'quality/r8_validation.json').read_text())['displays']['exact']
    assert json.loads((PAPER_ROOT/'quality/r8_validation.json').read_text())['risk_displays']['exact_display_replay']==14
    assert json.loads((PAPER_ROOT/'quality/r8_validation.json').read_text())['regime_displays']['exact_display_replay']==2
    assert json.loads((ART/'validation.json').read_text())['exact_aggregation_replay']
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Release destination must not exist')
    out.mkdir(parents=True)
    source_files=['main_R8.tex','supplement_R8.tex','calibrating_the_oracle.bib','elsarticle.cls']
    source_dirs=['sections_r8','analysis/provenance_r8','scripts/extension_20260831']
    helpers=['analysis/phase3_dynamic/run_dynamic_var.py','analysis/ae_point4/run_ae_point4.py',
             'Quantlets/cfp_config.py','Quantlets/CO_gamlss/baseline_gamlss.py',
             'Quantlets/CO_gamlss/test_sst_inverse.py',
             'Quantlets/CO_full_evaluation/run_full_evaluation.py',
             'scripts/build_guards.py','scripts/reproduction_compare.py','scripts/test_reproduction_compare.py']
    for name in source_files+helpers:
        p=REPO/'source'/name;q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name in source_dirs:
        q=out/'source'/name;q.parent.mkdir(parents=True,exist_ok=True);copy_tree(REPO/'source'/name,q)
    (out/'source/figures').mkdir()
    for name in ['fig_loss_august.pdf','fig_mc_august.pdf',
                 'fig_traffic_light_august.pdf','fig_rolling_august.pdf','fig_complexity_august.pdf',
                 'fig_dependence_cost.pdf','fig_mechanism_one.pdf',
                 'fig_mechanism_five.pdf','fig_strong_frontier.pdf']:
        shutil.copy2(REPO/'source/figures'/name,out/'source/figures'/name)
    for name in ['main_R8.pdf','supplement_R8.pdf']:
        shutil.copy2(REPO/'source'/name,out/'source'/name)
    # TeX's cross-document labels are required by checks before their first
    # rebuild. The standalone source ZIP below deliberately omits these caches.
    for doc in ['main_R8','supplement_R8']:
        for ext in ['aux','log','bbl']:
            shutil.copy2(REPO/'source'/f'{doc}.{ext}',out/'source'/f'{doc}.{ext}')
    target=out/'artifacts/extension_20260831';target.mkdir(parents=True)
    for name in ['raw_responses','prices','data','native','parameters','provenance','posthoc',
                 'evt_fhs','draws','results','quality','models']:
        copy_tree(ROOT/name,target/name)
    for name in ['README.md','primary_ready.json','calendar_primary.json']:
        shutil.copy2(ROOT/name,target/name)
    copy_tree(REPO/'research/r8_review',out/'research/r8_review')
    copy_tree(REPO/'artifacts/review_20260909',out/'artifacts/review_20260909')
    for name in ['r8_decision','r8_mechanism','r8_integration','r8_external','r8_regime','r8_native_panel','r8_commodity_etp','r8_native_candidates','r8_grid_candidates']:
        copy_tree(REPO/'research'/name,out/'research'/name)
    for name in ['r8_decision','r8_mechanism','r8_risk_integration','r8_external','r8_regime','r8_native_panel','r8_commodity_etp','r8_native_candidates','r8_grid_candidates']:
        copy_tree(REPO/'artifacts'/name,out/'artifacts'/name)
    # Keep the two source replay reports to which the primary certificates refer.
    repair=ROOT/'unfiltered_repair/quality';(target/'unfiltered_repair/quality').mkdir(parents=True)
    for p in repair.glob('native_replay_*_full.json'):
        shutil.copy2(p,target/'unfiltered_repair/quality'/p.name)
    shutil.copy2(ROOT/'README.md',out/'README.md')
    # The latest instructions precede the unchanged base-data reconstruction guide.
    (out/'README.md').write_text((REPO/'research/r8_commodity_etp/README.md').read_text()+
                               '\n\n---\n\n'+(REPO/'research/r8_integration/README.md').read_text()+
                               '\n\n---\n\n'+(ROOT/'README.md').read_text())
    for name in ['IRFA_AUGUST2026_R8_VALIDATION.md','IRFA_R8_FIGURE_STYLE_VALIDATION.md',
                 'IRFA_R8_FIGURE_RESTORATION.md','IRFA_R8_SCIENTIFIC_REPAIR_VALIDATION.md',
                 'IRFA_R8_RISK_INTEGRATION.md','IRFA_STRONG_COMPARATORS_RESULTS.md',
                 'IRFA_CONTROLLED_MECHANISM_RESULTS.md','IRFA_REGIME_CHANGE_RESULTS.md','CONFERENCE_PROGRAM_ABSTRACT.md',
                 'IRFA_REVIEW_STATE.md','IRFA_NATIVE_PANEL.md','IRFA_COMMODITY_ETP_REPLACEMENT.md']:
        report=REPO/'docs'/name
        if report.exists():
            (out/'docs').mkdir(exist_ok=True);shutil.copy2(report,out/'docs'/report.name)
    files={str(p.relative_to(out)):digest(p)
           for p in sorted(out.rglob('*')) if p.is_file()}
    manifest=dict(vintage='2026-08-31',revision='R8',files=files)
    (out/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Small portable LaTeX source package for clean builds and editorial review.
    with zipfile.ZipFile(out.parent/'R8_LaTeX_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=[Path(x) for x in source_files]+list((out/'source/sections_r8').glob('*.tex'))
        for name in names:
            p=name if name.is_absolute() else out/'source'/name
            z.write(p,p.relative_to(out/'source'))
        for p in (out/'source/figures').glob('*.pdf'):z.write(p,p.relative_to(out/'source'))
        z.writestr('BUILD.txt','Run latexmk -g -pdf main_R8.tex, then supplement_R8.tex, then each again with -g. TeX Live 2026.\n')
    zip_tree(out,out.with_suffix('.zip'))
    result=dict(files=len(files),bytes=sum((out/name).stat().st_size for name in files),
                archive=str(out.with_suffix('.zip')),sha256=digest(out.with_suffix('.zip')))
    (out.parent/'R8_commodity_release_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
