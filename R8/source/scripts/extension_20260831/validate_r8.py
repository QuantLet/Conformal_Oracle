#!/usr/bin/env python3
"""Validate the complete R8 document scope and its generated numerical displays.

Reuses the repository's build, reference, numeric-literal and producer guards,
with their negative controls. Hash checks supplement (not replace) numerical
replay. Legacy revisions are not inputs to this self-contained release.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
import pandas as pd
from panel_statistics import ALPHAS,load_pair
from commodity_scope import ROOT,ARCHIVE,MODELS,N_PAIRS

SOURCE=Path(__file__).resolve().parents[2]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_displays(manifest,root=ROOT,source=SOURCE):
    assert manifest['producer_sha256']==digest(Path(__file__).with_name('build_paper_outputs.py'))
    assert manifest['scope_sha256']==digest(Path(__file__).with_name('commodity_scope.py'))
    for name,want in manifest['inputs'].items():assert digest(root/name)==want,('stale input',name)
    for name,want in manifest['outputs'].items():assert digest(source/name)==want,('stale display',name)


def display_replay():
    import build_paper_outputs as build
    manifest=json.loads((ROOT/'results/paper_outputs_manifest.json').read_text())
    verify_displays(manifest)
    damaged=json.loads(json.dumps(manifest));key=next(iter(damaged['outputs']));damaged['outputs'][key]='0'*64
    try:verify_displays(damaged)
    except AssertionError:pass
    else:raise AssertionError('Display corruption negative control did not fail')
    with tempfile.TemporaryDirectory(prefix='irfa-r8-displays-') as folder:
        root=Path(folder)/'archive';source=Path(folder)/'source';root.mkdir()
        shutil.copy2(ROOT/'primary_ready.json',root/'primary_ready.json')
        for name in manifest['inputs']:
            out=root/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,out)
        build.ROOT=root;build.SOURCE=source;build.OUT=source/'sections_r8';build.FIG=source/'figures'
        build.OUT.mkdir(parents=True);build.FIG.mkdir()
        build.main();new=json.loads((root/'results/paper_outputs_manifest.json').read_text())
        changed={k:[v,new['outputs'].get(k)] for k,v in manifest['outputs'].items() if new['outputs'].get(k)!=v}
        assert new==manifest,('Rebuilt displays/macros differ',changed,
                              'macros_equal',new['macros']==manifest['macros'])
    return dict(displays=len(manifest['outputs']),input_files=len(manifest['inputs']),exact=True,negative_control=True)


def panel():
    import panel_statistics
    panel_statistics.ROOT=ARCHIVE
    inv=pd.read_csv(ROOT/'quality/asset_inventory.csv').set_index('asset');rows=[]
    assert len(inv)==24
    for asset in inv.index:
        ret=pd.read_csv(ARCHIVE/'data/returns'/f'{asset}.csv',index_col='date',parse_dates=True).log_return
        assert np.isfinite(ret).all() and ret.index.is_unique and ret.index.is_monotonic_increasing
        assert len(ret)==inv.loc[asset,'n_returns']
        endpoint='2026-08-28' if asset in ['FTSE100','CBU0','IBGL'] else '2026-08-31'
        assert str(ret.index[-1].date())==endpoint==inv.loc[asset,'last_date']
        for model in MODELS:
            y,p=load_pair(model,asset);q=p[[f'VaR_{a:g}' for a in ALPHAS]].to_numpy()
            assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all(),(asset,model,'quantile order')
            rows.append(dict(model=model,asset=asset,rows=len(q),first=str(p.index[0].date()),last=endpoint,
                             positive_lower_quantiles=int((q[:,0]>0).sum())))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'quality/forecast_structure.csv',index=False)
    methods=pd.read_csv(ROOT/'results/posthoc.csv')
    assert len(methods)==N_PAIRS*10 and not methods.duplicated(['model','asset','method']).any()
    assert methods.groupby(['model','asset']).n_test.nunique().eq(1).all()
    assert ((methods.pihat*methods.n_test-methods.viol).abs()<1e-10).all()
    assert methods.groupby('method').size().eq(N_PAIRS).all()
    assert set(methods.model)==set(MODELS)
    common=pd.read_csv(ROOT/'results/common_support.csv')
    assert len(common)==24*((len(MODELS)+3)*2+2)
    assert common.groupby('asset')[['first','last','n_test']].nunique().eq(1).all().all()
    # The complete native replay records must bind to the current inputs.
    from validate_panel import native,stage
    stage(json.loads((ROOT.parent/'stage.json').read_text()))
    native_count=sum(row['rows'] for row in native())
    return dict(assets=24,pairs=N_PAIRS,tail_cells=N_PAIRS*len(ALPHAS),daily_forecast_rows=int(frame.rows.sum()),native_rows=native_count)


def documents(skip_build=False):
    path=SOURCE/'scripts/build_guards.py';spec=importlib.util.spec_from_file_location('r8_guards',path)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    g.DOCS=['main_R8']  # The appendix is included in the canonical article.
    g.LITERAL_DOCS=g.DOCS+[str(p.relative_to(SOURCE).with_suffix('')) for p in sorted((SOURCE/'sections_r8').glob('*.tex')) if not p.name.startswith(('tab_','numbers_'))]
    g.DOCS_TEX=[x+'.tex' for x in g.LITERAL_DOCS]
    g.DECLARED=SOURCE/'analysis/provenance_r8/DECLARED_CONSTANTS.md';g.PRODUCERS=SOURCE/'analysis/provenance_r8/PRODUCERS.tsv'
    names=['undefined','literals','producers']+([] if skip_build else ['compiles'])
    for name in names:
        assert getattr(g,'control_'+name)(),name+' negative control failed'
        assert getattr(g,'guard_'+name)(),name+' guard failed'
    for doc in g.DOCS:
        log=(SOURCE/f'{doc}.log').read_text(encoding='utf-8',errors='backslashreplace')
        assert not re.search(r'Citation .*undefined|There were undefined|multiply defined|Overfull \\[hv]box|^!',log,re.M),(doc,'LaTeX diagnostic')
    labels={};texts=[]
    for name in g.DOCS_TEX:
        s=(SOURCE/name).read_text();texts.append(s)
        for label in re.findall(r'\\label\{([^}]+)\}',s):
            # Cross-document labels may share names; duplicates within one
            # document's inclusion closure are checked by LaTeX above.
            labels.setdefault(label,[]).append(name)
    assert all('sections_r7' not in text and 'numbers.tex' not in text for text in texts)
    return dict(guards=names,negative_controls=len(names),undefined_references=0,undefined_citations=0,overfull_boxes=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--skip-build',action='store_true');args=ap.parse_args()
    path=SOURCE.parent/'research/r8_integration/build.py'
    spec=importlib.util.spec_from_file_location('risk_display_builder',path)
    risk=importlib.util.module_from_spec(spec);spec.loader.exec_module(risk)
    path=SOURCE.parent/'research/r8_regime/build_paper.py'
    spec=importlib.util.spec_from_file_location('regime_display_builder',path)
    regime=importlib.util.module_from_spec(spec);spec.loader.exec_module(regime)
    path=SOURCE.parent/'research/r8_ten_integration/build.py'
    spec=importlib.util.spec_from_file_location('ten_external_display_builder',path)
    ten=importlib.util.module_from_spec(spec);spec.loader.exec_module(ten)
    path=SOURCE.parent/'research/r8_partial_integration/build.py'
    spec=importlib.util.spec_from_file_location('partial_display_builder',path)
    partial=importlib.util.module_from_spec(spec);spec.loader.exec_module(partial)
    path=SOURCE.parent/'research/r8_information_integration/build.py'
    spec=importlib.util.spec_from_file_location('information_display_builder',path)
    information=importlib.util.module_from_spec(spec);spec.loader.exec_module(information)
    result=dict(panel=panel(),displays=display_replay(),risk_displays=risk.check(),regime_displays=regime.check(),
                ten_external_displays=ten.check(),partial_displays=partial.check(),
                information_displays=information.check(),documents=documents(args.skip_build))
    (ROOT/'quality/r8_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
