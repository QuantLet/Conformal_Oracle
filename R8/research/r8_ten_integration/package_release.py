"""Write a new complete numerical release without modifying the preceding one."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from build import PROJECT,SOURCE,OUT,sha,check

SKIP={'__pycache__','.DS_Store','.cache','.pytest_cache'}


def main():
    check();base=PROJECT/'release/R8_20260910_commodity_funds'
    assert base.is_dir()
    report=json.loads((PROJECT/'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json').read_text())
    assert report['ten_external_displays']['exact_display_replay']==13
    assert json.loads((PROJECT/'artifacts/r8_external/july2026/completion.json').read_text())['status']=='complete'
    name='R8_20260910_ten_external';target=PROJECT/'release'/f'{name}.zip'
    assert not target.exists(),'Preserve a preceding release rather than overwriting it'
    partial=target.with_suffix('.partial.zip');assert not partial.exists()
    files={str(p.relative_to(base)):p for p in base.rglob('*') if p.is_file() and p.name!='RELEASE_MANIFEST.json' and not SKIP.intersection(p.parts)}
    trees=['source/sections_r8','source/analysis/provenance_r8','source/scripts/extension_20260831',
           'research/r8_model_extension','research/r8_ten_comparators','research/r8_ten_integration','research/r8_external',
           'artifacts/r8_model_extension','artifacts/r8_ten_comparators','artifacts/r8_ten_integration','artifacts/r8_external',
           'artifacts/r8_commodity_etp','artifacts/r8_native_candidates','artifacts/r8_grid_candidates',
           'research/r8_native_candidates','research/r8_grid_candidates']
    for tree in trees:
        for p in (PROJECT/tree).rglob('*'):
            if p.is_file() and not SKIP.intersection(p.parts) and p.name not in ['full_release.json','package_release.log']:
                files[str(p.relative_to(PROJECT))]=p
    for name_doc in ['main_R8','supplement_R8']:
        for ext in ['tex','pdf','aux','log','bbl']:files[f'source/{name_doc}.{ext}']=SOURCE/f'{name_doc}.{ext}'
    for rel in ['source/calibrating_the_oracle.bib','docs/IRFA_REVIEW_STATE.md','docs/IRFA_TEN_MODEL_COMPLETION.md',
                'docs/IRFA_TEN_STRONG_COMPARATORS.md','docs/IRFA_EXTERNAL_JULY_RESULTS.md','docs/IRFA_TEN_EXTERNAL_INTEGRATION.md']:
        files[rel]=PROJECT/rel
    for manifest in [PROJECT/'artifacts/r8_commodity_etp/panel/base/results/paper_outputs_manifest.json',
                     PROJECT/'artifacts/r8_commodity_etp/panel/risk_displays/displays.json',
                     PROJECT/'artifacts/r8_regime/paper_displays.json',OUT/'displays.json']:
        assert manifest.exists(),manifest
        for p in json.loads(manifest.read_text())['outputs']:files['source/'+p]=SOURCE/p
    files['README.md']=PROJECT/'research/r8_ten_integration/README.md'
    files['historical/commodity_release_manifest.json']=base/'RELEASE_MANIFEST.json'
    total=sum(p.stat().st_size for p in files.values())
    assert shutil.disk_usage(target.parent).free>total*1.03,'Insufficient free space for a new release'
    print('Packaging',len(files),'files,',total,'bytes',flush=True)
    hashes={};stored={'.npz','.npy','.png','.pdf','.parquet','.zip','.safetensors','.bin','.pt','.pth','.gz'}
    with zipfile.ZipFile(partial,'w',allowZip64=True) as z:
        for i,(relative,p) in enumerate(sorted(files.items()),1):
            info=zipfile.ZipInfo(name+'/'+relative);info.compress_type=zipfile.ZIP_STORED if p.suffix in stored else zipfile.ZIP_DEFLATED
            digest=hashlib.sha256()
            with p.open('rb') as source,z.open(info,'w',force_zip64=True) as destination:
                for block in iter(lambda:source.read(4*1024*1024),b''):digest.update(block);destination.write(block)
            hashes[relative]=digest.hexdigest()
            if i%1000==0:print('Written',i,'/',len(files),flush=True)
        manifest=dict(revision='R8',market_endpoint='2026-08-31',external_endpoint='2026-07-31',
                      base_release_manifest_sha256=sha(base/'RELEASE_MANIFEST.json'),files=hashes)
        z.writestr(name+'/RELEASE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(partial) as z:
        assert len(z.namelist())==len(hashes)+1
        for i,(relative,want) in enumerate(hashes.items(),1):
            digest=hashlib.sha256()
            with z.open(name+'/'+relative) as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==want,relative
            if i%1000==0:print('Verified',i,'/',len(hashes),flush=True)
    partial.rename(target)
    record=dict(status='complete',archive=str(target.relative_to(PROJECT)),archive_sha256=sha(target),files=len(hashes),
                uncompressed_bytes=total,archived_bytes_verified=True,preceding_release_unchanged=True)
    (OUT/'full_release.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)


if __name__=='__main__':main()
