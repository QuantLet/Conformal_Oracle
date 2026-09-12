"""Complete release with the verified finite-count construction and current documents."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'artifacts/r8_count_law'
BLOCK=4*1024*1024


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(BLOCK),b''):h.update(block)
    return h.hexdigest()


def verify_extension(archive_path,prefix,hashes,check):
    # Exercise the new validator using only bytes extracted from this release.
    # This catches an input omitted from an otherwise internally hashed ZIP.
    names=set(check['current_manuscript'])
    names.update(p for p in hashes if p.startswith(('research/r8_count_law/',
        'artifacts/r8_count_law/','artifacts/r8_count_law_replay/')))
    names.update(json.loads((OUT/'validation.json').read_text())['inputs'])
    names.update(['artifacts/r8_horizon_bridge/final_validation.json',
        'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json',
        'artifacts/r8_ten_integration/source_package.json',
        'release/R8_LaTeX_sources.zip',check['source_package']['source_zip']])
    for document in check['source_package']['documents']:
        names.update(f'source/{document}.{extension}' for extension in ('pdf','log'))
    assert names.issubset(hashes),sorted(names-set(hashes))
    with zipfile.ZipFile(archive_path) as archive,tempfile.TemporaryDirectory(prefix='irfa-count-archived-') as folder:
        for name in names:
            target=Path(folder)/name;target.parent.mkdir(parents=True,exist_ok=True)
            data=archive.read(prefix+'/'+name)
            assert hashlib.sha256(data).hexdigest()==hashes[name]
            target.write_bytes(data)
        run=subprocess.run([sys.executable,'research/r8_count_law/validate.py'],
            cwd=folder,capture_output=True,text=True)
        assert run.returncode==0,(run.stdout,run.stderr)
        replay=json.loads((Path(folder)/'artifacts/r8_count_law/final_validation.json').read_text())
        assert replay==check
    print('Archive-only extension validation reproduced exactly:',len(names),'members',flush=True)
    return len(names)


def main():
    receipt=json.loads((PROJECT/'artifacts/r8_horizon_bridge/full_release.json').read_text())
    assert receipt['status']=='complete'
    prior=PROJECT/receipt['archive']
    assert sha(prior)==receipt['archive_sha256']
    check=json.loads((OUT/'final_validation.json').read_text())
    assert check['status']=='passed'
    assert all(sha(PROJECT/p)==h for p,h in check['current_manuscript'].items())
    assert all(sha(PROJECT/f'source/{p}.pdf')==v['pdf_sha256'] for p,v in check['source_package']['documents'].items())
    assert sha(PROJECT/check['source_package']['source_zip'])==check['source_package']['source_zip_sha256']
    assert sha(PROJECT/'release/R8_LaTeX_sources.zip')==check['source_package']['source_zip_sha256']
    prefix='R8_20260910_count_law'
    target=PROJECT/'release'/f'{prefix}.zip'
    partial=target.with_suffix('.partial.zip')
    assert not target.exists() and not partial.exists(),'Preserve preceding releases'
    overlays={}
    for root in ['research/r8_count_law','artifacts/r8_count_law','artifacts/r8_count_law_replay']:
        for p in (PROJECT/root).rglob('*'):
            if p.is_file() and not {'__pycache__','.DS_Store','.pytest_cache'}.intersection(p.parts) and p.name not in ('full_release.json','package_release.log'):
                overlays[str(p.relative_to(PROJECT))]=p
    extra=[*check['changed_manuscript_files'],
        'docs/IRFA_COUNT_LAW_RESULT.md','docs/IRFA_REVIEW_STATE.md',
        'research/r8_ten_integration/README.md','Manuscript_R8.pdf',
        'artifacts/r8_commodity_etp/panel/base/quality/r8_validation.json',
        'artifacts/r8_ten_integration/source_package.json',
        'artifacts/r8_horizon_bridge/full_release.json',
        'release/R8_20260910_commodity_sources.zip',
        'release/R8_20260910_ten_external_sources.zip','release/R8_LaTeX_sources.zip']
    extra.extend(json.loads((OUT/'validation.json').read_text())['inputs'])
    for doc in ('main_R8','supplement_R8'):
        extra.extend(f'source/{doc}.{ext}' for ext in ('tex','pdf','aux','log','bbl'))
    extra.extend(str(p.relative_to(PROJECT)) for p in (PROJECT/'artifacts/r8_ten_integration').glob('portable_*.log'))
    for relative in extra:overlays[relative]=PROJECT/relative
    overlays['README.md']=PROJECT/'research/r8_ten_integration/README.md'
    stored={'.npz','.npy','.png','.pdf','.parquet','.zip','.safetensors','.bin','.pt','.pth','.gz'}
    with zipfile.ZipFile(prior) as old:
        manifests=[p for p in old.namelist() if p.endswith('/RELEASE_MANIFEST.json')]
        assert len(manifests)==1
        old_prefix=manifests[0].split('/')[0]
        before=json.loads(old.read(manifests[0]))
        names=set(before['files'])|set(overlays)
        assert all(not p.startswith('/') and '..' not in Path(p).parts for p in names)
        total=sum(overlays[p].stat().st_size if p in overlays else old.getinfo(old_prefix+'/'+p).file_size for p in names)
        assert shutil.disk_usage(target.parent).free>total*1.03
        hashes={};print('Packaging',len(names),'files',total,'bytes',flush=True)
        with zipfile.ZipFile(partial,'w',allowZip64=True) as new:
            for i,p in enumerate(sorted(names),1):
                info=zipfile.ZipInfo(prefix+'/'+p)
                info.compress_type=zipfile.ZIP_STORED if Path(p).suffix in stored else zipfile.ZIP_DEFLATED
                digest=hashlib.sha256()
                source=overlays[p].open('rb') if p in overlays else old.open(old_prefix+'/'+p)
                with source,new.open(info,'w',force_zip64=True) as destination:
                    for block in iter(lambda:source.read(BLOCK),b''):digest.update(block);destination.write(block)
                hashes[p]=digest.hexdigest()
                if p not in overlays:assert hashes[p]==before['files'][p],p
                if i%2000==0:print('Written',i,'/',len(names),flush=True)
            historical='historical/contiguous_loss_release_manifest.json'
            data=old.read(manifests[0]);new.writestr(prefix+'/'+historical,data)
            hashes[historical]=hashlib.sha256(data).hexdigest()
            manifest=dict(revision='R8',extension='exact finite-count limitation of pairwise dependence summaries',
                          market_endpoint='2026-08-31',external_endpoint='2026-07-31',
                          preceding_archive_sha256=receipt['archive_sha256'],files=hashes)
            new.writestr(prefix+'/RELEASE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(partial) as new:
        assert len(new.namelist())==len(hashes)+1
        for i,(p,want) in enumerate(hashes.items(),1):
            digest=hashlib.sha256()
            with new.open(prefix+'/'+p) as stream:
                for block in iter(lambda:stream.read(BLOCK),b''):digest.update(block)
            assert digest.hexdigest()==want,p
            if i%2000==0:print('Verified',i,'/',len(hashes),flush=True)
    replay_members=verify_extension(partial,prefix,hashes,check)
    partial.rename(target)
    record=dict(status='complete',archive=str(target.relative_to(PROJECT)),archive_sha256=sha(target),
                files=len(hashes),uncompressed_bytes=total+len(data),archived_bytes_verified=True,
                preceding_archive_sha256=receipt['archive_sha256'],preceding_archive_unchanged=sha(prior)==receipt['archive_sha256'],
                extension_archive_replay_exact=True,extension_replay_members=replay_members)
    assert record['preceding_archive_unchanged']
    (OUT/'full_release.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
