"""Self-contained analytic study with archive-only verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from run import ROOT, OUT, sha


def main():
    before=json.loads((OUT/'before.json').read_text())
    check=json.loads((OUT/'independent_validation.json').read_text())
    primary=json.loads((OUT/'run/validation.json').read_text())
    assert check['status']==primary['status']=='passed'
    assert check['producer_sha256']==sha(ROOT/'research/r8_information_limit/validate.py')
    assert check['proof_sha256']==sha(ROOT/'research/r8_information_limit/PROOF.md')
    assert all(sha(ROOT/p)==h for p,h in before['canonical'].items())
    names=set(before['canonical'])|set(primary['inputs'])
    for folder in [ROOT/'research/r8_information_limit', OUT]:
        names.update(str(p.relative_to(ROOT)) for p in folder.rglob('*') if p.is_file()
                     and not {'__pycache__','.DS_Store'}.intersection(p.parts)
                     and p.name not in ('package.json','package.log'))
    names.add('docs/IRFA_INFORMATION_LIMIT_RESULTS.md')
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in names)
    prefix='R8_20260910_information_limit'
    target=ROOT/'release'/(prefix+'.zip')
    partial=target.with_suffix('.partial.zip')
    assert not target.exists() and not partial.exists(),'Preserve preceding releases'
    hashes={p:sha(ROOT/p) for p in sorted(names)}
    total=sum((ROOT/p).stat().st_size for p in names)
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            compression=zipfile.ZIP_STORED if Path(name).suffix in ('.png','.pdf','.zip') else zipfile.ZIP_DEFLATED
            archive.write(ROOT/name,prefix+'/'+name,compress_type=compression)
        archive.writestr(prefix+'/STUDY_MANIFEST.json',json.dumps(dict(
            kind='Exact information limits for tail correction',files=hashes),indent=2)+'\n')
    with zipfile.ZipFile(partial) as archive,tempfile.TemporaryDirectory(prefix='irfa-information-archive-') as folder:
        for name,wanted in hashes.items():
            data=archive.read(prefix+'/'+name)
            assert hashlib.sha256(data).hexdigest()==wanted
            path=Path(folder)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        for name in ['contiguous.py','validate.py','plot.py','report.py']:
            result=subprocess.run([sys.executable,'research/r8_information_limit/'+name],
                                  cwd=folder,capture_output=True,text=True)
            assert result.returncode==0,(name,result.stdout,result.stderr)
        for name in ['contiguous.json','independent_validation.json','figures.json','report.json']:
            assert json.loads((OUT/name).read_text())==json.loads((Path(folder)/'artifacts/r8_information_limit'/name).read_text()),name
    partial.rename(target)
    record=dict(status='complete',archive=str(target.relative_to(ROOT)),archive_sha256=sha(target),
                members=len(hashes),uncompressed_bytes=total,every_member_verified=True,
                archive_only_validation_exact=True,archive_only_contiguous_exact=True,
                archive_only_figure_exact=True,archive_only_report_exact=True,
                protected_files=len(before['canonical']),canonical_unchanged=True)
    assert all(sha(ROOT/p)==h for p,h in before['canonical'].items())
    (OUT/'package.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
