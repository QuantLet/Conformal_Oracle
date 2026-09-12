"""Self-contained research archive with an isolated validation/report replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from run import ROOT, OUT, sha


def main():
    study = json.loads((OUT/'run/validation.json').read_text())
    audit = json.loads((OUT/'independent_validation.json').read_text())
    before = json.loads((OUT/'before.json').read_text())
    assert audit['status'] == study['status'] == 'passed'
    assert audit['producer_sha256'] == sha(ROOT/'research/r8_partial_shift/validate.py')
    assert audit['study_validation_sha256'] == sha(OUT/'run/validation.json')
    findings = json.loads((OUT/'findings.json').read_text())['0.01']
    state = ' '.join((ROOT/'docs/IRFA_REVIEW_STATE.md').read_text().split('## R8 finite-count result validated')[0].split())
    expected = f"in {findings['selected_better_than_full']}/72 configurations but over Raw in only {findings['selected_better_than_raw']}/72."
    assert expected in state, 'Research-state counts differ from the final numerical report'
    for bindings in [study['inputs'], audit['additional_inputs'], before['canonical']]:
        assert all(sha(ROOT/p) == h for p,h in bindings.items())
    names = set(study['inputs']) | set(audit['additional_inputs']) | set(before['canonical'])
    for directory in [ROOT/'research/r8_partial_shift', OUT]:
        names.update(str(p.relative_to(ROOT)) for p in directory.rglob('*') if p.is_file()
                     and not {'__pycache__','.DS_Store'}.intersection(p.parts)
                     and p.name not in ('package.json','package.log'))
    names.update(['docs/IRFA_PARTIAL_SHIFT_RESULTS.md','docs/IRFA_REVIEW_STATE.md'])
    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in names)
    prefix = 'R8_20260910_partial_shift_study'
    target = ROOT/'release'/f'{prefix}.zip'
    partial = target.with_suffix('.partial.zip')
    assert not target.exists() and not partial.exists(), 'Preserve existing releases'
    hashes = {p:sha(ROOT/p) for p in sorted(names)}
    total = sum((ROOT/p).stat().st_size for p in names)
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            compression = zipfile.ZIP_STORED if Path(name).suffix in ('.npz','.parquet','.png','.pdf','.zip') else zipfile.ZIP_DEFLATED
            archive.write(ROOT/name,prefix+'/'+name,compress_type=compression)
        manifest = dict(kind='R8 partial-shift development study',files=hashes)
        archive.writestr(prefix+'/STUDY_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(partial) as archive,tempfile.TemporaryDirectory(prefix='irfa-partial-archive-') as tmp:
        for name,want in hashes.items():
            data = archive.read(prefix+'/'+name)
            assert hashlib.sha256(data).hexdigest() == want
            file = Path(tmp)/name; file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
        for command in ['validate.py','plot.py','report.py']:
            process = subprocess.run([sys.executable,'research/r8_partial_shift/'+command],cwd=tmp,capture_output=True,text=True)
            assert process.returncode == 0,(command,process.stdout,process.stderr)
        assert json.loads((Path(tmp)/'artifacts/r8_partial_shift/independent_validation.json').read_text()) == audit
        for receipt in ['figures.json','report.json']:
            source = json.loads((OUT/receipt).read_text())
            replay = json.loads((Path(tmp)/'artifacts/r8_partial_shift'/receipt).read_text())
            assert source == replay, receipt
    partial.rename(target)
    record = dict(status='complete',archive=str(target.relative_to(ROOT)),archive_sha256=sha(target),
                  members=len(hashes),uncompressed_bytes=total,every_member_verified=True,
                  isolated_numerical_validation_exact=True,isolated_figures_exact=True,isolated_report_exact=True,
                  research_state_counts_match=True,
                  canonical_unchanged=all(sha(ROOT/p)==h for p,h in before['canonical'].items()))
    assert record['canonical_unchanged']
    (OUT/'package.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
