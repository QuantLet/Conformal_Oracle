"""Complete R8 release with immutable research and current document validation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from build import PROJECT, OUT

BLOCK = 4 * 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_extension(path, prefix, hashes, check):
    names = set(check['scope']['current_files'])
    names.update(p for p in hashes if p.startswith((
        'research/r8_partial_integration/', 'artifacts/r8_partial_integration/',
        'research/r8_partial_shift/', 'artifacts/r8_partial_shift/')))
    study = json.loads((PROJECT / 'artifacts/r8_partial_shift/run/validation.json').read_text())
    names.update(study['inputs'])
    protected = json.loads((PROJECT / 'artifacts/r8_partial_shift/before.json').read_text())['canonical']
    names.update(protected)
    old_scope = json.loads((PROJECT / 'artifacts/r8_count_law/final_validation.json').read_text())['current_manuscript']
    names.update(old_scope)
    names.update(['artifacts/r8_count_law/final_validation.json',
                  'release/R8_20260910_partial_shift_study.zip'])
    names.update('source/' + doc + '.log' for doc in ('main_R8', 'supplement_R8'))
    assert names.issubset(hashes), sorted(names - set(hashes))
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix='irfa-partial-release-') as folder:
        for name in names:
            target = Path(folder) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(prefix + '/' + name)
            assert hashlib.sha256(content).hexdigest() == hashes[name]
            target.write_bytes(content)
        run = subprocess.run([sys.executable, 'research/r8_partial_integration/validate.py', '--replay-study'],
                             cwd=folder, capture_output=True, text=True)
        assert run.returncode == 0, (run.stdout, run.stderr)
        replay = json.loads((Path(folder) / 'artifacts/r8_partial_integration/final_validation.json').read_text())
        assert replay == check, 'Archive-only integration replay differs'
    return len(names)


def main():
    previous = json.loads((PROJECT / 'artifacts/r8_count_law/full_release.json').read_text())
    assert previous['status'] == 'complete'
    prior = PROJECT / previous['archive']
    assert sha(prior) == previous['archive_sha256']
    check = json.loads((OUT / 'final_validation.json').read_text())
    assert check['status'] == 'passed'
    for name, expected in check['scope']['current_files'].items():
        assert sha(PROJECT / name) == expected, name
    prefix = 'R8_20260910_partial_integration'
    target = PROJECT / 'release' / (prefix + '.zip')
    partial = target.with_suffix('.partial.zip')
    assert not target.exists() and not partial.exists(), 'Preserve preceding releases'
    overlays = {}
    for root in ['research/r8_partial_shift', 'artifacts/r8_partial_shift',
                 'research/r8_partial_integration', 'artifacts/r8_partial_integration']:
        for path in (PROJECT / root).rglob('*'):
            if path.is_file() and not {'__pycache__', '.DS_Store'}.intersection(path.parts) \
                    and path.name not in ('full_release.json', 'package_release.log'):
                overlays[str(path.relative_to(PROJECT))] = path
    extras = set(check['scope']['current_files'])
    extras.update(['docs/IRFA_PARTIAL_SHIFT_RESULTS.md', 'docs/IRFA_PARTIAL_INTEGRATION_VALIDATION.md',
                   'docs/IRFA_REVIEW_STATE.md', 'artifacts/r8_count_law/full_release.json',
                   'release/R8_20260910_partial_shift_study.zip',
                   'release/R8_20260910_commodity_sources.zip'])
    study = json.loads((PROJECT / 'artifacts/r8_partial_shift/run/validation.json').read_text())
    extras.update(study['inputs'])
    extras.update(json.loads((PROJECT / 'artifacts/r8_partial_shift/independent_validation.json').read_text())['additional_inputs'])
    for doc in ('main_R8', 'supplement_R8'):
        extras.update('source/' + doc + '.' + ext for ext in ['tex', 'pdf', 'aux', 'log', 'bbl'])
    extras.update(str(path.relative_to(PROJECT)) for path in
                  (PROJECT / 'artifacts/r8_ten_integration').glob('portable_*.log'))
    for name in extras:
        overlays[name] = PROJECT / name
    overlays['README.md'] = PROJECT / 'research/r8_partial_integration/README.md'
    stored = {'.npz', '.npy', '.png', '.pdf', '.parquet', '.zip', '.safetensors', '.bin', '.pt', '.pth', '.gz'}
    with zipfile.ZipFile(prior) as old:
        manifest_name, = [name for name in old.namelist() if name.endswith('/RELEASE_MANIFEST.json')]
        old_prefix = manifest_name.split('/')[0]
        manifest = json.loads(old.read(manifest_name))
        names = set(manifest['files']) | set(overlays)
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
        total = sum(overlays[name].stat().st_size if name in overlays
                    else old.getinfo(old_prefix + '/' + name).file_size for name in names)
        assert shutil.disk_usage(target.parent).free > total * 1.03
        print('Packaging', len(names), 'files;', total, 'bytes', flush=True)
        hashes = {}
        with zipfile.ZipFile(partial, 'w', allowZip64=True) as new:
            for i, name in enumerate(sorted(names), 1):
                info = zipfile.ZipInfo(prefix + '/' + name)
                info.compress_type = zipfile.ZIP_STORED if Path(name).suffix in stored else zipfile.ZIP_DEFLATED
                digest = hashlib.sha256()
                source = overlays[name].open('rb') if name in overlays else old.open(old_prefix + '/' + name)
                with source, new.open(info, 'w', force_zip64=True) as destination:
                    for block in iter(lambda: source.read(BLOCK), b''):
                        digest.update(block)
                        destination.write(block)
                hashes[name] = digest.hexdigest()
                if name not in overlays:
                    assert hashes[name] == manifest['files'][name], name
                if i % 3000 == 0:
                    print('Written', i, '/', len(names), flush=True)
            historical = 'historical/count_law_release_manifest.json'
            content = old.read(manifest_name)
            new.writestr(prefix + '/' + historical, content)
            hashes[historical] = hashlib.sha256(content).hexdigest()
            record = dict(revision='R8', extension='partial-shift development study integrated',
                          market_endpoint='2026-08-31', external_endpoint='2026-07-31',
                          preceding_archive_sha256=previous['archive_sha256'], files=hashes)
            new.writestr(prefix + '/RELEASE_MANIFEST.json', json.dumps(record, indent=2) + '\n')
    with zipfile.ZipFile(partial) as archive:
        assert len(archive.namelist()) == len(hashes) + 1
        for i, (name, expected) in enumerate(hashes.items(), 1):
            digest = hashlib.sha256()
            with archive.open(prefix + '/' + name) as stream:
                for block in iter(lambda: stream.read(BLOCK), b''):
                    digest.update(block)
            assert digest.hexdigest() == expected, name
            if i % 3000 == 0:
                print('Verified', i, '/', len(hashes), flush=True)
    replay_members = verify_extension(partial, prefix, hashes, check)
    partial.rename(target)
    receipt = dict(status='complete', archive=str(target.relative_to(PROJECT)), archive_sha256=sha(target),
                   files=len(hashes), uncompressed_bytes=total + len(content), every_member_verified=True,
                   preceding_archive_sha256=previous['archive_sha256'],
                   preceding_archive_unchanged=sha(prior) == previous['archive_sha256'],
                   archive_only_integration_replay_exact=True, archive_replay_members=replay_members,
                   historical_study_validation_repeated=True, historical_study_archive_included=True)
    assert receipt['preceding_archive_unchanged']
    (OUT / 'full_release.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
