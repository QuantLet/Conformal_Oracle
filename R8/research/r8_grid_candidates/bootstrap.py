"""Recreate one pilot environment from its archived source and package lock."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]/'artifacts/r8_grid_candidates'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['patchtst', 'tsicl'], required=True)
    parser.add_argument('--env-dir', type=Path, required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3,12), 'Use Python 3.12; original execution used 3.12.14'
    assert not args.env_dir.exists(), 'Choose a fresh environment directory'
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    name, item = next((name, item) for name, item in manifest['files'].items()
                      if name.startswith('sources/'+args.model+'-'))
    archive = ROOT/name
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == item['sha256']
    extracted = archive.parent/archive.name.removesuffix('.tar.gz')
    if not extracted.exists():
        extracted.mkdir()
        with tarfile.open(archive) as source: source.extractall(extracted, filter='data')
    subprocess.run([sys.executable, '-m', 'venv', str(args.env_dir)], check=True)
    lock = ROOT/'source_review'/f'{args.model}_requirements.lock'
    # The local project reference in pip freeze is host-specific. Runtime
    # imports use the verified source archive, so install its dependency lock.
    dependencies = [line for line in lock.read_text().splitlines()
                    if line and not line.startswith('#') and ' @ file:' not in line]
    portable_lock = args.env_dir/'dependencies.lock'
    portable_lock.write_text('\n'.join(dependencies)+'\n')
    python = args.env_dir/'bin/python'
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '-r', str(portable_lock)], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    print('Runtime ready; inference imports the pinned local source directly.')


if __name__ == '__main__': main()
