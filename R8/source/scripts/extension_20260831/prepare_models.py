#!/usr/bin/env python3
"""Freeze public model revisions and source code before inference."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[3]
REPOS = {'moirai': 'Salesforce/moirai-1.1-R-small', 'moirai2': 'Salesforce/moirai-2.0-R-small',
         'lagllama': 'time-series-foundation-models/Lag-Llama', 'timesfm25': 'google/timesfm-2.5-200m-pytorch'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT / 'artifacts/extension_20260831')
    ap.add_argument('--models', nargs='+', default=list(REPOS))
    a = ap.parse_args()
    out = a.root / 'models'
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    api = HfApi()
    for name in a.models:
        repo = REPOS[name]
        revision = records.get(name, {}).get('revision') or api.model_info(repo).sha
        records[name] = dict(repo_id=repo, revision=revision)
        manifest.write_text(json.dumps(records, indent=2)+'\n')
        snapshot_download(repo_id=repo, revision=revision, local_dir=out/name,
                          allow_patterns=['*.json', '*.safetensors', '*.ckpt', 'README.md', 'LICENSE*'])
        files = [p for p in (out/name).rglob('*') if p.is_file() and '.cache' not in p.parts]
        records[name]['files'] = {str(p.relative_to(out/name)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        manifest.write_text(json.dumps(records, indent=2)+'\n')
        print(name, revision, len(files), 'files frozen', flush=True)
    if 'lagllama_source' not in records:
        url = 'https://api.github.com/repos/time-series-foundation-models/lag-llama/commits/main'
        req = urllib.request.Request(url, headers={'User-Agent': 'IRFA-reproduction'})
        commit = json.load(urllib.request.urlopen(req, timeout=30))['sha']
        archive = out / 'lag-llama-source.tar.gz'
        urllib.request.urlretrieve(f'https://codeload.github.com/time-series-foundation-models/lag-llama/tar.gz/{commit}', archive)
        target = out / 'lag-llama-source'
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                rel = Path(*Path(member.name).parts[1:])
                if not member.isfile() or not rel.parts:
                    continue
                if '..' in rel.parts or rel.is_absolute():
                    raise ValueError('Unsafe archive member')
                p = target / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(tf.extractfile(member).read())
        records['lagllama_source'] = dict(repo='https://github.com/time-series-foundation-models/lag-llama',
                                         commit=commit, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
