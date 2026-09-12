"""Archive immutable public sources/checkpoints for the two grid pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / 'artifacts/r8_grid_candidates'
MODELS = {
    'patchtst': ('ibm-granite/granite-timeseries-patchtst-fm-r1', 'ibm-granite/granite-tsfm',
                ['README.md', 'config.json', 'model.safetensors']),
    'tsicl': ('taharnbl/TS-ICL', 'EDF-Lab/ts-icl',
              ['README.md', 'config.json', 'LICENSE', 'tsicl-v1.ckpt']),
}


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'IRFA-public-source-review'})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open('wb') as dest:
        while data := source.read(4 * 1024 * 1024):
            dest.write(data)
            digest.update(data)
    temporary.replace(path)
    return {'url': url, 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['source', 'weights'], required=True)
    args = parser.parse_args()
    manifest = {'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files': {}}
    for name, (hf, repo, filenames) in MODELS.items():
        hf_sha = json.loads((ROOT / 'source_review' / f'{name}_hf.json').read_text())['sha']
        git_sha = json.loads((ROOT / 'source_review' / f'{name}_git.json').read_text())['sha']
        items = [(f'https://codeload.github.com/{repo}/tar.gz/{git_sha}', ROOT / 'sources' / f'{name}-{git_sha}.tar.gz')]
        items += [(f'https://huggingface.co/{hf}/resolve/{hf_sha}/{f}', ROOT / 'models' / name / f)
                  for f in filenames if not f.endswith(('.safetensors', '.ckpt'))]
        if args.kind == 'weights':
            items = [(f'https://huggingface.co/{hf}/resolve/{hf_sha}/{f}', ROOT / 'models' / name / f)
                     for f in filenames if f.endswith(('.safetensors', '.ckpt'))]
        for url, path in items:
            entry = fetch(url, path)
            manifest['files'][str(path.relative_to(ROOT))] = entry
            print(name, path.name, entry['bytes'], flush=True)
    (ROOT / f'{args.kind}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
