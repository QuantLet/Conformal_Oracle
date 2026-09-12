"""Download public checkpoints at the revisions recorded before the pilot."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]/'artifacts/r8_native_candidates/models'


def fetch(task):
    repo, revision, name, target = task
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_name(target.name+'.partial')
        with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
            while block := response.read(1024*1024): stream.write(block)
        temp.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(target.name, target.stat().st_size, flush=True)
    return str(target.relative_to(ROOT)), digest


if __name__ == '__main__':
    tasks = []; revisions = {}
    for path in sorted(ROOT.glob('*_metadata.json')):
        meta = json.loads(path.read_text()); repo = meta['id']; revision = meta['sha']
        tag = repo.split('/')[-1]; revisions[repo] = revision
        for file in meta['siblings']:
            name = file['rfilename']
            if name.endswith(('.py','.json','.safetensors','.md')):
                tasks.append((repo, revision, name, ROOT/tag/name))
    with ThreadPoolExecutor(max_workers=4) as pool: files = dict(pool.map(fetch,tasks))
    (ROOT/'manifest.json').write_text(json.dumps({'revisions':revisions,'files':files},indent=2)+'\n')
