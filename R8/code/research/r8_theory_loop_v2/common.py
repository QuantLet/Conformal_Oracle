"""Input binding shared by v2 entry points; no computations on import."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/r8_theory_loop_v2'
OUT = ROOT/'results/theory_loop_v2'
LOCK = OUT/'protocol_lock.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def binding(path):
    path=Path(path).resolve(); st=path.stat()
    return dict(path=str(path),sha256=sha(path),mtime_ns=st.st_mtime_ns,size=st.st_size)

def validate_lock(path=LOCK):
    lock=json.loads(Path(path).read_text())
    assert sha(lock['protocol_path'])==lock['protocol_sha256'], 'Protocol digest mismatch'
    for rec in lock['input_files']+lock.get('window_files',[]):
        got=binding(rec['path'])
        assert all(got[k]==rec[k] for k in ('sha256','mtime_ns','size')), 'Input binding mismatch: '+rec['path']
    if 'metadata_sha256' in lock:
        assert sha(lock['metadata_path'])==lock['metadata_sha256'], 'Metadata digest mismatch'
    return lock

def save_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
