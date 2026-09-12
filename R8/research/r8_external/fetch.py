"""Archive the public source responses under the author-approved July endpoint."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
from readiness import endpoint

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/'artifacts/r8_external/july2026'
DETAILS = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html'
DATA = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(name, url):
    target = OUT/'raw'/name
    receipt = target.with_suffix(target.suffix+'.json')
    if target.exists() or receipt.exists():
        old = json.loads(receipt.read_text())
        assert sha(target) == old['sha256'] and old['source'] == url
        return target.read_bytes()
    req = urllib.request.Request(url, headers={'User-Agent':'IRFA-research-replication/1.0'})
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
        meta = dict(source=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    status=r.status, headers=dict(r.headers), bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), producer_sha256=sha(__file__),
                    protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    receipt.write_text(json.dumps(meta, indent=2)+'\n')
    print(name, len(raw), meta['sha256'], flush=True)
    return raw


if __name__ == '__main__':
    amendment = json.loads((OUT/'amendment.json').read_text())
    assert amendment['new_endpoint'] == '2026-07-31'
    raw = get('availability.html', DETAILS)
    assert endpoint(raw.decode('utf-8')) >= amendment['new_endpoint']
    get('12_Industry_Portfolios_daily_CSV.zip', DATA)
