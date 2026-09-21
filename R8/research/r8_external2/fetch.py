"""Archive the public source responses for the Developed ex-US 25 ME x BE-ME daily universe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from readiness import endpoint

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/'artifacts/r8_external2/devexus'
# The protocol's guessed details-page name returns HTTP 404 (checked 13 September 2026);
# the library index links this page as "Details" for the Developed ex-US 25 daily row.
PROTOCOL_DETAILS = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_25_port_form_sz_bm_daily_dev_ex_us.html'
DETAILS = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_developed.html'
DATA = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip'
REQUIRED = '2025-12-31'


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


def probe_protocol_page():
    receipt = OUT/'raw'/'protocol_details_page_probe.json'
    if receipt.exists(): return
    req = urllib.request.Request(PROTOCOL_DETAILS, headers={'User-Agent':'IRFA-research-replication/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=45) as r: status = r.status
    except urllib.error.HTTPError as e: status = e.code
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(source=PROTOCOL_DETAILS, status=status, checked_utc=datetime.now(timezone.utc).isoformat(),
                                       used_details_page=DETAILS), indent=2)+'\n')
    print('protocol details page status', status, flush=True)


if __name__ == '__main__':
    probe_protocol_page()
    raw = get('availability.html', DETAILS)
    advertised = endpoint(raw.decode('utf-8', errors='replace'))
    print('advertised daily endpoint', advertised, flush=True)
    assert advertised >= REQUIRED, advertised
    get('Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip', DATA)
