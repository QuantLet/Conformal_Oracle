"""Read only the official availability page; never infer missing endpoint data."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
DETAILS='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html'
REQUIRED='2026-07-31'


def endpoint(raw):
    text=html.unescape(re.sub('<[^>]+>',' ',raw))
    text=' '.join(text.split())
    match=re.search(r'Daily Returns\s*:?\s*(.{0,180})',text,re.I)
    if not match:raise ValueError('Daily-return metadata not found')
    dates=re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',match.group(1))
    if len(dates)<2:raise ValueError('Unrecognised daily-return date range')
    return datetime.strptime(dates[1].replace(',',''),'%B %d %Y').date().isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--html',type=Path);a=ap.parse_args()
    raw=a.html.read_bytes() if a.html else urllib.request.urlopen(DETAILS,timeout=30).read()
    out=ROOT/'artifacts/r8_external/july2026';out.mkdir(parents=True,exist_ok=True)
    last=endpoint(raw.decode('utf-8',errors='strict'))
    receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'source':DETAILS,
             'required_endpoint':REQUIRED,'advertised_endpoint':last,
             'ready_for_download':last>=REQUIRED,'outcomes_evaluated':False,
             'protocol_sha256':hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest(),
             'source_sha256':hashlib.sha256(raw).hexdigest(),
             'status':'metadata_ready_actual_data_must_still_pass' if last>=REQUIRED else 'waiting_for_required_endpoint'}
    (out/'availability.html').write_bytes(raw)
    (out/'readiness.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
