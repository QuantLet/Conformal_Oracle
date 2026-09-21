"""Bind the external completion verdict to all actual validated files."""
from datetime import datetime,timezone
import json
from pathlib import Path
from prepare import PROJECT,OUT,sha


def main():
    reports={n:json.loads((OUT/n).read_text()) for n in ['admission.json','base_validation.json','correction_validation.json','aggregation_validation.json']}
    assert all(x['status']=='passed' for x in reports.values())
    results=json.loads((OUT/'results/complete.json').read_text())
    assert results['base_validation_sha256']==sha(OUT/'base_validation.json')
    assert results['correction_validation_sha256']==sha(OUT/'correction_validation.json')
    assert reports['aggregation_validation.json']['result_complete_sha256']==sha(OUT/'results/complete.json')
    for n,h in results['outputs'].items():assert sha(OUT/'results'/n)==h
    for key,h in results['pair_receipts'].items():
        assert sha(OUT/'pairs'/key/'complete.json')==h
        assert reports['correction_validation.json']['pair_receipts'][key]==h
    for relative,h in reports['base_validation.json']['receipts'].items():assert sha(OUT/relative)==h
    amendment=json.loads((OUT/'amendment.json').read_text())
    assert not amendment['external_outcomes_evaluated'] and not amendment['external_returns_downloaded']
    assert sha(PROJECT/'research/r8_external/PROTOCOL.md')==reports['admission.json']['protocol_sha256']
    inputs={p:sha(PROJECT/p) for p in amendment['sources']}
    assert inputs==amendment['sources']
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json')
    files+=sorted(p for p in (PROJECT/'research/r8_external').iterdir() if p.is_file())
    files+= [PROJECT/p for p in amendment['sources'] if PROJECT/p not in files]
    record=dict(status='complete',verified_utc=datetime.now(timezone.utc).isoformat(),producer_sha256=sha(__file__),
                external_endpoint='2026-07-31',models=4,assets=12,pairs=48,
                primary_decisions=json.loads((OUT/'results/primary_decisions.json').read_text()),
                checks={n:sha(OUT/n) for n in reports},files={str(p.relative_to(PROJECT)):sha(p) for p in files})
    (OUT/'completion.json').write_text(json.dumps(record,indent=2)+'\n')
    print(dict(status='complete',files=len(record['files']),pairs=48),flush=True)


if __name__=='__main__':main()
