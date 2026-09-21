"""Continue this computation when all current TS-ICL assets finish replay."""
import json
import subprocess
import time
from pathlib import Path
from scope import ROOT, ASSETS, dump

PYTHON = '/private/tmp/irfa-r8-conda-clean/bin/python'


def main():
    state = ROOT/'ten_pipeline.json'
    while True:
        ready = []
        for asset in ASSETS:
            folder = ROOT/'tsicl_full'/asset
            replay = folder/'replay.json'
            if replay.exists():
                receipt = json.loads(replay.read_text())
                assert receipt['status'] == 'complete' and receipt['fresh_replay_exact']
                ready.append(asset)
            for phase in ['production','replay']:
                log = folder/f'{phase}.log'
                if log.exists() and 'Traceback (most recent call last)' in log.read_text():
                    dump(state,dict(status='failed',asset=asset,phase=phase,log=str(log)))
                    raise RuntimeError(f'{asset} {phase} failed')
        dump(state,dict(status='waiting_for_native_replay',complete=len(ready),assets=24))
        if len(ready) == len(ASSETS): break
        time.sleep(15)
    stages = [('assemble_ten.py','preparation_ten.json'),
              ('evaluate_ten.py','ten_common_evaluation/complete.json'),
              ('validate_ten.py','ten_common_evaluation/validation.json'),
              ('policy_ten.py','ten_policy_evaluation/complete.json'),
              ('validate_policy_ten.py','ten_policy_evaluation/validation.json')]
    for script,receipt in stages:
        dump(state,dict(status='running',stage=script))
        if not (ROOT/receipt).exists():
            with (ROOT/(script+'.log')).open('a') as stream:
                result = subprocess.run([PYTHON,str(Path(__file__).with_name(script))],stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                dump(state,dict(status='failed',stage=script,returncode=result.returncode))
                raise RuntimeError(f'{script} failed')
    dump(state,dict(status='complete',models=10,pairs=240,
        scope='Native 1% comparison and unchanged past-loss policy; canonical manuscript integration remains separate.'))
    print('Ten-model evaluation and independent policy validation complete.',flush=True)


if __name__ == '__main__': main()
