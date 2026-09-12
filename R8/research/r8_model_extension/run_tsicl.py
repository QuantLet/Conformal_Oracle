"""Run independent assets and then replay each in the recreated environment."""
import concurrent.futures
import subprocess
from pathlib import Path
from scope import ROOT, ASSETS, dump

PRODUCTION = Path('/private/tmp/irfa-grid-tsicl/bin/python')
REPLAY = Path('/private/tmp/irfa-grid-tsicl-recreated/bin/python')


def work(asset):
    folder = ROOT/'tsicl_full'/asset; folder.mkdir(parents=True,exist_ok=True)
    for phase, python in [('production',PRODUCTION),('replay',REPLAY)]:
        cmd = [str(python),str(Path(__file__).with_name('tsicl.py')),'--asset',asset]
        if phase == 'replay': cmd.append('--replay')
        with (folder/f'{phase}.log').open('a') as stream:
            result = subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{asset} {phase} failed; see {folder/ (phase+".log")}')
    return asset


def main():
    assert PRODUCTION.exists() and REPLAY.exists()
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(work,asset) for asset in ASSETS]
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result()); dump(ROOT/'tsicl_progress.json',dict(completed=sorted(done),total=len(ASSETS)))
            print('Production and exact replay complete:',done[-1],len(done),'/',len(ASSETS),flush=True)


if __name__ == '__main__': main()
