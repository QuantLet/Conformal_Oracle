#!/usr/bin/env python3
"""Run unchanged fitted-method producers on the two staged return histories."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from panel_statistics import ROOT,MODELS
TARGET=ROOT/'unfiltered_repair'


def run(stage,asset,model=None):
    if stage=='dynamic':
        import dynamic
        # The original producer resolves its output relative to a project
        # root. This local fixture redirects only that output directory.
        base=TARGET/'runtime_root';parent=base/'artifacts';parent.mkdir(parents=True,exist_ok=True)
        link=parent/'extension_20260831'
        try:link.symlink_to('../..',target_is_directory=True)
        except FileExistsError:pass
        dynamic.ROOT=base
        return dynamic.work(asset)
    if stage=='evt':
        import evt_fhs
        evt_fhs.ROOT=TARGET
        return evt_fhs.work(asset)
    import panel_statistics,posthoc
    panel_statistics.ROOT=TARGET;posthoc.ROOT=TARGET
    return posthoc.work(model,asset)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dynamic','evt','posthoc'],required=True);a=ap.parse_args()
    tasks=[(a.stage,asset,model) for asset in ['ETH','NATGAS'] for model in (MODELS if a.stage=='posthoc' else [None])]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(run,*t) for t in tasks]):f.result()
    print(a.stage,'two-asset repair complete',flush=True)
