"""Restore only the archived pre-integration sources in a disposable project copy."""
import argparse
from pathlib import Path
from zipfile import ZipFile
from build import PROJECT,OUT


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('copy',type=Path);dest=ap.parse_args().copy.resolve()
    assert dest.is_dir() and dest!=PROJECT and PROJECT not in dest.parents,'Use an existing disposable copy outside this project'
    with ZipFile(OUT/'before_source.zip') as z:
        for name in z.namelist():
            target=(dest/name).resolve();assert dest in target.parents
            target.parent.mkdir(parents=True,exist_ok=True)
            # Unlink protects the original if the disposable copy used hardlinks.
            if target.exists():target.unlink()
            target.write_bytes(z.read(name))
    print('Restored the pre-integration sources only in',dest)
