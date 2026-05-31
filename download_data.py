#!/usr/bin/env python3
"""
download_data.py — fetch the canonical dataset for the Conformal_Oracle Quantlets.

Only the 24 return series are committed to this repository. The ~126 MB of
TSFM/benchmark forecast parquets and pre-computed paper_outputs/ tables are
published as a GitHub Release asset (keeping the code repo lean). Run this script
once to download and unpack them into ``cfp_ijf_data/`` before running the
table/figure Quantlets.

Usage
-----
    python download_data.py                 # download + extract into ./cfp_ijf_data
    python download_data.py --dest /tmp/d   # extract elsewhere
    python download_data.py --url <zip-url>  # override the archive URL

Requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset published as a GitHub Release asset. To (re)publish, from the repo root:
#   zip -r cfp_ijf_data.zip cfp_ijf_data -x '*.DS_Store'
#   gh release create data-v1 cfp_ijf_data.zip --repo QuantLet/Conformal_Oracle \
#       --title "Conformal_Oracle dataset" --notes "Forecast parquets + tables."
RELEASE_PAGE = "https://github.com/QuantLet/Conformal_Oracle/releases/tag/data-v1"
ARCHIVE_URL  = "https://github.com/QuantLet/Conformal_Oracle/releases/download/data-v1/cfp_ijf_data.zip"
ARCHIVE_SHA256 = "a6349b795b21fd7bdb63ea39d927f1671f5dedacbc64763b3d26c88b5a7fd98f"
# ---------------------------------------------------------------------------

DEFAULT_DEST = Path(__file__).resolve().parent / "cfp_ijf_data"


def _download(url: str, out: Path) -> None:
    print(f"Downloading {url}")
    print(f"        ->  {out}")

    def _hook(block, bsize, total):
        if total > 0:
            done = min(block * bsize, total)
            pct = 100 * done / total
            sys.stdout.write(f"\r  {pct:5.1f}%  ({done/1e6:.1f}/{total/1e6:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, out, reporthook=_hook)  # noqa: S310
    sys.stdout.write("\n")


def _verify(path: Path, expected: str) -> None:
    if not expected:
        return
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise SystemExit(f"Checksum mismatch:\n  expected {expected}\n  got      {got}")
    print("Checksum OK")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # Archive is expected to contain a top-level cfp_ijf_data/ directory, so we
    # extract into dest's parent and let the folder land in place.
    target_root = dest.parent
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_root)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix in {".tgz", ".gz"}:
        with tarfile.open(archive) as tf:
            tf.extractall(target_root)  # noqa: S202
    else:
        raise SystemExit(f"Unsupported archive type: {archive.name}")
    print(f"Extracted into {target_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=ARCHIVE_URL, help="archive download URL")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="extraction target (cfp_ijf_data)")
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the downloaded archive")
    args = ap.parse_args()

    only_returns = args.dest / "returns"
    if args.dest.exists() and any(p for p in args.dest.iterdir() if p != only_returns):
        print(f"{args.dest} already has data beyond returns/ — nothing to do.")
        print(f"(Release: {RELEASE_PAGE})")
        return 0

    archive = args.dest.parent / "cfp_ijf_data_download.zip"
    _download(args.url, archive)
    _verify(archive, ARCHIVE_SHA256)
    _extract(archive, args.dest)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
