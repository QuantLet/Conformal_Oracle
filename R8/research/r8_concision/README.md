# R8 shortened manuscript

This editorial revision reduces the article to 41 pages. It retains the
preceding numerical study, formal statements and proofs unchanged. The
supplement remains 32 pages.

Run the existing document/display guard with the recorded Python environment:

    python source/scripts/extension_20260831/validate_r8.py

Build both TeX documents twice with `latexmk -g -pdf`, then run
`python research/r8_ten_integration/package_sources.py` to rebuild the
66-member portable source package and compare its PDF text independently.
Finally run `python research/r8_concision/validate.py` after visual review.
These commands check the editorial revision; the inherited numerical
reconstruction instructions and historical receipts remain in the archive.
