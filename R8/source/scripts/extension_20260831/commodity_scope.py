"""Canonical R8 scope after the authorised commodity-share replacement."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT/'research/r8_commodity_etp'))
from panel_scope import ROOT, DECISION, MODELS, CLASSES

ARCHIVE = ROOT
N_ASSETS = 24
N_PAIRS = N_ASSETS*len(MODELS)
