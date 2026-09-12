"""Fixed 21 unchanged plus three commodity-share exposures."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT/'source/scripts/extension_20260831'
sys.path.insert(0, str(SCRIPTS))
from panel_statistics import MODELS as ORIGINAL_MODELS

OLD = PROJECT/'artifacts/extension_20260831'
NEW = PROJECT/'artifacts/r8_commodity_etp'
ART = NEW/'panel'
ROOT = ART/'base'
DECISION = ART/'decision'
REPLACEMENTS = {'WTI':'USO', 'GOLD':'GLD', 'NATGAS':'UNG'}
MODELS = {m:s for m,s in ORIGINAL_MODELS.items() if m not in ['TimesFM-2.5', 'Moirai-2.0']}
CLASSES = {'Equity':['ASX200','BOVESPA','FCHI','FTSE100','GDAXI','HSI','ICLN','NIFTY','NIKKEI','SP500','STOXX'],
           'Bond ETF':['CBU0','IBGL','TLT'], 'Commodity':['DJCI','GLD','UNG','USO'],
           'Crypto':['BTC','ETH'], 'FX':['AUDUSD','EURUSD','GBPUSD','USDJPY']}
CLASS = {a:c for c,group in CLASSES.items() for a in group}
ASSETS = sorted(CLASS)


def source(asset):
    return NEW if asset in REPLACEMENTS.values() else OLD


def controlled(asset):
    return NEW/'controlled' if asset in REPLACEMENTS.values() else PROJECT/'artifacts/review_20260909/controlled'


def decisions(asset):
    return NEW/'decision/pairs' if asset in REPLACEMENTS.values() else PROJECT/'artifacts/r8_decision/pairs'
