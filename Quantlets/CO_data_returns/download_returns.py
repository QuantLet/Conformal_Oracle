from pathlib import Path
import json
import yfinance as yf
import pandas as pd
import numpy as np

START_DATE = "2000-01-01"
END_DATE = "2026-03-19"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = REPO_ROOT / "cfp_ijf_data" / "returns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SCRIPT_DIR / "ticker_mapping.json") as f:
    tickers = json.load(f)

results = []
for short_name, ticker in tickers.items():
    print(f"Downloading {short_name} ({ticker})...", end=" ")
    try:
        df = yf.download(
            ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False
        )
        if df.empty:
            print("EMPTY -- SKIPPED")
            results.append((short_name, ticker, 0, None, None))
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].dropna().rename(columns={"Close": "price"})
        df.index.name = "date"
        df["log_return"] = np.log(df["price"] / df["price"].shift(1))
        df = df.dropna()
        df = df[df["log_return"].abs() <= 0.50]

        out_path = OUT_DIR / f"{short_name}.csv"
        df[["log_return"]].to_csv(out_path)

        print(f"{len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
        results.append((short_name, ticker, len(df), df.index.min(), df.index.max()))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append((short_name, ticker, 0, None, None))

print("\n=== Summary ===")
for r in results:
    status = f"{r[2]:5d} rows  {r[3]} -> {r[4]}" if r[2] > 0 else "FAILED"
    print(f"{r[0]:10s} ({r[1]:12s}): {status}")

n_ok = sum(1 for r in results if r[2] > 0)
n_failed = sum(1 for r in results if r[2] == 0)
print(f"\nTotal: {n_ok} OK, {n_failed} failed out of {len(results)}")
