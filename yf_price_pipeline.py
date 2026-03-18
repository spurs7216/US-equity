
#!/usr/bin/env python3

# yfinance / stooq price downloader
#
# Usage examples:
#   1) Python API
#       from yf_price_pipeline import download_prices_yf, to_returns
#       prices, volumes = download_prices_yf(["AAPL","MSFT"], "2015-01-01", "2022-12-31")
#       rets = to_returns(prices)
#
#   2) CLI (requires a CSV with 'ticker' column; optional 'start_date','end_date' per row)
#       python yf_price_pipeline.py --tickers tickers.csv --start 2015-01-01 --end 2022-12-31 --outdir data/
#
# Notes:
#   - Uses Adj Close (auto_adjust=True) so splits/dividends are handled.
#   - Batches tickers to avoid rate limits; robust retries with backoff.
#   - Saves Parquet files: prices_adj.parquet, volume.parquet, returns.parquet
#   - Optional Stooq fallback via pandas-datareader if yfinance fails for a symbol.

import argparse
import time
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# yfinance is optional at import time so the module can be inspected without the package.
try:
    import yfinance as yf
except Exception:
    yf = None

def _chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns into a wide (date x tickers) frame for a single field.
    If a single-level is already present, pass through.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Example: columns like ('Adj Close','AAPL'), ('Adj Close','MSFT'), etc.
        # Keep lowest level as ticker
        # If top level has multiple fields, the caller should select one before calling.
        df = df.copy()
        # Drop the top level if it's constant
        if len(set(df.columns.get_level_values(0))) == 1:
            df.columns = df.columns.get_level_values(-1)
        else:
            # If mixed fields, pivot to a dict of fields
            raise ValueError("Expected a single field (e.g., 'Adj Close') in columns; got multiple.")
    return df

def _yf_download_field(tickers: List[str], start: str, end: str, field: str,
                       chunk: int = 200, auto_adjust: bool = False,
                       retries: int = 3, sleep_base: float = 1.0,
                       progress: bool = False) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`")
    frames = []
    for batch in _chunked(tickers, chunk):
        last_err = None
        for k in range(retries):
            try:
                df = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=auto_adjust,
                    actions=False,   # we just need prices/volume here
                    group_by="column",   # ensures field-first MultiIndex (yfinance>=0.2)
                    threads=True,
                    progress=progress,
                )
                print(df.columns)
                # Select the field and flatten to (date x tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    if field not in df.columns.get_level_values(0):
                        raise KeyError(f"Field '{field}' not in downloaded data columns.")
                    sub = df[field]
                else:
                    # Single ticker returns single-level columns; yfinance returns legacy format in some versions
                    if field in df.columns:
                        sub = df[[field]].copy()
                        sub.columns = batch  # rename to ticker name for consistency when single ticker
                    else:
                        raise KeyError(f"Field '{field}' not present for tickers {batch}")
                sub = _normalize_columns(sub)
                frames.append(sub)
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_base * (2 ** k))
        else:
            raise RuntimeError(f"yfinance download failed for batch {batch}: {last_err}")
    out = pd.concat(frames, axis=1).sort_index(axis=1)
    # Ensure business-day frequency (NY) — forward-fill missing columns won't be applied; we keep NaNs.
    out = out[~out.index.duplicated(keep="first")]
    return out

def download_prices_yf(
    tickers: Iterable[str],
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    chunk: int = 200,
    auto_adjust: bool = True,
    retries: int = 3,
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download Adj Close and Volume with yfinance, return (prices_adj, volume).
    Both are DataFrames indexed by date with ticker columns.
    """
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]))  # unique, clean
    prices = _yf_download_field(tickers, start, end, "Close", chunk, auto_adjust, retries, progress=progress)
    volume = _yf_download_field(tickers, start, end, "Volume", chunk, auto_adjust, retries, progress=progress)
    sector = _yf_download_field(tickers, start, end, "sector", chunk, auto_adjust, retries, progress=progress)
    industry = _yf_download_field(tickers, start, end, "industry", chunk, auto_adjust, retries, progress=progress)

    return prices, volume, sector, industry

def to_returns(prices_adj: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from adjusted close (handles splits/dividends).
    Leaves NaNs where prices are missing (no back/forward fill).
    """
    return prices_adj.pct_change()

def stooq_fallback(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch one ticker from Stooq via pandas-datareader as a fallback.
    Returns a DataFrame with 'Adj Close' if available, else None.
    """
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(ticker, "stooq", start=start, end=end)
        if "Close" in df.columns and "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Adj Close", "Volume"]].sort_index()
    except Exception:
        return None

def align_to_universe(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Apply a dynamic universe mask.
    universe: DataFrame with the same columns (tickers) and index (dates) containing 1/0 or True/False.
    Returns prices masked (NaN outside active dates).
    """
    # Align indices/columns
    ms = universe.reindex_like(prices).astype(float)
    return prices.where(ms > 0.5)

def cli():
    parser = argparse.ArgumentParser(description="Download prices with yfinance and save Parquet panels.")
    parser.add_argument("--tickers", type=str, required=True,
                        help="CSV file with a 'ticker' column (optionally start_date,end_date per row).")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--cost_bps", type=float, default=5.0)
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    df_t = pd.read_csv(args.tickers)
    if "ticker" not in df_t.columns:
        raise ValueError("CSV must contain a 'ticker' column.")
    tickers = df_t["ticker"].dropna().astype(str).tolist()

    prices, volume = download_prices_yf(tickers, start=args.start, end=args.end, chunk=args.chunk)
    rets = to_returns(prices)

    prices.to_parquet(os.path.join(args.outdir, "prices_adj.parquet"))
    volume.to_parquet(os.path.join(args.outdir, "volume.parquet"))
    rets.to_parquet(os.path.join(args.outdir, "returns.parquet"))
    print(f"Saved to {args.outdir}: prices_adj.parquet, volume.parquet, returns.parquet")

if __name__ == "__main__":
    cli()
