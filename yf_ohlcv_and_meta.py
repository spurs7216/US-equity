
import argparse
import time
from typing import Iterable, List, Dict, Tuple, Optional
import os

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _chunked(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    if isinstance(df.columns, pd.MultiIndex):
        if len(set(df.columns.get_level_values(0))) == 1:
            df.columns = df.columns.get_level_values(-1)
        else:
            raise ValueError("Expected a single field per DataFrame; got multiple fields.")
    return df

def _extract_field(df: pd.DataFrame, field: str, batch: List[str]) -> Optional[pd.DataFrame]:
    if isinstance(df.columns, pd.MultiIndex):
        top = set(df.columns.get_level_values(0))
        if field not in top:
            return None
        sub = df[field].copy()
        return _normalize_columns(sub)
    else:
        # Single ticker legacy: columns like ['Open','High','Low','Close','Adj Close','Volume'] (subset)
        if field not in df.columns:
            return None
        sub = df[[field]].copy()
        # Name the column as the ticker symbol (batch has one ticker)
        if len(batch) == 1:
            sub.columns = [batch[0]]
        return sub

def _download_batch(batch: List[str], start: str, end: str, auto_adjust: bool,
                    retries: int, progress: bool) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is not installed. pip install yfinance")
    last_err = None
    for k in range(retries):
        try:
            df = yf.download(
                tickers=batch,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=auto_adjust,
                actions=False,
                group_by="column",
                threads=True,
                progress=progress,
            )
            out = {}
            for f in FIELDS:
                sub = _extract_field(df, f, batch)
                if sub is not None:
                    out[f] = sub
            # If auto_adjust=True, no 'Adj Close' typically; set it equal to 'Close' for convenience
            if auto_adjust and "Adj Close" not in out and "Close" in out:
                out["Adj Close"] = out["Close"].copy()
            # Basic sanity: ensure index is unique
            for f in out:
                tmp = out[f]
                out[f] = tmp[~tmp.index.duplicated(keep="first")]
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (2 ** k))
    raise RuntimeError(f"yfinance download failed for batch {batch}: {last_err}")

def download_ohlcv_yf(
    tickers: Iterable[str],
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    chunk: int = 200,
    auto_adjust: bool = True,
    retries: int = 3,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))  # unique
    # Collect per-field frames then concat across batches
    accum: Dict[str, List[pd.DataFrame]] = {k: [] for k in ["Open","High","Low","Close","Adj Close","Volume"]}
    for batch in _chunked(tickers, chunk):
        out = _download_batch(batch, start, end, auto_adjust, retries, progress)
        for f, df in out.items():
            accum[f].append(df)
    panels: Dict[str, pd.DataFrame] = {}
    for f, lst in accum.items():
        if lst:
            panels[f] = pd.concat(lst, axis=1).sort_index(axis=1)
    # Normalize keys to lowercase API
    final = {}
    final['open'] = panels.get('Open')
    final['high'] = panels.get('High')
    final['low'] = panels.get('Low')
    final['close'] = panels.get('Close')
    final['adj_close'] = panels.get('Adj Close', panels.get('Close'))
    final['volume'] = panels.get('Volume')
    return final

def to_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    return close_prices.pct_change()

def align_to_universe(panel: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    return panel.reindex_like(universe_mask).where(universe_mask.astype(float) > 0.5)

def fetch_metadata_yf(
    tickers: Iterable[str],
    cache_path: Optional[str] = None,
    retries: int = 2,
    sleep_s: float = 0.25,
) -> pd.DataFrame:

    if yf is None:
        raise ImportError("yfinance is not installed. pip install yfinance")
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))

    cache = None
    if cache_path and os.path.exists(cache_path):
        try:
            cache = pd.read_parquet(cache_path)
            if cache.index.name != 'ticker':
                cache.index.name = 'ticker'
        except Exception:
            cache = None

    rows = []
    have = set(cache.index) if isinstance(cache, pd.DataFrame) else set()
    todo = [t for t in tickers if t not in have]

    def _get_info_safe(t: str) -> Dict[str, object]:
        last_err = None
        for k in range(retries):
            try:
                tk = yf.Ticker(t)
                # yfinance>=0.2: prefer get_info(); fall back to .info
                info = {}
                try:
                    info = tk.get_info()
                except Exception:
                    info = getattr(tk, 'info', {}) or {}
                return {
                    'ticker': t,
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'longName': info.get('longName') or info.get('shortName'),
                    'shortName': info.get('shortName'),
                    'marketCap': info.get('marketCap'),
                    'quoteType': info.get('quoteType'),
                    'exchange': info.get('exchange'),
                    'country': info.get('country'),
                }
            except Exception as e:
                last_err = e
                time.sleep(sleep_s * (2 ** k))
        # On repeated failure, return a blank row
        return {'ticker': t, 'sector': None, 'industry': None, 'longName': None,
                'shortName': None, 'marketCap': None, 'quoteType': None, 'exchange': None, 'country': None}

    # Collect
    if todo:
        for t in todo:
            rows.append(_get_info_safe(t))
            time.sleep(sleep_s)
        add = pd.DataFrame(rows).set_index('ticker')
        meta = (cache.combine_first(add) if isinstance(cache, pd.DataFrame) else add)
    else:
        meta = cache if isinstance(cache, pd.DataFrame) else pd.DataFrame(columns=['sector','industry','longName','shortName','marketCap','quoteType','exchange','country'])
        meta.index.name = 'ticker'

    # Persist cache
    if cache_path:
        try:
            meta.to_parquet(cache_path)
        except Exception:
            pass
    return meta

def cli():
    parser = argparse.ArgumentParser(description="OHLCV + sector/industry downloader with yfinance")
    parser.add_argument("--tickers", type=str, required=True, help="CSV with a 'ticker' column")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--no-auto-adjust", action="store_true", help="Use unadjusted OHLC + explicit Adj Close")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--meta-cache", type=str, default=None, help="Path to Parquet cache for metadata")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tickers = pd.read_csv(args.tickers)
    if 'ticker' not in tickers.columns:
        raise ValueError("CSV must contain 'ticker' column")
    tickers = tickers['ticker'].dropna().astype(str).tolist()

    ohlcv = download_ohlcv_yf(
        tickers, start=args.start, end=args.end,
        chunk=args.chunk, auto_adjust=(not args.no_auto_adjust),
        progress=args.progress
    )
    # Save panels
    for key, df in ohlcv.items():
        if isinstance(df, pd.DataFrame):
            df.to_parquet(os.path.join(args.outdir, f"{key}.parquet"))
    # Metadata
    meta = fetch_metadata_yf(tickers, cache_path=args.meta_cache)
    if isinstance(meta, pd.DataFrame):
        meta.to_parquet(os.path.join(args.outdir, "metadata.parquet"))

    print(f"Saved OHLCV panels and metadata to {args.outdir}")

if __name__ == "__main__":
    cli()

