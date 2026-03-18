import pandas as pd
from typing import Optional

def load_sector_series(parquet_path: str,
                       tickers,
                       sector_col: str = "sector",
                       ticker_col: str = "ticker",
                       fill: str = "Unknown") -> pd.Series:
    """
    Load a parquet file with at least ['ticker','sector'] and return a pandas Series
    indexed by TICKER (uppercased) mapping to sector strings, aligned to `tickers`.
    """
    meta = pd.read_parquet(parquet_path)
    if ticker_col not in meta or sector_col not in meta:
        raise ValueError(f"Parquet must contain columns [{ticker_col!r}, {sector_col!r}].")
    meta = meta[[ticker_col, sector_col]].dropna().drop_duplicates(ticker_col)
    meta[ticker_col] = meta[ticker_col].astype(str).str.upper()
    s = meta.set_index(ticker_col)[sector_col]
    return s.reindex(pd.Index([str(t).upper() for t in tickers], dtype=object)).fillna(fill)
