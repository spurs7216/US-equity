
import numpy as np
import pandas as pd
from typing import Union, Mapping, Iterable

def _ensure_df(x: Union[pd.DataFrame, Mapping], key_candidates: Iterable[str], name_for_error: str) -> pd.DataFrame:
    """Return a DataFrame whether x is already a DataFrame or a dict-like.
    Tries keys in order (e.g., ['close','adj_close']).
    Raises a clear error if nothing found.
    """
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, Mapping):
        for k in key_candidates:
            v = x.get(k, None)
            if isinstance(v, pd.DataFrame):
                return v
    raise ValueError(f"Expected a DataFrame or dict with one of keys {list(key_candidates)} for '{name_for_error}', got: {type(x).__name__}")

def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, drop all-NaN columns, sort columns for stability."""
    df = df.copy()
    # Try to coerce non-numeric to NaN
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')
    # Ensure datetime index sorted
    df = df.sort_index()
    # Sort columns to fixed order
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def alpha_ma_crossover(
    close,
    volume=None,
    short_w: int = 12,
    long_w: int = 24,
    threshold_q: float = 0.95,
    atr_window: int = 24,
    vol_floor: float = 1e-6,
    use_volume_gate: bool = False,
    vol_gate_lookback: int = 20,
    vol_gate_quantile: float = 0.2,
) -> pd.DataFrame:
    """Moving-average crossover alpha → (dates × tickers) matrix.

    Accepts either:
      - close: DataFrame, or dict-like with keys among ['close','adj_close','price']
      - volume: DataFrame, or dict-like with keys among ['volume','vol']

    Steps per date:
      1) diff = (SMA_short - SMA_long) / |SMA_long|   (safe divide; zeros→NaN)
      2) Keep only top-|diff| names via threshold_q (cross-section)
      3) sig = sign(diff) for kept names; 0 elsewhere
      4) Risk scale by ATR proxy = mean(|return|, atr_window)
      5) Optional volume gate: keep top quantile by relative volume

    Returns a float DataFrame (NaNs where insufficient history)."""

    # Resolve inputs
    close_df = _ensure_df(close, ['close','adj_close','price'], 'close')
    if volume is not None:
        try:
            volume_df = _ensure_df(volume, ['volume','vol'], 'volume')
        except Exception:
            volume_df = None
    else:
        volume_df = None

    # Clean frames
    close_df = _clean_numeric_frame(close_df)
    if volume_df is not None:
        volume_df = _clean_numeric_frame(volume_df)

    if close_df.empty:
        raise ValueError("'close' DataFrame is empty after cleaning.")

    # Rolling SMAs (past-only)
    sma_s = close_df.rolling(short_w, min_periods=short_w).mean()
    sma_l = close_df.rolling(long_w, min_periods=long_w).mean()

    # Safe denominator: avoid None/zero; keep NaN where insufficient data
    denom = sma_l.abs()
    denom = denom.where(denom > 0, np.nan)

    diff = (sma_s - sma_l) / denom

    # Cross-sectional threshold per date
    abs_diff = diff.abs()
    tau = abs_diff.quantile(threshold_q, axis=1)
    sig = -np.sign(diff).where(abs_diff.ge(tau), 0.0)

    # ATR proxy from close-to-close returns
    ret = close_df.pct_change()
    atr = ret.abs().rolling(atr_window, min_periods=atr_window).mean()
    inv_vol = 1.0 / atr.clip(lower=vol_floor)

    alpha = sig * inv_vol

    # Optional: volume gate
    if use_volume_gate and volume_df is not None and not volume_df.empty:
        rel_vol = volume_df / volume_df.rolling(vol_gate_lookback, min_periods=vol_gate_lookback).mean()
        cutoff = rel_vol.quantile(1.0 - vol_gate_quantile, axis=1)
        liquid_mask = rel_vol.ge(cutoff, axis=0)
        alpha = alpha.where(liquid_mask, 0.0)

    # Keep numeric, preserve NaNs where no signal/history
    alpha = alpha.where(np.isfinite(alpha))
    return alpha
