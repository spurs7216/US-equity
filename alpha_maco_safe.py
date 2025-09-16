# Performance upgrades to push Sharpe toward ~1.8 on your MA-crossover:
# - Adaptive EMA crossover (less whipsaw than SMA)
# - Cross-sectional tail + fallback top-fraction keep
# - OBV (volume) confirmation filter
# - Hysteresis / temporal smoothing to cut turnover
# - Sector/industry neutrality hook (optional; pass your metadata map)
# - Volatility targeting (ex-ante, no look-ahead) on portfolio weights
#
# Drop-in functions below. You can plug them into your existing pipeline.
#
# File saved as: /mnt/data/alpha_maco_plus.py


import numpy as np
import pandas as pd
from typing import Optional, Dict

# -----------------------------
# 1) Tech indicators (EMA, OBV)
# -----------------------------

def ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    # past-only EMA
    return df.ewm(span=span, adjust=False, min_periods=span).mean()

def obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    # On-Balance Volume using close-to-close direction
    r = close.pct_change(fill_method=None)
    direction = np.sign(r).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()

def zscore_by_row(df: pd.DataFrame) -> pd.DataFrame:
    def _z(s: pd.Series) -> pd.Series:
        m = s.mean(skipna=True); v = s.std(ddof=1, skipna=True)
        if not np.isfinite(v) or v == 0: return s*0.0
        return (s - m)/v
    return df.apply(_z, axis=1)

# ------------------------------------
# 2) Core alpha: EMA crossover + gating
# ------------------------------------
def alpha_maco_plus(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    fast_span: int = 12,
    slow_span: int = 24,
    atr_window: int = 24,
    threshold_q: Optional[float] = 0.95,  # tail keep
    top_frac: Optional[float] = 0.20,     # fallback keep top 20% by |diff|
    min_keep: int = 300,                  # ensure breadth
    use_obv_filter: bool = True,
    obv_span: int = 20,                   # OBV z-score window (via EMA approx)
    vol_floor: float = 1e-6,
    mask: Optional[pd.DataFrame] = None,
    smooth_span: Optional[int] = 5,       # temporal smoothing of alpha to cut turnover
) -> pd.DataFrame:
    if not isinstance(close, pd.DataFrame):
        raise TypeError("close must be a DataFrame")
    if volume is not None and not isinstance(volume, pd.DataFrame):
        raise TypeError("volume must be a DataFrame or None")

    # Clean / align
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()
        for c in df.columns:
            if not np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reindex(sorted(df.columns), axis=1)

    close = _clean(close)
    if volume is not None:
        volume = _clean(volume).reindex_like(close)

    if mask is not None:
        mask = mask.reindex_like(close).astype(bool)
    else:
        mask = close.notna()

    # EMA crossover
    ema_fast = ema(close, fast_span)
    ema_slow = ema(close, slow_span)
    denom = ema_slow.abs().where(lambda x: x>0, np.nan)
    diff = (ema_fast - ema_slow) / denom
    diff = diff.where(mask)

    # Cross-sectional gating
    abs_diff = diff.abs()
    kept = pd.DataFrame(False, index=diff.index, columns=diff.columns)

    active_counts = abs_diff.notna().sum(axis=1)
    def _keep_top_k(row: pd.Series, k: int) -> pd.Series:
        vals = row.dropna()
        if vals.empty: return pd.Series(False, index=row.index)
        k = max(1, min(k, len(vals)))
        thr = vals.abs().nlargest(k).min()
        return row.abs() >= thr

    for dt, row in abs_diff.iterrows():
        n_active = int(active_counts.loc[dt])
        if n_active == 0:
            continue
        row_kept = None
        if threshold_q is not None:
            q = row.quantile(threshold_q, interpolation="linear")
            if np.isfinite(q):
                row_kept = row >= q
        if row_kept is None or row_kept.sum() < min_keep:
            frac = top_frac if top_frac is not None else max(0.10, min_keep / max(1, n_active))
            k = int(np.ceil(frac * n_active))
            row_kept = _keep_top_k(row, k)
        kept.loc[dt] = row_kept.reindex_like(row).fillna(False)

    sig = np.sign(diff).where(kept, 0.0)

    # Risk scaling with ATR proxy (no forward-fill)
    ret = close.pct_change(fill_method=None)
    atr = ret.abs().rolling(atr_window, min_periods=atr_window).mean()
    inv_vol = 1.0 / atr.clip(lower=vol_floor)

    alpha = sig * inv_vol

    # OBV confirmation (filter out contradictory names)
    if use_obv_filter and (volume is not None):
        obv_raw = obv(close, volume)
        obv_z = zscore_by_row(obv_raw.diff().ewm(span=obv_span, adjust=False, min_periods=obv_span).mean())
        # Keep longs when OBV z > 0; shorts when OBV z < 0
        long_mask  = (alpha > 0) & (obv_z > 0)
        short_mask = (alpha < 0) & (obv_z < 0)
        alpha = alpha.where(long_mask | short_mask, 0.0)

    # Temporal smoothing to cut churn (optional)
    if smooth_span and smooth_span > 1:
        alpha = alpha.ewm(span=smooth_span, adjust=False, min_periods=smooth_span).mean()

    # Final clean & mask
    alpha = -alpha.where(np.isfinite(alpha), 0.0).where(mask, 0.0)
    return alpha

# --------------------------------------
# 3) Group (sector/industry) neutrality
# --------------------------------------
def neutralize_by_group(weights: pd.DataFrame, group_map: pd.Series) -> pd.DataFrame:
    group_map = group_map.dropna()
    common = weights.columns.intersection(group_map.index)
    if len(common) == 0: return weights
    g = group_map.loc[common]
    W = weights[common].copy()
    # subtract group means day by day
    for grp, cols in g.groupby(g).groups.items():
        sub = W[cols]
        W[cols] = sub.sub(sub.mean(axis=1, skipna=True), axis=0)
    # put back
    out = weights.copy()
    out[common] = W
    return out

# --------------------------------------
# 4) Volatility targeting (ex-ante)
# --------------------------------------
def vol_target_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_ann_vol: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    # Align
    weights = weights.reindex_like(returns).fillna(0.0)
    # Portfolio daily ret using lagged weights
    w_lag = weights.shift(1).fillna(0.0)
    port_ret = (w_lag * returns).sum(axis=1)

    # trailing vol estimate (rolling std)
    daily_target = target_ann_vol / np.sqrt(252.0)
    rolling_vol = port_ret.rolling(lookback, min_periods=lookback).std(ddof=1)
    scale = (daily_target / rolling_vol).clip(upper=max_leverage).fillna(0.0)

    # apply scale to *current* weights (decided at t for use on t+1)
    w_scaled = weights.mul(scale, axis=0).fillna(0.0)
    return w_scaled



