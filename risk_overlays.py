
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def market_series(close: pd.DataFrame, winsor_k: float = 7.0) -> pd.Series:
    r = np.log(close).diff()
    # cross-sec MAD winsor per day
    def _w(row):
        row = row.dropna()
        if row.empty: return np.nan
        med = row.median(); mad = (row - med).abs().median()
        if not np.isfinite(mad) or mad == 0: return med
        sigma = 1.4826 * mad; lo, hi = med - 7*sigma, med + 7*sigma
        row = row.clip(lower=lo, upper=hi)
        return row.median()
    return r.apply(_w, axis=1).dropna()

def scale_by_market_vol(alpha: pd.DataFrame, close: pd.DataFrame, lookback: int = 20, gamma: float = 0.25, s_min: float = 0.3) -> pd.DataFrame:
    m = market_series(close)
    vol = m.rolling(lookback, min_periods=lookback).std(ddof=1)
    # robust z using median/MAD
    med = vol.rolling(252, min_periods=60).median()
    mad = (vol - med).abs().rolling(252, min_periods=60).median()
    z = (vol - med) / (1.4826 * mad.replace(0, np.nan))
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = (1.0 - gamma * z.clip(lower=0.0)).clip(lower=s_min, upper=1.0)
    s = s.reindex(alpha.index).fillna(method="ffill").fillna(1.0)
    return alpha.mul(s.values, axis=0)

def neutralize_to_market_beta(weights: pd.DataFrame, close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    rets = np.log(close).diff()
    m = market_series(close)
    # rolling betas per column
    betas = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for col in rets.columns:
        # Cov/Var rolling
        cov = (rets[col] * m).rolling(lookback, min_periods=lookback).mean() - rets[col].rolling(lookback, min_periods=lookback).mean()*m.rolling(lookback, min_periods=lookback).mean()
        var = m.rolling(lookback, min_periods=lookback).var(ddof=1)
        betas[col] = (cov / var).replace([np.inf, -np.inf], np.nan)
    betas = betas.reindex_like(weights)

    # Remove projection: w' = w - ( (w·b)/(b·b) ) * b
    out = weights.copy()
    bb = (betas**2).sum(axis=1)
    wb = (weights * betas).sum(axis=1)
    coef = (wb / bb.replace(0,np.nan)).fillna(0.0)
    adj = betas.mul(coef, axis=0).fillna(0.0)
    out = (weights - adj).fillna(0.0)

    # Re-impose dollar neutrality and gross exposure (approx)
    # center to zero net
    out = out.sub(out.mean(axis=1), axis=0)
    # rescale to match original gross per day
    orig_gross = weights.abs().sum(axis=1).replace(0,np.nan)
    new_gross = out.abs().sum(axis=1).replace(0,np.nan)
    scale = (orig_gross / new_gross).clip(upper=10.0).fillna(0.0)
    out = out.mul(scale, axis=0)
    return out

def dd_exposure_scale(ret_series: pd.Series, window: int = 252, dd_cut: float = -0.08, floor: float = 0.3, halflife: int = 20) -> pd.Series:

    eq = (1.0 + ret_series.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq/peak - 1.0).clip(lower=-1.0)
    # smooth drawdown via EMA
    lam = np.exp(np.log(0.5)/max(1,halflife))
    dd_s = dd.ewm(alpha=1-lam, adjust=False).mean()
    # map to scale
    scale = np.where(dd_s <= dd_cut, floor + (1-floor)*(dd_s/dd_cut), 1.0)
    scale = np.clip(scale, floor, 1.0)
    return pd.Series(scale, index=ret_series.index, name="dd_scale")
