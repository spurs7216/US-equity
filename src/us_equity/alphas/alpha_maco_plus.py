# EMA crossover (robust): same robustness as the HMA version, but using EMAs on LOG prices.
# - Winsorize returns (MAD-based by default) before vol-scaling
# - Robust volatility via rolling MAD
# - Hard-cap absurd returns (mask), stale-quote filter
# - Cross-sectional gating with fallback (quantile/top-fraction/min breadth)
# - Causal EMA averaging
# Produces a dates×tickers alpha matrix compatible with your make_weights/backtest.
#
# File: /mnt/data/alpha_ema_robust.py

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

def ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    return df.ewm(span=span, adjust=False, min_periods=span).mean()

def winsorize_series_by_mad(s: pd.Series, k: float = 7.0) -> pd.Series:
    x = s.copy()
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return x
    sigma = 1.4826 * mad
    lo, hi = med - k * sigma, med + k * sigma
    return x.clip(lower=lo, upper=hi)

def winsorize_returns(
    rets: pd.DataFrame,
    method: str = "global_mad",  # "global_mad", "global_quantile", "per_name_mad", "per_name_quantile"
    q_low: float = 0.005,
    q_high: float = 0.995,
    k_mad: float = 7.0,
) -> pd.DataFrame:
    r = rets.copy()
    if method == "global_quantile":
        ql, qh = r.stack().quantile([q_low, q_high]).values
        return r.clip(lower=ql, upper=qh)
    if method == "global_mad":
        s = r.stack()
        s_w = winsorize_series_by_mad(s, k=k_mad)
        return s_w.unstack().reindex_like(r)
    if method == "per_name_quantile":
        def _clip_q(col):
            if col.notna().sum() == 0: return col
            ql, qh = col.quantile([q_low, q_high])
            return col.clip(lower=ql, upper=qh)
        return r.apply(_clip_q, axis=0)
    if method == "per_name_mad":
        return r.apply(lambda col: winsorize_series_by_mad(col, k=k_mad), axis=0)
    return r

def rolling_robust_vol(rets: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    def _mad_sigma(x: pd.Series) -> float:
        x = x.dropna()
        if x.empty:
            return np.nan
        med = x.median()
        mad = (x - med).abs().median()
        return 1.4826 * mad
    return rets.rolling(window, min_periods=window).apply(_mad_sigma, raw=False)

def alpha_ema_crossover_robust(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    fast_span: int = 12,
    slow_span: int = 24,
    atr_window: int = 24,
    # winsor/robust settings
    winsor_method: str = "global_mad",
    q_low: float = 0.005,
    q_high: float = 0.995,
    k_mad: float = 7.0,
    # cross-section gating
    threshold_q: Optional[float] = 0.95,
    top_frac: Optional[float] = 0.20,
    min_keep: int = 300,
    # stale filter & hard cap
    stale_lookback: int = 5,
    stale_std_bp: float = 1.0,
    hard_cap_abs_return: float = 0.50,
    # causal EMA averaging
    smooth_span: Optional[int] = 5,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:

    # 1) Returns cleanup
    rets = close.pct_change(fill_method=None)
    bad = rets.abs() > hard_cap_abs_return
    rets = rets.mask(bad)

    rets_w = winsorize_returns(rets, method=winsor_method, q_low=q_low, q_high=q_high, k_mad=k_mad)

    roll_std = rets_w.rolling(stale_lookback, min_periods=stale_lookback).std(ddof=1)
    stale_mask = (roll_std * 10000.0) < stale_std_bp
    tradable_mask = close.notna() & ~stale_mask

    # 2) EMA crossover on LOG prices
    lp = np.log(close)
    ema_fast = ema(lp, fast_span)
    ema_slow = ema(lp, slow_span)
    diff = (ema_fast - ema_slow)  # log spread ~ % difference
    diff = diff.where(tradable_mask)

    # 3) Cross-sectional gating with fallback
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

    # 4) Robust volatility scaling
    sigma = rolling_robust_vol(rets_w, window=atr_window)
    inv_vol = 1.0 / sigma.clip(lower=1e-6)
    alpha = sig * inv_vol

    # 5) Causal EMA averaging
    if smooth_span and smooth_span > 1:
        alpha = alpha.ewm(span=smooth_span, adjust=False, min_periods=smooth_span).mean()

    alpha = -alpha.where(tradable_mask, 0.0).fillna(0.0)

    diag = {
        "n_active": active_counts,
        "n_kept": kept.sum(axis=1).astype(int),
        "winsor_method": winsor_method,
        "hard_capped_frac": float(bad.mean().mean()) if bad.size else 0.0,
        "stale_frac": float(stale_mask.mean().mean()) if stale_mask.size else 0.0,
    }
    return alpha, diag

# Save file for the user to import later
# with open("/mnt/data/alpha_ema_robust.py", "w") as f:
#     f.write(
#         "import numpy as np\nimport pandas as pd\nfrom typing import Optional, Tuple, Dict\n\n" +
#         ema.__code__.co_consts[0] if False else ""
#     )

# print("Defined: alpha_ema_crossover_robust (file saved at /mnt/data/alpha_ema_robust.py)")
