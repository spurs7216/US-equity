import numpy as np
import pandas as pd
from typing import Dict, Tuple

EPS = 1e-12

def _ewmad_sigma(returns: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    m = returns.ewm(span=span, adjust=False, min_periods=span).mean()
    d = (returns - m).abs()
    s = d.ewm(span=span, adjust=False, min_periods=span).mean()
    return 1.2533141373155 * s

def _robust_cs_z(row: pd.Series) -> pd.Series:
    x = row.values.astype(float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if not mask.any():
        return pd.Series(out, index=row.index)
    xm = np.nanmedian(x[mask])
    mad = np.nanmedian(np.abs(x[mask] - xm))
    denom = 1.4826 * mad if mad and np.isfinite(mad) and mad > 0 else (np.nanstd(x[mask]) + EPS)
    out[mask] = (x[mask] - xm) / (denom if denom != 0 and np.isfinite(denom) else EPS)
    return pd.Series(out, index=row.index)

def _blend(dict_z: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    ks = [k for k in dict_z.keys() if k in weights and weights[k] != 0.0]
    if not ks:
        raise ValueError("No components selected for blending.")
    acc = None; ws = 0.0
    for k in ks:
        acc = (dict_z[k] * weights[k]) if acc is None else (acc + dict_z[k] * weights[k])
        ws += abs(weights[k])
    return acc / max(ws, EPS)

def alpha_fund_mom_stronger(
    close: pd.DataFrame,
    fund_daily: Dict[str, pd.DataFrame],
    # price momentum leg
    lookback_L: int = 252, skip_S: int = 10, vol_span: int = 20,
    # fundamental components (positive weights = good; negative = penalize)
    comp_weights: Dict[str, float] = None,
    # gating & breadth
    thresh_q: float = 0.90, top_frac: float = 0.20, min_keep: int = 300,
    # combine mode
    mode: str = "tilt", tilt_strength: float = 0.6,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:

    if comp_weights is None:
        comp_weights = {
            "rev_yoy": 1.0,
            "gmom_yoy": 0.75,
            "opmom_yoy": 0.75,
            "cfo_yoy": 0.75,
            "lev_delta": -0.5,
            "wc_delta": -0.5,
            "accr_delta": -0.5
        }

    # Price momentum (log-price over [t-L, t-S]) vol-normalized
    close = close.sort_index()
    lp = np.log(close)
    rets = lp.diff()
    sigma = _ewmad_sigma(rets, span=vol_span)
    if lookback_L <= skip_S:
        raise ValueError("lookback_L must be > skip_S")
    mom = (lp.shift(skip_S) - lp.shift(lookback_L)) / (sigma + EPS)
    Z_price = mom.apply(_robust_cs_z, axis=1)

    # Fundamental z-scores (per day)
    Zdict = {}
    for feat, w in comp_weights.items():
        if feat not in fund_daily:  # silently skip missing components
            continue
        Zdict[feat] = fund_daily[feat].apply(_robust_cs_z, axis=1)
    if not Zdict:
        raise ValueError("None of the requested fundamental features were found in fund_daily.")

    Z_fund = _blend(Zdict, comp_weights)

    # Combine
    if mode.lower() == "tilt":
        alpha_raw = Z_price * (1.0 + float(tilt_strength) * Z_fund)
    else:  # 'gate'
        gate = (Z_fund >= 0.0).astype(float)
        alpha_raw = Z_price * gate

    # Tail gating with breadth
    absz = alpha_raw.abs()
    keep = pd.DataFrame(False, index=absz.index, columns=absz.columns)
    for t, row in absz.iterrows():
        x = row.values
        mask = np.isfinite(x)
        if not mask.any(): continue
        x_ok = x[mask]; n = x_ok.size

        tau = np.nanquantile(x_ok, thresh_q)
        prim = np.zeros_like(x, dtype=bool); prim[mask] = x_ok >= tau

        target_frac = max(top_frac, min_keep / max(1, n))
        q_fb = max(0.0, 1.0 - target_frac)
        tau_fb = np.nanquantile(x_ok, q_fb)
        fb = np.zeros_like(x, dtype=bool); fb[mask] = x_ok >= tau_fb

        keep.loc[t] = prim | fb

    alpha = (alpha_raw * keep.astype(float)).fillna(0.0)

    diag = {
        "n_kept": keep.sum(axis=1),
        "comp_weights": pd.Series(comp_weights),
    }
    return alpha, diag