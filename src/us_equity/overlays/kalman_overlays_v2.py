"""Versioned HMM plus causal Kalman regime scaling."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from us_equity.alphas.alpha_hmm_v3 import robust_market_series, fit_hmm_filtered_prob_up_v2


def _expanding_quantile_shifted(series: pd.Series, q: float, min_periods: int) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        if len(hist) >= min_periods:
            out.iloc[i] = float(hist.quantile(q))
    return out


def fit_hmm_2state_prob_up_v2(
    series: pd.Series,
    min_history: int = 252,
    refit_every: int = 1,
    train_window: Optional[int] = None,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[pd.Series, Dict[str, Any]]:
    return fit_hmm_filtered_prob_up_v2(
        series,
        min_history=min_history,
        refit_every=refit_every,
        train_window=train_window,
        n_iter=n_iter,
        tol=tol,
    )


def map_prob_to_scale_v2(
    p_up: pd.Series,
    mode: str = "dynamic",
    p_lo: float = 0.45,
    p_hi: float = 0.60,
    roll: int = 252,
    q_lo: float = 0.40,
    q_hi: float = 0.60,
    min_quantile_history: Optional[int] = None,
) -> pd.Series:
    if mode == "fixed":
        out = p_up.copy()
        for idx, value in p_up.items():
            if not np.isfinite(value):
                out.loc[idx] = np.nan
            elif value <= p_lo:
                out.loc[idx] = 0.0
            elif value >= p_hi:
                out.loc[idx] = 1.0
            else:
                out.loc[idx] = (value - p_lo) / max(1e-9, (p_hi - p_lo))
        return out

    min_hist = int(min_quantile_history or max(20, roll // 2))
    p = p_up.copy()
    plo_roll = p.rolling(roll, min_periods=min_hist).quantile(q_lo).shift(1)
    phi_roll = p.rolling(roll, min_periods=min_hist).quantile(q_hi).shift(1)
    plo_exp = _expanding_quantile_shifted(p, q=q_lo, min_periods=min_hist)
    phi_exp = _expanding_quantile_shifted(p, q=q_hi, min_periods=min_hist)

    plo = plo_roll.fillna(plo_exp).fillna(p_lo)
    phi = phi_roll.fillna(phi_exp).fillna(p_hi)
    denom = (phi - plo).where((phi - plo).abs() > 1e-9)

    scale = ((p - plo) / denom).clip(0.0, 1.0)
    scale = scale.where(p.notna())
    return scale


def kalman_filter_series_diag(
    y: pd.Series,
    q: float = 5e-5,
    r: float = 5e-4,
    x0: Optional[float] = None,
    p0: float = 1.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = y.index
    yv = y.astype(float).values
    x = np.zeros_like(yv, dtype=float)
    p = np.zeros_like(yv, dtype=float)
    innov = np.zeros_like(yv, dtype=float)
    z = np.zeros_like(yv, dtype=float)

    if x0 is None:
        finite = np.isfinite(yv)
        x0 = yv[np.argmax(finite)] if finite.any() else 0.5

    x_prev, p_prev = float(x0), float(p0)
    for t in range(len(yv)):
        x_pred = x_prev
        p_pred = p_prev + q

        yt = yv[t]
        if np.isfinite(yt):
            k = p_pred / (p_pred + r)
            innov_t = yt - x_pred
            x_curr = x_pred + k * innov_t
            p_curr = (1.0 - k) * p_pred
            innov[t] = innov_t
            z[t] = innov_t / np.sqrt(max(1e-12, p_pred + r))
        else:
            x_curr, p_curr = x_pred, p_pred
            innov[t] = np.nan
            z[t] = np.nan

        x[t], p[t] = x_curr, p_curr
        x_prev, p_prev = x_curr, p_curr

    return (
        pd.Series(x, index=idx, name="kf"),
        pd.Series(p, index=idx, name="kf_var"),
        pd.Series(innov, index=idx, name="innov"),
        pd.Series(z, index=idx, name="innov_z"),
    )


def build_regime_scale_v2(
    close: pd.DataFrame,
    winsor_k: float = 7.0,
    hmm_min_history: int = 252,
    hmm_refit_every: int = 1,
    hmm_train_window: Optional[int] = None,
    hmm_iter: int = 60,
    hmm_tol: float = 1e-6,
    mode: str = "dynamic",
    p_lo: float = 0.45,
    p_hi: float = 0.60,
    roll: int = 252,
    q_lo: float = 0.40,
    q_hi: float = 0.60,
    min_quantile_history: Optional[int] = None,
    kf_q: float = 5e-5,
    kf_r: float = 5e-4,
    z_gate: float = 2.5,
    floor: float = 0.3,
) -> Tuple[pd.Series, Dict[str, Any]]:
    market = robust_market_series(close, winsor_k=winsor_k)
    p_up, hmm_diag = fit_hmm_2state_prob_up_v2(
        market,
        min_history=hmm_min_history,
        refit_every=hmm_refit_every,
        train_window=hmm_train_window,
        n_iter=hmm_iter,
        tol=hmm_tol,
    )

    scale_raw = map_prob_to_scale_v2(
        p_up,
        mode=mode,
        p_lo=p_lo,
        p_hi=p_hi,
        roll=roll,
        q_lo=q_lo,
        q_hi=q_hi,
        min_quantile_history=min_quantile_history,
    )
    scale_kf, scale_var, innov, innov_z = kalman_filter_series_diag(scale_raw, q=kf_q, r=kf_r, x0=0.5, p0=1.0)

    gate = (innov_z.abs() <= z_gate).astype(float)
    gate = gate.where(scale_raw.notna())
    scale_final = np.maximum(floor, (scale_kf * gate + floor * (1 - gate)))
    scale_final = pd.Series(scale_final, index=scale_kf.index, name="regime_scale")

    available = scale_raw.notna()
    scale_kf = scale_kf.where(available)
    scale_var = scale_var.where(available)
    innov = innov.where(available)
    innov_z = innov_z.where(available)
    scale_final = scale_final.where(available)

    diag = {
        "p_up": p_up,
        "scale_raw": scale_raw,
        "scale_kf": scale_kf,
        "scale_var": scale_var,
        "innov": innov,
        "innov_z": innov_z,
        "scale_final": scale_final,
        **hmm_diag,
    }
    return scale_final, diag


def apply_scale_to_weights(weights: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    scale = scale.reindex(weights.index).ffill().fillna(1.0)
    return weights.mul(scale.values, axis=0)
