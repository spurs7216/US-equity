# Regime-aware improvements: make Kalman matter by moving regime scaling *after* vol targeting
# and by using dynamic thresholds + innovation gating.

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ---------- Robust market series ----------
def market_series(close: pd.DataFrame, winsor_k: float = 7.0) -> pd.Series:
    r = np.log(close).diff()
    def _w(row):
        row = row.dropna()
        if row.empty: return np.nan
        med = row.median(); mad = (row - med).abs().median()
        if not np.isfinite(mad) or mad == 0: return med
        sigma = 1.4826 * mad; lo, hi = med - winsor_k*sigma, med + winsor_k*sigma
        row = row.clip(lower=lo, upper=hi)
        return row.median()
    return r.apply(_w, axis=1).dropna()

# ---------- 2-state Gaussian HMM (1D) ----------
def _init_params_2s(x: np.ndarray):
    q25, q75 = np.quantile(x, [0.25, 0.75])
    mu = np.array([q25, q75], dtype=float)
    sigma = np.array([np.std(x), np.std(x)], dtype=float) + 1e-6
    pi = np.array([0.5, 0.5], dtype=float)
    A = np.array([[0.95, 0.05],[0.05, 0.95]], dtype=float)
    return pi, A, mu, sigma

def _gauss_pdf(x, mu, sig):
    return (1.0 / np.sqrt(2*np.pi*sig**2)) * np.exp(-0.5*((x-mu)/sig)**2)

def _forward_backward(x, pi, A, mu, sigma):
    T = x.shape[0]; S=2
    B = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(S)]).T
    alpha = np.zeros((T,S)); c = np.zeros(T)
    alpha[0] = pi * B[0]; c[0] = alpha[0].sum() + 1e-300; alpha[0] /= c[0]
    for t in range(1,T):
        alpha[t] = (alpha[t-1] @ A) * B[t]
        c[t] = alpha[t].sum() + 1e-300; alpha[t] /= c[t]
    beta = np.zeros((T,S)); beta[-1] = 1.0
    for t in range(T-2,-1,-1):
        beta[t] = (A @ (B[t+1]*beta[t+1])); beta[t] /= (beta[t].sum() + 1e-300)
    gamma = alpha*beta; gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
    return gamma, c

def em_fit_hmm_2s(x: np.ndarray, n_iter: int = 50, tol: float = 1e-6):
    x = x[np.isfinite(x)]
    if x.size < 50: raise ValueError("Not enough data for HMM.")
    pi, A, mu, sigma = _init_params_2s(x)
    prev_ll = -np.inf
    for _ in range(n_iter):
        gamma, c = _forward_backward(x, pi, A, mu, sigma)
        # log-likelihood
        ll = np.sum(np.log(c + 1e-300))
        # E step helper for A
        # Backward part for xi via one-step recursion (approx for speed)
        # Use smoothed state probs at t and t+1 to re-estimate A (quick EM)
        g0 = gamma[:-1]; g1 = gamma[1:]
        A = (g0.T @ g1); A = A / (A.sum(axis=1, keepdims=True) + 1e-300)
        # pi
        pi = gamma[0]
        # mu, sigma
        mu = (gamma * x[:,None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        sigma = np.sqrt(((gamma * (x[:,None]-mu)**2).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)).clip(min=1e-12))
        if abs(ll - prev_ll) < tol: break
        prev_ll = ll
    order = np.argsort(mu)  # ensure mu0 < mu1
    return pi[order], A[order][:,order], mu[order], sigma[order]

def filter_probs_2s(x: np.ndarray, pi, A, mu, sigma) -> np.ndarray:
    T = x.shape[0]; S=2
    B = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(S)]).T
    f = np.zeros((T,S)); f[0] = pi * B[0]; f[0] /= (f[0].sum() + 1e-300)
    for t in range(1,T):
        pred = f[t-1] @ A
        f[t] = pred * B[t]; f[t] /= (f[t].sum() + 1e-300)
    return f

def fit_hmm_2state_prob_up(series: pd.Series, n_iter: int = 50, tol: float = 1e-6) -> pd.Series:
    x = series.dropna().values.astype(float)
    pi, A, mu, sigma = em_fit_hmm_2s(x, n_iter=n_iter, tol=tol)
    f = filter_probs_2s(x, pi, A, mu, sigma)
    p_up = pd.Series(f[:,1], index=series.dropna().index, name="p_up")
    return p_up

# ---------- Map P(up) → scale in [0,1] ----------
def map_prob_to_scale(p_up: pd.Series,
                      mode: str = "dynamic",
                      p_lo: float = 0.45, p_hi: float = 0.60,
                      roll: int = 252, q_lo: float = 0.40, q_hi: float = 0.60) -> pd.Series:
    if mode == "fixed":
        plo, phi = p_lo, p_hi
        def _s(p):
            if p <= plo: return 0.0
            if p >= phi: return 1.0
            return (p - plo)/max(1e-9, (phi - plo))
        return p_up.apply(_s)
    else:  # dynamic: rolling quantile thresholds
        p = p_up.copy()
        plo = p.rolling(roll, min_periods=roll//2).quantile(q_lo).shift(1)
        phi = p.rolling(roll, min_periods=roll//2).quantile(q_hi).shift(1)
        # fallback to global if NaN
        plo = plo.fillna(p.quantile(q_lo))
        phi = phi.fillna(p.quantile(q_hi))
        s = (p - plo) / (phi - plo).replace(0, np.nan)
        s = s.clip(0.0, 1.0).fillna(0.5)
        return s

# ---------- Kalman with diagnostics ----------
def kalman_filter_series_diag(y: pd.Series, q: float = 5e-5, r: float = 5e-4,
                              x0: Optional[float] = None, p0: float = 1.0):
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
        # predict
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
            innov[t] = 0.0; z[t] = 0.0

        x[t], p[t] = x_curr, p_curr
        x_prev, p_prev = x_curr, p_curr

    return (pd.Series(x, index=idx, name="kf"),
            pd.Series(p, index=idx, name="kf_var"),
            pd.Series(innov, index=idx, name="innov"),
            pd.Series(z, index=idx, name="innov_z"))

# ---------- Composite builder & application ----------
def build_regime_scale(close: pd.DataFrame,
                       mode: str = "dynamic",
                       p_lo: float = 0.45, p_hi: float = 0.60,
                       roll: int = 252, q_lo: float = 0.40, q_hi: float = 0.60,
                       kf_q: float = 5e-5, kf_r: float = 5e-4,
                       z_gate: float = 2.5, floor: float = 0.3):
    m = market_series(close)
    p_up = fit_hmm_2state_prob_up(m, n_iter=60, tol=1e-6)
    s_raw = map_prob_to_scale(p_up, mode=mode, p_lo=p_lo, p_hi=p_hi, roll=roll, q_lo=q_lo, q_hi=q_hi)
    s_kf, s_var, innov, innov_z = kalman_filter_series_diag(s_raw, q=kf_q, r=kf_r, x0=0.5, p0=1.0)
    # Innovation gate: if |z| > z_gate, shrink to floor that day (uncertain transition)
    gate = (innov_z.abs() <= z_gate).astype(float)
    s_final = np.maximum(floor, (s_kf * gate + floor*(1-gate)))
    s_final = pd.Series(s_final, index=s_kf.index, name="regime_scale")
    diag = {"p_up": p_up, "scale_raw": s_raw, "scale_kf": s_kf, "scale_var": s_var,
            "innov": innov, "innov_z": innov_z, "scale_final": s_final}
    return s_final, diag

def apply_scale_to_weights(weights: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    scale = scale.reindex(weights.index).fillna(method="ffill").fillna(1.0)
    return weights.mul(scale.values, axis=0)
