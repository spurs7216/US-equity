# Build an HMM-based alpha that fits a 2‑state Gaussian HMM on a robust
# 1D market series (median winsorized log returns), filters the daily
# regime probability, and modulates ANY base alpha you provide.
#
# The output is a dates×tickers alpha matrix, compatible with your
# make_weights → vol_target → backtest → stats pipeline.
#
# File: /mnt/data/alpha_hmm.py

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any

# ---------- Utilities ----------

def _winsor_mad(x: pd.Series, k: float = 7.0) -> pd.Series:
    x = x.dropna()
    if x.empty:
        return x
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return x
    sigma = 1.4826 * mad
    lo, hi = med - k * sigma, med + k * sigma
    return x.clip(lower=lo, upper=hi)

def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(close)
    r = lp.diff()  # t over t-1
    return r

# ---------- 2-state Gaussian HMM (1D) ----------

def _init_params_2s(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize pi, A, mu, sigma for 2-state HMM using quantiles."""
    # x is 1D finite numpy array
    q25, q75 = np.quantile(x, [0.25, 0.75])
    mu = np.array([q25, q75], dtype=float)
    sigma = np.array([np.std(x), np.std(x)], dtype=float) + 1e-6
    pi = np.array([0.5, 0.5], dtype=float)
    A = np.array([[0.95, 0.05],
                  [0.05, 0.95]], dtype=float)
    return pi, A, mu, sigma

def _gauss_pdf(x, mu, sig):
    return (1.0 / np.sqrt(2*np.pi*sig**2)) * np.exp(-0.5*((x-mu)/sig)**2)

def _forward_backward(x: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """Scaled forward-backward for 2-state Gaussian HMM; returns gammas, xis, loglik."""
    T = x.shape[0]
    S = 2
    # Emission probs B_t(i)
    B = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(S)]).T  # (T,S)
    # Forward
    alpha = np.zeros((T, S))
    c = np.zeros(T)  # scaling
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum() + 1e-300
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[t]
        c[t] = alpha[t].sum() + 1e-300
        alpha[t] /= c[t]
    # Backward
    beta = np.zeros((T, S))
    beta[-1] = 1.0
    for t in range(T-2, -1, -1):
        beta[t] = (A @ (B[t+1] * beta[t+1]))
        beta[t] /= (beta[t].sum() + 1e-300)
    # Gammas and xis
    gamma = alpha * beta
    gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)
    xi = np.zeros((T-1, S, S))
    for t in range(T-1):
        denom = (alpha[t][:,None] * A * (B[t+1]*beta[t+1])[None,:]).sum() + 1e-300
        xi[t] = (alpha[t][:,None] * A * (B[t+1]*beta[t+1])[None,:]) / denom
    # log-likelihood
    loglik = np.sum(np.log(c + 1e-300))
    return gamma, xi, loglik

def _em_fit_2s(x: np.ndarray, n_iter: int = 50, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """EM for 2-state Gaussian HMM on 1D data. Returns pi, A, mu, sigma, loglik."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Not enough data to fit HMM.")
    pi, A, mu, sigma = _init_params_2s(x)
    prev_ll = -np.inf
    for _ in range(n_iter):
        gamma, xi, ll = _forward_backward(x, pi, A, mu, sigma)
        # M-step
        pi = gamma[0]
        A = xi.sum(axis=0)
        A = A / (A.sum(axis=1, keepdims=True) + 1e-300)
        mu = (gamma * x[:,None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        sigma = np.sqrt(((gamma * (x[:,None]-mu)**2).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)).clip(min=1e-12))
        # Convergence
        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    # Order states so that state 1 has higher mean (trend/up) than state 0
    order = np.argsort(mu)
    mu = mu[order]
    sigma = sigma[order]
    pi = pi[order]
    A = A[order][:,order]
    return pi, A, mu, sigma, ll

def _filter_probs_2s(x: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Filtered probabilities P(S_t | x_{1:t}) for t=0..T-1 (2 states)."""
    T = x.shape[0]
    S = 2
    B = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(S)]).T  # (T,S)
    f = np.zeros((T, S))
    f[0] = pi * B[0]
    f[0] /= (f[0].sum() + 1e-300)
    for t in range(1, T):
        pred = f[t-1] @ A  # (S,)
        f[t] = pred * B[t]
        f[t] /= (f[t].sum() + 1e-300)
    return f  # (T,2)

# ---------- Public API ----------

def hmm_market_regime_alpha(
    close: pd.DataFrame,
    base_alpha_fn: Callable[..., pd.DataFrame],
    base_alpha_kwargs: Dict[str, Any],
    volume: Optional[pd.DataFrame] = None,
    winsor_k: float = 7.0,
    hmm_iter: int = 50,
    hmm_tol: float = 1e-6,
    smooth_span: Optional[int] = 3,   # causal EMA averaging of the regime series for stability
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    # 1) Returns and winsor
    r = _log_returns(close)
    # winsorize each day across names, robust to outliers in panel
    r_w = r.apply(lambda row: _winsor_mad(row, k=winsor_k), axis=1)

    # 2) Robust market series (median across names)
    r_bar = r_w.median(axis=1, skipna=True).dropna()
    if len(r_bar) < 50:
        raise ValueError("Not enough observations in r_bar to fit HMM.")

    # 3) Fit HMM on r_bar
    x = r_bar.values.astype(float)
    pi, A, mu, sigma, ll = _em_fit_2s(x, n_iter=hmm_iter, tol=hmm_tol)

    # 4) Filter probs and regime signal
    f = _filter_probs_2s(x, pi, A, mu, sigma)  # (T,2)
    p_up = pd.Series(f[:,1], index=r_bar.index, name="p_up")
    regime = 2.0 * p_up - 1.0  # [-1,1]

    if smooth_span and smooth_span > 1:
        regime = regime.ewm(span=smooth_span, adjust=False, min_periods=smooth_span).mean()

    # 5) Build base alpha
    alpha = base_alpha_fn(**base_alpha_kwargs)
    if isinstance(alpha, tuple):
        alpha = alpha[0]
    # Align
    regime_mat = pd.DataFrame(regime).reindex(alpha.index).fillna(method="ffill").fillna(0.0).values
    alpha_hmm = alpha * regime_mat

    diag = {
        "pi": pi, "A": A, "mu": mu, "sigma": sigma, "loglik": ll,
        "p_up": p_up, "regime": regime,
    }
    return alpha_hmm, diag

