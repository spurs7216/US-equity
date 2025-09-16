import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable, Dict, Any

def _winsor_mad(x: pd.Series, k: float = 7.0) -> pd.Series:
    x = x.dropna()
    if x.empty: return x
    med = x.median(); mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0: return x
    sigma = 1.4826 * mad
    lo, hi = med - k*sigma, med + k*sigma
    return x.clip(lower=lo, upper=hi)

def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()

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
    T = x.shape[0]; S = 2
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
    xi = np.zeros((T-1,S,S))
    for t in range(T-1):
        denom = (alpha[t][:,None]*A*(B[t+1]*beta[t+1])[None,:]).sum() + 1e-300
        xi[t] = (alpha[t][:,None]*A*(B[t+1]*beta[t+1])[None,:]) / denom
    ll = np.sum(np.log(c + 1e-300))
    return gamma, xi, ll

def _em_fit_2s(x: np.ndarray, n_iter: int = 50, tol: float = 1e-6):
    x = x[np.isfinite(x)]
    if x.size < 50: raise ValueError("Not enough data for HMM.")
    pi, A, mu, sigma = _init_params_2s(x)
    prev_ll = -np.inf
    for _ in range(n_iter):
        gamma, xi, ll = _forward_backward(x, pi, A, mu, sigma)
        pi = gamma[0]
        A = xi.sum(axis=0); A = A / (A.sum(axis=1, keepdims=True) + 1e-300)
        mu = (gamma * x[:,None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        sigma = np.sqrt(((gamma * (x[:,None]-mu)**2).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)).clip(min=1e-12))
        if abs(ll - prev_ll) < tol: break
        prev_ll = ll
    order = np.argsort(mu)  # ensure mu0 < mu1  (1 = up)
    return pi[order], A[order][:,order], mu[order], sigma[order], ll

def _filter_probs_2s(x, pi, A, mu, sigma):
    T = x.shape[0]; S=2
    B = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(S)]).T
    f = np.zeros((T,S)); f[0] = pi * B[0]; f[0] /= (f[0].sum() + 1e-300)
    for t in range(1,T):
        pred = f[t-1] @ A
        f[t] = pred * B[t]; f[t] /= (f[t].sum() + 1e-300)
    return f

def hmm_overlay_scale(
    close: pd.DataFrame,
    base_alpha_fn: Callable[..., pd.DataFrame],
    base_alpha_kwargs: Dict[str, Any],
    winsor_k: float = 7.0,
    p_lo: float = 0.45,     # below this, scale≈0
    p_hi: float = 0.60,     # above this, scale≈1
    smooth_span: int = 3,
):
    r = _log_returns(close)
    r_w = r.apply(lambda row: _winsor_mad(row, k=winsor_k), axis=1)
    r_bar = r_w.median(axis=1, skipna=True).dropna()

    x = r_bar.values.astype(float)
    pi, A, mu, sigma, ll = _em_fit_2s(x, n_iter=50, tol=1e-6)
    f = _filter_probs_2s(x, pi, A, mu, sigma)
    p_up = pd.Series(f[:,1], index=r_bar.index, name="p_up")

    # Map p_up to scale s in [0,1], piecewise-linear (no inversion)
    def _scale(p):
        if p <= p_lo: return 0.0
        if p >= p_hi: return 1.0
        return (p - p_lo) / max(1e-9, (p_hi - p_lo))

    s = p_up.apply(_scale)
    if smooth_span and smooth_span > 1:
        s = s.ewm(span=smooth_span, adjust=False, min_periods=smooth_span).mean()

    base_alpha = base_alpha_fn(**base_alpha_kwargs)
    if isinstance(base_alpha, tuple):
        base_alpha = base_alpha[0]
    s_mat = pd.DataFrame(s).reindex(base_alpha.index).fillna(method="ffill").fillna(0.0).values
    alpha = base_alpha * s_mat

    diag = {"p_up": p_up, "scale": s, "mu": mu, "sigma": sigma, "pi": pi, "A": A, "loglik": ll}
    return alpha, diag

