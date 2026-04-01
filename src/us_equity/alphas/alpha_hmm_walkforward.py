"""Walk-forward HMM regime overlays without full-sample parameter leakage."""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


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
    return np.log(close).diff()


def robust_market_series(close: pd.DataFrame, winsor_k: float = 7.0) -> pd.Series:
    r = _log_returns(close)
    r_w = r.apply(lambda row: _winsor_mad(row, k=winsor_k), axis=1)
    return r_w.median(axis=1, skipna=True).dropna()


def _init_params_2s(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q25, q75 = np.quantile(x, [0.25, 0.75])
    mu = np.array([q25, q75], dtype=float)
    sigma = np.array([np.std(x), np.std(x)], dtype=float) + 1e-6
    pi = np.array([0.5, 0.5], dtype=float)
    a = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    return pi, a, mu, sigma


def _order_params(
    pi: np.ndarray,
    a: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(mu)
    return pi[order], a[order][:, order], mu[order], sigma[order]


def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _forward_backward(
    x: np.ndarray,
    pi: np.ndarray,
    a: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    t_count = x.shape[0]
    states = 2
    b = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(states)]).T

    alpha = np.zeros((t_count, states))
    c = np.zeros(t_count)
    alpha[0] = pi * b[0]
    c[0] = alpha[0].sum() + 1e-300
    alpha[0] /= c[0]
    for t in range(1, t_count):
        alpha[t] = (alpha[t - 1] @ a) * b[t]
        c[t] = alpha[t].sum() + 1e-300
        alpha[t] /= c[t]

    beta = np.zeros((t_count, states))
    beta[-1] = 1.0
    for t in range(t_count - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1])
        beta[t] /= beta[t].sum() + 1e-300

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
    loglik = float(np.sum(np.log(c + 1e-300)))
    return gamma, b, loglik


def _em_fit_2s(
    x: np.ndarray,
    n_iter: int = 50,
    tol: float = 1e-6,
    init_params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Not enough data to fit HMM.")

    if init_params is None:
        pi, a, mu, sigma = _init_params_2s(x)
    else:
        pi, a, mu, sigma = init_params
        pi = pi.astype(float).copy()
        a = a.astype(float).copy()
        mu = mu.astype(float).copy()
        sigma = sigma.astype(float).copy()

    prev_ll = -np.inf
    for _ in range(n_iter):
        gamma, b, ll = _forward_backward(x, pi, a, mu, sigma)

        xi = np.zeros((x.shape[0] - 1, 2, 2))
        for t in range(x.shape[0] - 1):
            numer = gamma[t][:, None] * a * b[t + 1][None, :]
            denom = numer.sum() + 1e-300
            xi[t] = numer / denom

        pi = gamma[0]
        a = xi.sum(axis=0)
        a = a / (a.sum(axis=1, keepdims=True) + 1e-300)
        mu = (gamma * x[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)
        sigma = np.sqrt(
            ((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / (gamma.sum(axis=0) + 1e-300)).clip(min=1e-12)
        )

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    pi, a, mu, sigma = _order_params(pi, a, mu, sigma)
    return pi, a, mu, sigma, ll


def _filter_probs_2s(
    x: np.ndarray,
    pi: np.ndarray,
    a: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    t_count = x.shape[0]
    states = 2
    b = np.vstack([_gauss_pdf(x, mu[i], sigma[i]) for i in range(states)]).T
    f = np.zeros((t_count, states))
    f[0] = pi * b[0]
    f[0] /= f[0].sum() + 1e-300
    for t in range(1, t_count):
        pred = f[t - 1] @ a
        f[t] = pred * b[t]
        f[t] /= f[t].sum() + 1e-300
    return f


def _update_filtered_state(
    prev_state: np.ndarray,
    x_t: float,
    a: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    b = np.array([_gauss_pdf(np.array([x_t]), mu[i], sigma[i])[0] for i in range(2)], dtype=float)
    pred = prev_state @ a
    curr = pred * b
    curr /= curr.sum() + 1e-300
    return curr


def walkforward_hmm_filtered_prob_up(
    series: pd.Series,
    min_history: int = 252,
    refit_every: int = 1,
    train_window: Optional[int] = None,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[pd.Series, Dict[str, Any]]:
    values = series.dropna().astype(float)
    p_up = pd.Series(np.nan, index=series.index, dtype=float, name="p_up")
    if values.empty:
        empty_mask = pd.Series(False, index=series.index, name="refit_mask")
        return p_up, {"refit_mask": empty_mask, "refit_dates": empty_mask.index[:0]}

    x = values.to_numpy()
    refit_mask = pd.Series(False, index=values.index, name="refit_mask")

    params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    filtered_state: Optional[np.ndarray] = None
    last_refit_end = -1
    step = max(1, int(refit_every))

    for end in range(len(x)):
        obs_count = end + 1
        if obs_count < min_history:
            continue

        should_refit = params is None or (end - last_refit_end) >= step
        if should_refit:
            start = 0 if train_window is None else max(0, obs_count - int(train_window))
            x_train = x[start:obs_count]
            pi, a, mu, sigma, _ = _em_fit_2s(
                x_train,
                n_iter=n_iter,
                tol=tol,
                init_params=params,
            )
            params = (pi, a, mu, sigma)
            filtered_state = _filter_probs_2s(x_train, pi, a, mu, sigma)[-1]
            last_refit_end = end
            refit_mask.loc[values.index[end]] = True
        else:
            assert params is not None and filtered_state is not None
            pi, a, mu, sigma = params
            filtered_state = _update_filtered_state(filtered_state, x[end], a, mu, sigma)

        p_up.loc[values.index[end]] = float(filtered_state[1])

    full_refit_mask = refit_mask.reindex(series.index, fill_value=False)
    return p_up, {
        "refit_mask": full_refit_mask,
        "refit_dates": full_refit_mask[full_refit_mask].index,
    }


def hmm_market_regime_alpha_walkforward(
    close: pd.DataFrame,
    base_alpha_fn: Callable[..., pd.DataFrame],
    base_alpha_kwargs: Dict[str, Any],
    volume: Optional[pd.DataFrame] = None,
    winsor_k: float = 7.0,
    min_history: int = 252,
    refit_every: int = 1,
    train_window: Optional[int] = None,
    hmm_iter: int = 50,
    hmm_tol: float = 1e-6,
    regime_ema_span: Optional[int] = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    del volume

    r_bar = robust_market_series(close, winsor_k=winsor_k)
    if len(r_bar) < min_history:
        raise ValueError("Not enough observations in r_bar to fit the walk-forward HMM.")

    p_up, wf_diag = walkforward_hmm_filtered_prob_up(
        r_bar,
        min_history=min_history,
        refit_every=refit_every,
        train_window=train_window,
        n_iter=hmm_iter,
        tol=hmm_tol,
    )
    regime = 2.0 * p_up - 1.0

    if regime_ema_span and regime_ema_span > 1:
        regime = regime.ewm(span=regime_ema_span, adjust=False, min_periods=regime_ema_span).mean()

    alpha = base_alpha_fn(**base_alpha_kwargs)
    if isinstance(alpha, tuple):
        alpha = alpha[0]

    regime_aligned = regime.reindex(alpha.index).ffill().fillna(0.0)
    alpha_hmm = alpha.mul(regime_aligned.values, axis=0)

    diag = {
        "p_up": p_up,
        "regime": regime,
        "regime_aligned": regime_aligned,
        **wf_diag,
    }
    return alpha_hmm, diag
