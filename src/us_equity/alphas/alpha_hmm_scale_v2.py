"""Versioned HMM scale overlay with expanding or rolling re-estimation."""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from us_equity.alphas.alpha_hmm_v3 import robust_market_series, fit_hmm_filtered_prob_up_v2


def _map_prob_to_scale(p: float, p_lo: float, p_hi: float) -> float:
    if not np.isfinite(p):
        return np.nan
    if p <= p_lo:
        return 0.0
    if p >= p_hi:
        return 1.0
    return (p - p_lo) / max(1e-9, (p_hi - p_lo))


def hmm_overlay_scale_v2(
    close: pd.DataFrame,
    base_alpha_fn: Callable[..., pd.DataFrame],
    base_alpha_kwargs: Dict[str, Any],
    winsor_k: float = 7.0,
    min_history: int = 252,
    refit_every: int = 1,
    train_window: Optional[int] = None,
    hmm_iter: int = 50,
    hmm_tol: float = 1e-6,
    p_lo: float = 0.45,
    p_hi: float = 0.60,
    scale_ema_span: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r_bar = robust_market_series(close, winsor_k=winsor_k)
    if len(r_bar) < min_history:
        raise ValueError("Not enough data for the v2 HMM scale overlay.")

    p_up, wf_diag = fit_hmm_filtered_prob_up_v2(
        r_bar,
        min_history=min_history,
        refit_every=refit_every,
        train_window=train_window,
        n_iter=hmm_iter,
        tol=hmm_tol,
    )

    scale_raw = p_up.apply(lambda p: _map_prob_to_scale(p, p_lo=p_lo, p_hi=p_hi))
    scale = scale_raw.copy()
    if scale_ema_span and scale_ema_span > 1:
        scale = scale.ewm(span=scale_ema_span, adjust=False, min_periods=scale_ema_span).mean()

    base_alpha = base_alpha_fn(**base_alpha_kwargs)
    if isinstance(base_alpha, tuple):
        base_alpha = base_alpha[0]

    scale_aligned = scale.reindex(base_alpha.index).ffill().fillna(0.0)
    alpha = base_alpha.mul(scale_aligned.values, axis=0)

    diag = {
        "p_up": p_up,
        "scale_raw": scale_raw,
        "scale": scale,
        "scale_aligned": scale_aligned,
        **wf_diag,
    }
    return alpha, diag
