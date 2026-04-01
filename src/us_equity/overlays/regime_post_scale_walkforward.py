"""Walk-forward regime post-scaling wrapper."""

import pandas as pd

from us_equity.overlays.kalman_overlays_walkforward import (
    apply_scale_to_weights,
    build_regime_scale_walkforward,
)


def regime_post_scale_walkforward(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    winsor_k: float = 7.0,
    hmm_min_history: int = 252,
    hmm_refit_every: int = 1,
    hmm_train_window: int | None = None,
    hmm_iter: int = 60,
    hmm_tol: float = 1e-6,
    mode: str = "dynamic",
    p_lo: float = 0.45,
    p_hi: float = 0.60,
    roll: int = 252,
    q_lo: float = 0.40,
    q_hi: float = 0.60,
    min_quantile_history: int | None = None,
    kf_q: float = 5e-5,
    kf_r: float = 5e-4,
    z_gate: float = 2.5,
    floor: float = 0.3,
) -> pd.DataFrame:
    scale, _ = build_regime_scale_walkforward(
        close=close,
        winsor_k=winsor_k,
        hmm_min_history=hmm_min_history,
        hmm_refit_every=hmm_refit_every,
        hmm_train_window=hmm_train_window,
        hmm_iter=hmm_iter,
        hmm_tol=hmm_tol,
        mode=mode,
        p_lo=p_lo,
        p_hi=p_hi,
        roll=roll,
        q_lo=q_lo,
        q_hi=q_hi,
        min_quantile_history=min_quantile_history,
        kf_q=kf_q,
        kf_r=kf_r,
        z_gate=z_gate,
        floor=floor,
    )
    return apply_scale_to_weights(weights, scale)
