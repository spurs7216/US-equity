"""Versioned regime post-scaling wrapper."""

import pandas as pd

from us_equity.overlays.kalman_overlays_v2 import (
    apply_scale_to_weights,
    build_regime_scale_v2,
    build_regime_scale_from_prob_v2,
    prepare_regime_prob_v2,
)


def regime_post_scale_v2(
    weights: pd.DataFrame,
    close: pd.DataFrame | None = None,
    precomputed_scale: pd.Series | None = None,
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
    if precomputed_scale is None:
        if close is None:
            raise ValueError("close is required when precomputed_scale is not provided.")
        scale, _ = build_regime_scale_v2(
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
    else:
        scale = precomputed_scale
    return apply_scale_to_weights(weights, scale)


def prepare_regime_shared_v2(
    close: pd.DataFrame,
    winsor_k: float = 7.0,
    hmm_min_history: int = 252,
    hmm_refit_every: int = 1,
    hmm_train_window: int | None = None,
    hmm_iter: int = 60,
    hmm_tol: float = 1e-6,
) -> dict:
    return prepare_regime_prob_v2(
        close=close,
        winsor_k=winsor_k,
        hmm_min_history=hmm_min_history,
        hmm_refit_every=hmm_refit_every,
        hmm_train_window=hmm_train_window,
        hmm_iter=hmm_iter,
        hmm_tol=hmm_tol,
    )


def prepare_regime_combo_v2(
    prepared_shared: dict,
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
) -> dict:
    scale, diag = build_regime_scale_from_prob_v2(
        prepared_shared["p_up"],
        hmm_diag=prepared_shared,
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
    return {"scale": scale, "diag": diag}


def apply_prepared_regime_v2(weights: pd.DataFrame, prepared_combo: dict) -> pd.DataFrame:
    return apply_scale_to_weights(weights, prepared_combo["scale"])


regime_post_scale_v2.prepare_shared = prepare_regime_shared_v2
regime_post_scale_v2.prepare_combo = prepare_regime_combo_v2
regime_post_scale_v2.apply_prepared = apply_prepared_regime_v2
