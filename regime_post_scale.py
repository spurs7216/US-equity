
import pandas as pd
from typing import Optional
from kalman_overlays import build_regime_scale, apply_scale_to_weights

def regime_post_scale(weights: pd.DataFrame, close: pd.DataFrame,
                      mode: str = "dynamic",
                      p_lo: float = 0.45, p_hi: float = 0.60,
                      roll: int = 252, q_lo: float = 0.40, q_hi: float = 0.60,
                      kf_q: float = 5e-5, kf_r: float = 5e-4,
                      z_gate: float = 2.5, floor: float = 0.3) -> pd.DataFrame:
    scale, _ = build_regime_scale(close, mode=mode, p_lo=p_lo, p_hi=p_hi, roll=roll,
                                  q_lo=q_lo, q_hi=q_hi, kf_q=kf_q, kf_r=kf_r,
                                  z_gate=z_gate, floor=floor)
    return apply_scale_to_weights(weights, scale)
