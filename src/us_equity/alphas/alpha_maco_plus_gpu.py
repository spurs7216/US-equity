# GPU-accelerated EMA crossover (PyTorch): drop-in replacement for faster alpha builds.
# - Works on CUDA if available; otherwise falls back to CPU torch (still fast).
# - Uses LOG-price EMAs (fast/slow), robust vol (EWM absolute deviation), winsorized returns,
#   stale-quote filter, cross-sectional gating (quantile/top-fraction/min breadth), smoothing.
# - Returns a pandas DataFrame alpha (dates × tickers) and diagnostics.
#
# File: /mnt/data/alpha_ema_gpu.py

import math
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:
    raise ImportError("This GPU path requires PyTorch. Please `pip install torch`") from e

def _to_tensor(df: pd.DataFrame, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    x = torch.tensor(df.to_numpy(), dtype=dtype, device=device)
    # Ensure NaNs preserved
    return x

def _to_df(x: torch.Tensor, index, columns) -> pd.DataFrame:
    return pd.DataFrame(x.detach().cpu().numpy(), index=index, columns=columns)

def _ema_scan(x: torch.Tensor, span: int, min_periods: int = None) -> torch.Tensor:
    """
    Exponential moving average along time (dim=0), vectorized across columns.
    NaN-safe: when x[t] is NaN, we carry forward y[t-1] (no update).
    min_periods: until sufficient observations (non-NaN) are seen, output is NaN.
    """
    if span < 1:
        raise ValueError("span must be >= 1")
    if min_periods is None:
        min_periods = span
    alpha = 2.0 / (span + 1.0)
    T, N = x.shape
    y = torch.empty_like(x)
    y[:] = torch.nan
    # Track counts of finite obs per column
    is_ok = torch.isfinite(x)
    count = torch.zeros((N,), dtype=torch.int32, device=x.device)

    # Initialize: first finite value per column
    prev = torch.zeros((N,), dtype=x.dtype, device=x.device)
    prev[:] = 0.0
    has_prev = torch.zeros((N,), dtype=torch.bool, device=x.device)

    for t in range(T):
        xt = x[t]
        mt = is_ok[t]
        # For columns where we see first finite obs, set prev = x
        first_new = (~has_prev) & mt
        prev = torch.where(first_new, xt, prev)
        has_prev = has_prev | mt

        # Update EMA where we have finite x this step
        upd = prev * (1 - alpha) + xt * alpha
        prev = torch.where(mt, upd, prev)

        # Increase count only where finite x arrives
        count = count + mt.to(torch.int32)

        # Output: prev if count >= min_periods, else NaN
        good = count >= int(min_periods)
        y[t] = torch.where(good, prev, torch.tensor(torch.nan, dtype=x.dtype, device=x.device))

    return y

def _ewmad_sigma(r: torch.Tensor, span: int) -> torch.Tensor:
    """
    Robust-ish volatility via exponential moving mean absolute deviation:
      m_t = EMA(r_t)
      d_t = |r_t - m_t|
      s_t = EMA(d_t)
      sigma_t ≈ 1.2533 * s_t    (approx; 1.4826*MAD for normal, EWMAD ≈ 0.845*sigma; use 1.2533)
    All along time, vectorized over columns.
    """
    m = _ema_scan(r, span, min_periods=span)
    d = torch.abs(r - m)
    s = _ema_scan(d, span, min_periods=span)
    sigma = 1.2533141373155 * s  # calibration constant for normal
    return sigma

def _nanquantile_rowwise(x: torch.Tensor, q: float) -> torch.Tensor:
    """
    Row-wise nanquantile along columns (dim=1). Returns shape (T,1).
    Uses torch.nanquantile if available; otherwise manual via sort.
    """
    try:
        return torch.nanquantile(x, q, dim=1, keepdim=True)
    except Exception:
        # Manual: mask NaNs, sort, pick index
        T, N = x.shape
        out = torch.empty((T, 1), device=x.device, dtype=x.dtype)
        for t in range(T):
            row = x[t]
            row = row[torch.isfinite(row)]
            if row.numel() == 0:
                out[t, 0] = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
            else:
                k = int(max(0, min(row.numel() - 1, math.floor(q * (row.numel() - 1)))))
                vals, _ = torch.sort(row)
                out[t, 0] = vals[k]
        return out

def alpha_ema_crossover_gpu(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    fast_span: int = 12,
    slow_span: int = 24,
    atr_span: int = 24,
    # winsor / hard caps
    use_global_mad_winsor: bool = True,
    mad_k: float = 7.0,
    hard_cap_abs_return: float = 0.50,
    # stale detection (EWM std in bps)
    stale_span: int = 5,
    stale_std_bp: float = 1.0,
    # gating
    threshold_q: float = 0.95,
    top_frac: float = 0.20,
    min_keep: int = 300,
    # smoothing
    smooth_span: int = 5,
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Align / tensors
    idx, cols = close.index, close.columns
    X = _to_tensor(close.astype('float32'), dev)   # (T,N)
    V = _to_tensor(volume.astype('float32'), dev) if volume is not None else None

    # 1) Returns (log), winsorize (global MAD), hard-cap bad points
    lp = torch.log(X)
    r = lp[1:] - lp[:-1]   # log returns
    r = torch.nan_to_num(r, nan=float('nan'))

    # Hard-cap mask (mask > threshold)
    if hard_cap_abs_return is not None:
        bad = torch.abs(r) > hard_cap_abs_return
        r = torch.where(bad, torch.tensor(float('nan'), device=dev, dtype=r.dtype), r)
    else:
        bad = torch.zeros_like(r, dtype=torch.bool)

    # Global MAD winsor (flatten over T×N)
    if use_global_mad_winsor:
        flat = r.flatten()
        flat_ok = flat[torch.isfinite(flat)]
        if flat_ok.numel() > 0:
            med = torch.median(flat_ok)
            mad = torch.median(torch.abs(flat_ok - med))
            sigma = 1.4826 * mad
            lo = med - mad_k * sigma
            hi = med + mad_k * sigma
            r = torch.clamp(r, min=lo.item(), max=hi.item())

    # 2) Stale detection via EWM std (approx)
    #    Compute ewm std of returns in bps; stale if below threshold
    std_ewm = torch.sqrt(torch.clamp(_ema_scan(r**2, stale_span, min_periods=stale_span), min=0))
    std_bp = std_ewm * 1e4
    # prepend one row of NaNs to align with original T
    pad = torch.full((1, std_bp.shape[1]), float('nan'), device=dev, dtype=std_bp.dtype)
    std_bp_full = torch.cat([pad, std_bp], dim=0)
    stale = std_bp_full < stale_std_bp

    # 3) EMAs on LOG prices
    ema_fast = _ema_scan(lp, fast_span, min_periods=fast_span)
    ema_slow = _ema_scan(lp, slow_span, min_periods=slow_span)
    diff = ema_fast - ema_slow   # (T,N) log spread

    # Tradable mask: finite close and not stale
    tradable = torch.isfinite(X) & (~stale)

    diff = torch.where(tradable, diff, torch.tensor(float('nan'), device=dev, dtype=diff.dtype))

    # 4) Cross-sectional gating per day
    abs_diff = torch.abs(diff)
    # primary tail threshold
    tau_tail = _nanquantile_rowwise(abs_diff, threshold_q)  # (T,1)
    keep_tail = abs_diff >= tau_tail

    # fallback to ensure min breadth via top_frac
    n_active = torch.sum(torch.isfinite(abs_diff), dim=1, keepdim=True)  # (T,1)
    frac = float(max(0.0, min(1.0, top_frac)))
    target_frac = torch.maximum(frac * torch.ones_like(n_active, dtype=abs_diff.dtype), (min_keep / torch.clamp(n_active, min=1)).to(abs_diff.dtype))
    q_fb = torch.clamp(1.0 - target_frac, 0.0, 1.0)
    tau_fb = _nanquantile_rowwise(abs_diff, q_fb.squeeze(1).mean().item())  # approximate with mean q for speed
    keep_fb = abs_diff >= tau_fb

    kept = keep_tail | keep_fb
    sig = torch.sign(diff)
    sig = torch.where(kept, sig, torch.tensor(0.0, device=dev, dtype=diff.dtype))

    # 5) Robust volatility (EWM absolute deviation)
    # align r back to (T,N) with top padding row of NaNs
    r_full = torch.cat([torch.full((1, r.shape[1]), float('nan'), device=dev, dtype=r.dtype), r], dim=0)
    sigma = _ewmad_sigma(r_full, atr_span)
    inv_vol = 1.0 / torch.clamp(sigma, min=1e-6)
    alpha = sig * inv_vol

    # 6) Temporal smoothing (optional)
    if smooth_span and smooth_span > 1:
        alpha = _ema_scan(alpha, smooth_span, min_periods=smooth_span)

    # Final: zero-out non-tradable, replace NaN with 0
    alpha = torch.where(tradable, alpha, torch.tensor(0.0, device=dev, dtype=alpha.dtype))
    alpha = -torch.nan_to_num(alpha, nan=0.0)

    # Return as DataFrame
    alpha_df = _to_df(alpha, idx, cols)

    # Diagnostics
    n_active_pd = pd.Series(n_active.squeeze(1).detach().cpu().numpy(), index=idx, name="n_active")
    n_kept_pd = pd.Series(torch.sum(kept, dim=1).detach().cpu().numpy(), index=idx, name="n_kept")
    diag = {
        "n_active": n_active_pd,
        "n_kept": n_kept_pd,
        "device": str(dev),
        "hard_capped_frac": float(bad.float().mean().item()) if bad.numel() else 0.0
    }
    return alpha_df, diag

