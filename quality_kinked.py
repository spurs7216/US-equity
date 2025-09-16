
import numpy as np
from typing import Dict, Tuple, Optional, Callable

def _transform_for_direction(x: float, lo: float, hi: float, higher_is_better: bool):
    """If lower-is-better, flip signs so the scorer can assume 'higher is better'."""
    if higher_is_better:
        return x, lo, hi
    # flip domain: larger (negative original) is better
    return -x, -hi, -lo

def kinked_score(
    x: float,
    lo: float,
    hi: float,
    m_below: float = -5.0,
    m_in: float = 1.0,
    m_above: float = 0.2,
    higher_is_better: bool = True,
    cap: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Piecewise-affine, continuous, *increasing* scoring function.
      - f(lo) = 0
      - Below lo:  f = m_below * (x - lo)                (m_below should be negative / steep)
      - In range:  f = m_in    * (x - lo)                (positive slope)
      - Above hi:  f = m_in*(hi - lo) + m_above*(x - hi) (flatter positive slope)
    If higher_is_better=False, we flip x and the range so 'higher is better' holds internally.
    """
    if x is None or np.isnan(x):
        return -1e9  # very strong penalty for missing metric

    x, lo, hi = _transform_for_direction(x, lo, hi, higher_is_better)

    # guard: if lo > hi, swap
    if lo > hi:
        lo, hi = hi, lo

    # anchors for continuity
    base_hi = m_in * (hi - lo)

    if x < lo:
        val = m_below * (x - lo)  # negative if m_below<0
    elif x <= hi:
        val = m_in * (x - lo)
    else:
        val = base_hi + m_above * (x - hi)

    if cap is not None:
        lo_cap, hi_cap = cap
        if lo_cap is not None:
            val = max(lo_cap, val)
        if hi_cap is not None:
            val = min(hi_cap, val)
    return float(val)

def make_kinked_quality_fn(
    config: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
) -> Callable[[Dict[str, float]], float]:
    """
    Build a quality(metrics)->score function from a config dict.

    config format (per metric key among {'sharpe','turnover','max_weight','tot_return'}):
        {
          'range': (lo, hi),                 # acceptable range for the metric
          'higher_is_better': True/False,    # default: True
          'm_below': -5.0,                   # steep negative slope below lo
          'm_in': 1.0,                       # positive slope inside [lo, hi]
          'm_above': 0.2,                    # flatter slope beyond hi
          'cap': (min_score, max_score)      # optional clamp on score
        }

    weights (optional): relative importance per metric; if None, equal-weight average.
    """
    # Default directions
    defaults = {
        "sharpe":      {"higher_is_better": True},
        "turnover":    {"higher_is_better": False},
        "max_weight":  {"higher_is_better": False},
        "tot_return":  {"higher_is_better": True},
    }

    # Normalize weights
    if weights is None:
        weights = {k: 1.0 for k in defaults.keys()}
    w_sum = sum(weights.get(k, 0.0) for k in defaults.keys())
    if w_sum <= 0:
        raise ValueError("At least one metric weight must be positive.")
    norm_w = {k: weights.get(k, 0.0) / w_sum for k in defaults.keys()}

    # Freeze a local copy of metric configs with sensible defaults
    metric_cfg = {}
    for k in defaults.keys():
        user_cfg = config.get(k, {})
        if "range" not in user_cfg:
            raise ValueError(f"Missing 'range' for metric '{k}' in config.")
        lo, hi = user_cfg["range"]
        metric_cfg[k] = {
            "lo": float(lo),
            "hi": float(hi),
            "higher_is_better": bool(user_cfg.get("higher_is_better", defaults[k]["higher_is_better"])),
            "m_below": float(user_cfg.get("m_below", -5.0)),
            "m_in": float(user_cfg.get("m_in", 1.0)),
            "m_above": float(user_cfg.get("m_above", 0.2)),
            "cap": user_cfg.get("cap", None),
        }

    def quality(metrics: Dict[str, float]) -> float:
        scores = []
        weights_used = []
        for k in defaults.keys():
            mval = metrics.get({
                "sharpe": "ann_sharpe",
                "turnover": "avg_turnover",
                "max_weight": "max_abs_weight",
                "tot_return": "tot_return",
            }[k], np.nan)

            cfg = metric_cfg[k]
            s = kinked_score(
                x=mval,
                lo=cfg["lo"],
                hi=cfg["hi"],
                m_below=cfg["m_below"],
                m_in=cfg["m_in"],
                m_above=cfg["m_above"],
                higher_is_better=cfg["higher_is_better"],
                cap=cfg["cap"],
            )
            scores.append(s * norm_w[k])
            weights_used.append(norm_w[k])

        # Average of scores over metrics with positive weights
        denom = sum(weights_used)
        return float(sum(scores) / denom) if denom > 0 else -1e9

    return quality