
import itertools
import inspect
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple, Optional

def _flatten_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for tpl in itertools.product(*vals):
        combos.append({k: v for k, v in zip(keys, tpl)})
    return combos

def _call_alpha(alpha_fn: Callable, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    out = alpha_fn(**params)
    if isinstance(out, tuple) and len(out) >= 1:
        alpha = out[0]
        diag = out[1] if len(out) >= 2 else None
    else:
        alpha = out
        diag = None
    if not isinstance(alpha, pd.DataFrame):
        raise TypeError("alpha_fn must return a pandas DataFrame or (DataFrame, diag)")
    return alpha, diag

def _compute_metrics(ret_series: pd.Series, backtest_out: Dict[str, pd.Series], weights_used: pd.DataFrame) -> Dict[str, float]:
    ann = 252.0
    r = ret_series.dropna()
    mu = r.mean()
    sd = r.std(ddof=1)
    sharpe = float((mu / sd) * np.sqrt(ann)) if (sd is not None and sd != 0 and not np.isnan(sd)) else np.nan
    equity = (1.0 + r).cumprod()
    years = len(r) / ann
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and len(equity) else np.nan
    tot_return = float(equity.iloc[-1] - 1) if len(equity) else np.nan
    dd = (equity / equity.cummax() - 1.0).min() if len(equity) else np.nan

    one_way_to = backtest_out.get("one_way_turnover")
    avg_turnover = float(one_way_to.mean()) if one_way_to is not None else np.nan

    max_abs_w = float(np.nanmax(np.abs(weights_used.to_numpy()))) if isinstance(weights_used, pd.DataFrame) else np.nan

    return {
        "ann_sharpe": sharpe,
        "cagr": cagr,
        "tot_return": tot_return,
        "max_drawdown": float(dd) if dd == dd else np.nan,
        "avg_turnover": avg_turnover,
        "max_abs_weight": max_abs_w,
    }


def _hashable_key(d: Dict[str, Any], drop_keys: Tuple[str, ...] = ("close","volume")) -> Tuple[Tuple[str, Any], ...]:
    out = []
    for k, v in sorted(d.items(), key=lambda kv: kv[0]):
        if k in drop_keys:
            continue
        try:
            hash(v)
            out.append((k, v))
        except TypeError:
            if isinstance(v, (list, tuple)):
                out.append((k, tuple(v)))
            elif isinstance(v, dict):
                out.append((k, tuple(sorted(v.items()))))
            elif isinstance(v, _pd.DataFrame):
                out.append((k, ("DataFrame", v.shape, tuple(v.columns))))
            elif isinstance(v, _pd.Series):
                out.append((k, ("Series", v.shape, v.name)))
            elif hasattr(v, "shape") and hasattr(v, "dtype"):
                out.append((k, ("array", getattr(v, "shape", None), str(getattr(v, "dtype", None)))))
            else:
                out.append((k, repr(v)))
    return tuple(out)

def grid_search_with_post_weights(
    alpha_fn: Callable[..., pd.DataFrame],
    alpha_grid: Dict[str, List[Any]],
    regime_post_grid: Dict[str, List[Any]],
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    make_weights_fn: Callable[..., pd.DataFrame],
    make_weights_kwargs: Dict[str, Any],
    backtest_fn: Callable[..., Dict[str, pd.Series]],
    backtest_kwargs: Dict[str, Any],
    vol_target_fn: Optional[Callable[..., pd.DataFrame]] = None,
    vol_target_kwargs: Optional[Dict[str, Any]] = None,
    # post-vol-target transform
    post_weights_fn: Optional[Callable[..., pd.DataFrame]] = None,
    post_weights_base_kwargs: Optional[Dict[str, Any]] = None,
    # returns for backtest
    returns: Optional[pd.DataFrame] = None,
    # quality function
    quality_fn: Callable[[Dict[str, float]], float] = None,
    # fixed kwargs for alpha that are not searched
    alpha_static_kwargs: Optional[Dict[str, Any]] = None,
    progress: bool = True,
    save_csv: Optional[str] = "./grid_results_regime.csv",
    return_artifacts: bool = True,
):

    if quality_fn is None:
        raise ValueError("You must provide a quality_fn(metrics)->score.")

    if returns is None:
        returns = close.pct_change(fill_method=None)

    alpha_combos = _flatten_product(alpha_grid) if alpha_grid else [dict()]
    regime_combos = _flatten_product(regime_post_grid) if regime_post_grid else [dict()]

    results = []
    best_score = -np.inf
    best_payload = {}

    # cache alpha by its params to avoid re-compute across regime combos
    alpha_cache: Dict[Tuple[Tuple[str, Any], ...], pd.DataFrame] = {}

    for i, a_combo in enumerate(alpha_combos, 1):
        a_params = dict(a_combo)
        if alpha_static_kwargs:
            a_params.update(alpha_static_kwargs)

        # Inject data args
        sig = inspect.signature(alpha_fn)

        # Build/reuse alpha
        a_key = _hashable_key(a_params)
        if a_key not in alpha_cache:
            tmp_para = a_params.copy()
            tmp_para.pop("close", None)
            tmp_para.pop("volume", None)

            if progress:
                print(f"[{i}/{len(alpha_combos)}] {tmp_para}")

            alpha, _diag = _call_alpha(alpha_fn, a_params)
            alpha_cache[a_key] = alpha
        else:
            alpha = alpha_cache[a_key]
        
        if "close" in sig.parameters: a_params["close"] = close
        if "volume" in sig.parameters: a_params["volume"] = volume
        # Make base weights
        weights = make_weights_fn(alpha, **make_weights_kwargs)

        # Vol targeting (optional)
        weights_used = weights
        if vol_target_fn is not None and vol_target_kwargs is not None:
            weights_used = vol_target_fn(weights, returns, **vol_target_kwargs)

        for j, r_combo in enumerate(regime_combos, 1):
            # Post-vol-target transform (regime scaling)
            w_final = weights_used
            if post_weights_fn is not None:
                kwargs_pw = dict(post_weights_base_kwargs or {})
                kwargs_pw.update(r_combo)
                w_final = post_weights_fn(w_final, **kwargs_pw)

            # Backtest
            bt = backtest_fn(w_final, returns, **backtest_kwargs)
            ret_net = bt.get("ret_net", None)
            if ret_net is None:
                raise RuntimeError("backtest_fn must return a dict containing 'ret_net' Series.")
            metrics = _compute_metrics(ret_net, bt, bt.get("weights_used", w_final.shift(1)))
            score = float(quality_fn(metrics))

            row = {**{f"a_{k}":v for k,v in a_combo.items()}, **{f"r_{k}":v for k,v in r_combo.items()}, "score": score, **metrics}
            results.append(row)

            if score > best_score:
                best_score = score
                best_payload = {
                    "alpha_params": a_combo.copy(),
                    "regime_params": r_combo.copy(),
                    "metrics": metrics,
                    "score": score,
                    "alpha": alpha if return_artifacts else None,
                    "weights": w_final if return_artifacts else None,
                    "ret_net": ret_net if return_artifacts else None,
                }

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    if save_csv:
        results_df.to_csv(save_csv, index=False)

    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Grid (alpha + regime post-scaling)", results_df)
    except Exception:
        pass

    return results_df, best_payload
