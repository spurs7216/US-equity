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

    one_way_to = backtest_out.get("two_way_turnover")
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


def grid_search_alpha(
    alpha_fn: Callable[..., pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    make_weights_fn: Callable[..., pd.DataFrame],
    make_weights_kwargs: Dict[str, Any],
    backtest_fn: Callable[..., Dict[str, pd.Series]],
    backtest_kwargs: Dict[str, Any],
    vol_target_fn: Optional[Callable[..., pd.DataFrame]] = None,
    vol_target_kwargs: Optional[Dict[str, Any]] = None,
    returns: Optional[pd.DataFrame] = None,
    quality_fn: Callable[[Dict[str, float]], float] = None,
    alpha_static_kwargs: Optional[Dict[str, Any]] = None,
    progress: bool = True,
    save_csv: Optional[str] = "./grid_results.csv",
    return_artifacts: bool = True,
):
    if quality_fn is None:
        raise ValueError("You must provide a quality_fn(metrics)->score.")

    if returns is None:
        returns = close.pct_change(fill_method=None)

    combos = _flatten_product(param_grid)
    results = []
    best_score = -np.inf
    best_payload = {}

    for i, combo in enumerate(combos, 1):
        params = dict(combo)
        if alpha_static_kwargs:
            params.update(alpha_static_kwargs)

        sig = inspect.signature(alpha_fn)
        if progress:
            print(f"[{i}/{len(combos)}] {params}")
        
        if "close" in sig.parameters: params["close"] = close
        if "volume" in sig.parameters: params["volume"] = volume



        alpha, diag = _call_alpha(alpha_fn, params)

        weights = make_weights_fn(alpha, **make_weights_kwargs)

        weights_used = weights
        if vol_target_fn is not None and vol_target_kwargs is not None:
            weights_used = vol_target_fn(weights, returns, **vol_target_kwargs)

        bt = backtest_fn(weights_used, returns, **backtest_kwargs)
        ret_net = bt.get("ret_net", None)
        if ret_net is None:
            raise RuntimeError("backtest_fn must return a dict containing 'ret_net' Series.")
        metrics = _compute_metrics(ret_net, bt, bt.get("weights_used", weights_used.shift(1)))
        score = float(quality_fn(metrics))

        row = {**combo, "score": score, **metrics}
        results.append(row)

        if score > best_score:
            best_score = score
            best_payload = {
                "params": combo.copy(),
                "metrics": metrics,
                "score": score,
                "alpha": alpha if return_artifacts else None,
                "weights": weights_used if return_artifacts else None,
                "ret_net": ret_net if return_artifacts else None,
                "diag": diag,
            }

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    if save_csv:
        results_df.to_csv(save_csv, index=False)

    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Grid search results (sorted by score)", results_df)
    except Exception:
        pass

    return results_df, best_payload


def example_quality_fn(metrics: Dict[str, float]) -> float:
    sr = metrics.get("ann_sharpe", 0.0)
    tr = metrics.get("tot_return", 0.0)
    to = metrics.get("avg_turnover", 0.0)
    mw = metrics.get("max_abs_weight", 0.0)
    pen_to = 2.0 * to
    pen_mw = 5.0 * max(0.0, mw - 0.03)
    return 1.5*sr + 0.5*tr - pen_to - pen_mw
