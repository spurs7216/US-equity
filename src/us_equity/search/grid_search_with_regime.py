
import itertools
import inspect
import time
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

    if isinstance(weights_used, pd.DataFrame):
        max_abs_w = float(np.nanmax(np.abs(weights_used.to_numpy())))
    elif isinstance(weights_used, np.ndarray):
        max_abs_w = float(np.nanmax(np.abs(weights_used)))
    else:
        max_abs_w = np.nan

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
            elif isinstance(v, pd.DataFrame):
                out.append((k, ("DataFrame", v.shape, tuple(v.columns))))
            elif isinstance(v, pd.Series):
                out.append((k, ("Series", v.shape, v.name)))
            elif hasattr(v, "shape") and hasattr(v, "dtype"):
                out.append((k, ("array", getattr(v, "shape", None), str(getattr(v, "dtype", None)))))
            else:
                out.append((k, repr(v)))
    return tuple(out)


def _prepare_fast_backtest_inputs(returns: pd.DataFrame) -> Dict[str, Any]:
    return {
        "index": returns.index,
        "columns": returns.columns,
        "returns_values": returns.fillna(0.0).to_numpy(dtype=float, copy=False),
        "tradable": returns.notna().to_numpy(dtype=bool, copy=False),
    }


def _prepare_fast_make_weights_inputs(mask: Optional[pd.DataFrame]) -> Dict[str, Any]:
    return {
        "mask_values": None if mask is None else mask.to_numpy(dtype=bool, copy=False),
    }


def _fast_make_weights_zscore(
    alpha: pd.DataFrame,
    gross_exposure: float = 1.0,
    cap_per_name: float = None,
    neutralize: bool = True,
    winsor_pct: float = 0.01,
    min_names: int = 5,
    prepared: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    x = alpha.to_numpy(dtype=float, copy=True)
    mask_values = None if prepared is None else prepared.get("mask_values")
    valid_rows = np.isfinite(x).any(axis=1)

    if mask_values is not None:
        x = np.where(mask_values, x, np.nan)
        valid_rows = np.isfinite(x).any(axis=1)

    if winsor_pct and winsor_pct > 0:
        q_low = np.full(x.shape[0], np.nan, dtype=float)
        q_high = np.full(x.shape[0], np.nan, dtype=float)
        if valid_rows.any():
            q_low[valid_rows] = np.nanquantile(x[valid_rows], winsor_pct, axis=1)
            q_high[valid_rows] = np.nanquantile(x[valid_rows], 1 - winsor_pct, axis=1)
        x = np.clip(x, q_low[:, None], q_high[:, None])

    mu = np.zeros(x.shape[0], dtype=float)
    sd = np.ones(x.shape[0], dtype=float)
    if valid_rows.any():
        mu[valid_rows] = np.nanmean(x[valid_rows], axis=1)
        sd[valid_rows] = np.nanstd(x[valid_rows], axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (x - mu[:, None]) / sd[:, None]
    invalid_sd = (~valid_rows) | (~np.isfinite(sd)) | (sd == 0)
    x[invalid_sd, :] = 0.0

    if neutralize:
        row_means = np.zeros(x.shape[0], dtype=float)
        finite_after_z = np.isfinite(x).any(axis=1)
        if finite_after_z.any():
            row_means[finite_after_z] = np.nanmean(x[finite_after_z], axis=1)
        x = x - row_means[:, None]

    x = np.nan_to_num(x, nan=0.0)
    if mask_values is not None:
        x = np.where(mask_values, x, 0.0)

    active_counts = np.count_nonzero(x != 0.0, axis=1)
    x[active_counts < int(min_names), :] = 0.0

    abs_sum = np.abs(x).sum(axis=1)
    scale = np.divide(
        gross_exposure,
        abs_sum,
        out=np.zeros_like(abs_sum, dtype=float),
        where=abs_sum > 0,
    )
    w = x * scale[:, None]

    if cap_per_name is not None:
        w = np.clip(w, -cap_per_name, cap_per_name)
        abs_sum = np.abs(w).sum(axis=1)
        scale = np.divide(
            gross_exposure,
            abs_sum,
            out=np.zeros_like(abs_sum, dtype=float),
            where=abs_sum > 0,
        )
        w = w * scale[:, None]

    if mask_values is not None:
        w = np.where(mask_values, w, 0.0)

    return pd.DataFrame(w, index=alpha.index, columns=alpha.columns)


def _fast_backtest_long_short(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float,
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    weights = weights.reindex_like(returns).fillna(0.0)
    w = weights.to_numpy(dtype=float, copy=False)
    tradable = prepared["tradable"]

    w_lag = np.roll(w, 1, axis=0)
    w_lag[0, :] = 0.0
    w_lag = np.where(tradable, w_lag, 0.0)

    pnl = (w_lag * prepared["returns_values"]).sum(axis=1)

    w_now = np.where(tradable, w, 0.0)
    w_prev = np.roll(w, 1, axis=0)
    w_prev[0, :] = 0.0
    w_prev = np.where(tradable, w_prev, 0.0)
    dw = w_now - w_prev
    two_way_turnover = np.abs(dw).sum(axis=1)
    one_way_turnover = 0.5 * two_way_turnover

    cost = one_way_turnover * (cost_bps / 10000.0)
    ret_net = pnl - cost

    idx = prepared["index"]
    return {
        "ret_gross": pd.Series(pnl, index=idx, name="ret_gross"),
        "ret_net": pd.Series(ret_net, index=idx, name="ret_net"),
        "one_way_turnover": pd.Series(one_way_turnover, index=idx, name="one_way_turnover"),
        "two_way_turnover": pd.Series(two_way_turnover, index=idx, name="two_way_turnover"),
        "weights_used": w_lag,
    }

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
    regime_shared_cache: Dict[Tuple[Tuple[str, Any], ...], Any] = {}
    regime_combo_cache: Dict[Tuple[Tuple[Tuple[str, Any], ...], Tuple[Tuple[str, Any], ...]], Any] = {}

    use_fast_backtest = (
        backtest_fn is not None
        and getattr(backtest_fn, "__name__", "") == "backtest_long_short"
        and set((backtest_kwargs or {}).keys()).issubset({"cost_bps"})
    )
    fast_backtest_inputs = _prepare_fast_backtest_inputs(returns) if use_fast_backtest else None
    use_fast_make_weights = (
        make_weights_fn is not None
        and getattr(make_weights_fn, "__name__", "") == "make_weights"
        and (make_weights_kwargs or {}).get("method", "rank") == "zscore"
    )
    fast_make_weights_inputs = _prepare_fast_make_weights_inputs((make_weights_kwargs or {}).get("mask")) if use_fast_make_weights else None

    post_prepare_shared = getattr(post_weights_fn, "prepare_shared", None) if post_weights_fn is not None else None
    post_prepare_combo = getattr(post_weights_fn, "prepare_combo", None) if post_weights_fn is not None else None
    post_apply_prepared = getattr(post_weights_fn, "apply_prepared", None) if post_weights_fn is not None else None
    can_prepare_post_weights = all(callable(fn) for fn in (post_prepare_shared, post_prepare_combo, post_apply_prepared))

    prepared_shared = None
    prepared_shared_key = None
    if can_prepare_post_weights:
        shared_kwargs = dict(post_weights_base_kwargs or {})
        prepared_shared_key = _hashable_key(shared_kwargs, drop_keys=())
        if prepared_shared_key not in regime_shared_cache:
            started = time.perf_counter()
            regime_shared_cache[prepared_shared_key] = post_prepare_shared(**shared_kwargs)
            if progress:
                elapsed = time.perf_counter() - started
                print(f"[regime shared] prepared HMM state in {elapsed:.1f}s")
        prepared_shared = regime_shared_cache[prepared_shared_key]

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
        if use_fast_make_weights:
            weights = _fast_make_weights_zscore(
                alpha,
                gross_exposure=float((make_weights_kwargs or {}).get("gross_exposure", 1.0)),
                cap_per_name=(make_weights_kwargs or {}).get("cap_per_name"),
                neutralize=bool((make_weights_kwargs or {}).get("neutralize", True)),
                winsor_pct=float((make_weights_kwargs or {}).get("winsor_pct", 0.01)),
                min_names=int((make_weights_kwargs or {}).get("min_names", 5)),
                prepared=fast_make_weights_inputs,
            )
        else:
            weights = make_weights_fn(alpha, **make_weights_kwargs)

        # Vol targeting (optional)
        weights_used = weights
        if vol_target_fn is not None and vol_target_kwargs is not None:
            weights_used = vol_target_fn(weights, returns, **vol_target_kwargs)

        alpha_started = time.perf_counter()
        for j, r_combo in enumerate(regime_combos, 1):
            # Post-vol-target transform (regime scaling)
            w_final = weights_used
            if post_weights_fn is not None:
                if can_prepare_post_weights:
                    combo_key = (prepared_shared_key, _hashable_key(r_combo, drop_keys=()))
                    if combo_key not in regime_combo_cache:
                        started = time.perf_counter()
                        regime_combo_cache[combo_key] = post_prepare_combo(prepared_shared, **r_combo)
                        if progress:
                            elapsed = time.perf_counter() - started
                            print(f"    [regime prep {j}/{len(regime_combos)}] {r_combo} in {elapsed:.1f}s")
                    w_final = post_apply_prepared(w_final, regime_combo_cache[combo_key])
                else:
                    kwargs_pw = dict(post_weights_base_kwargs or {})
                    kwargs_pw.update(r_combo)
                    w_final = post_weights_fn(w_final, **kwargs_pw)

            # Backtest
            if use_fast_backtest:
                bt = _fast_backtest_long_short(
                    w_final,
                    returns,
                    cost_bps=float((backtest_kwargs or {}).get("cost_bps", 5.0)),
                    prepared=fast_backtest_inputs,
                )
            else:
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

            if progress and (j == 1 or j % 16 == 0 or j == len(regime_combos)):
                elapsed = time.perf_counter() - alpha_started
                print(f"    [alpha {i}/{len(alpha_combos)} regime {j}/{len(regime_combos)}] score={score:.4f} elapsed={elapsed:.1f}s")

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    if save_csv:
        results_df.to_csv(save_csv, index=False)

    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user("Grid (alpha + regime post-scaling)", results_df)
    except Exception:
        pass

    return results_df, best_payload
