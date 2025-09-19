# Long-short neutralized Equity Alpha Research (US Top 3000)

by Joseph Shih, Sep 2025

## 📌 **Data Overview**
This repository studies a daily, cross-sectional equity universe designed to reflect broad U.S. large/mid/small-cap exposure.
*  Universe: Top ~3000 U.S. common stocks by float-adjusted market cap (dynamic membership)
*  Time span: 2015-01-01 → 2022-12-31 (trading days only)
*  Frequency: Daily bars
*  Features (core price/volume):
    *  open, high, low, close, volume
    *  Corporate-action–adjusted prices (splits/dividends)
    *  Sector / Industry mappings for optional group-neutrality
  
*  Derived fields (used by the alphas/metrics):
    *  Log returns
    *  Rolling robust volatility (MAD/EWMAD)
    *  Tradability masks (missing/halts/stale quotes)

Data hygiene & robustness
*  Heavy-tail guardrails: MAD-based winsorization on returns; hard caps for clearly erroneous moves (e.g., ±50% one-day spikes from bad prints).
*  Stale-quote filter: drop names with sub-1bp rolling std over a short lookback.
*  Dynamic universe handling: missing data → masked; weights normalized on the active set.

Pipeline
1.  Build alpha (e.g., robust EMA crossover on log prices) → alpha matrix
2.  Make weights (z-score / rank, caps, neutralize, inverse-vol) → dollar-neutral
3.  (Optional) Vol targeting to desired portfolio risk.
4.  Backtest with costs; compute metrics; grid search with a kinked quality function (Sharpe ↑, turnover/max-weight ↓, total return ↑)


## 📌 **Alpha 1 — EMA Crossover**
Momentum signal computed on log prices with robust gating and inverse-vol scaling.
1. Uses trend continuation but avoids over-trading by acting only on clear separations from the cross-sectional crowd.
2. Inverse-vol normalization stabilizes risk contribution across names.

Counstrcution
1.  EMAs on log price:  $S_t$= $EMA_t$(fast)− $EMA_t$(slow)
2.  Tail gating (cross-sectional): Trade only names where ∣ $S_t$∣ is in the top-tail (e.g., ≥ 95th pct), with a min breadth fallback.
3.  Robust risk scaling: Multiply by 1/σ where σ is a rolling MAD/EWMAD of returns (per name).
4.  Sanity masks: zero out non-tradable names (missing/stale/halts).

Performance
| Metric | Num |
| --- | --- |
| Annual Sharpe | 2.65 |
| CAGR| 0.327|
| Total Return | 9.128 |
| Turnover| 0.447 |
| Maximum Drawdown| -0.159 |
| Max Weight| 0.0158 |

![alt text](https://github.com/spurs7216/US-equity/blob/master/image/ema.png "PnL line")

## 📌 **Alpha 2 — Hidden Markov Model (HMM) Regime Overlay**

A 2-state Gaussian HMM is fit to a robust market proxy (cross-sectional median of winsorized log returns). Instead of flipping the signal in risk-off, we scale exposure between 0→1 based on the probability of the “up/trend” state.
1.  In draws/vol spikes, we shrink, not invert; in steady “risk-on” regimes, we run full size.

Construction
1. Build a market series $m_t$ = median of cross-sectional, winsorized log returns.
2. Fit a 2-state HMM (EM / forward-backward, scaled), order states so $μ_0$ < $μ_1$.
3. Compute filtered $p_{up}$(t)= $Pr$($S_t$ = up ∣ $m_{1:t}$).
4. Map $p_{up}$(t) to exposure scale $s_t$ ∈ [0,1] with:
   *  Fixed thresholds (e.g., 0.45/0.60), or
   *  Dynamic thresholds (rolling quantiles, e.g., 40%/60%) to keep the partial zone active over time.
5.  Apply post vol-target: $w_t^{final}$ = $s_t$ ⋅ $w_t^{volTarget}$.

Performance
| Metric | Num |
| --- | --- |
| Annual Sharpe | 2.11 |
| CAGR| 0.254|
| Total Return | 5.3707 |
| Turnover| 0.3095 |
| Maximum Drawdown| -0.1686 |

![alt text](https://github.com/spurs7216/US-equity/blob/master/image/HMM.png "PnL line")

## 📌 **Alpha 3 — Kalman Filter Overlay (Forward-Only)**
A scalar Kalman filter smooths the HMM exposure scale $s_t$.
1.  Reduces regime jitter without laggy moving averages; integrates naturally with HMM to stabilize gross exposure in choppy tapes.

Construction
1.  State model: $x_t$= $x_{t−1}$+ $w_t$ , where $w_t$ ∼N(0,q)
2.  Observation: $s_t^{raw}$= $x_t$+ $v_t$ , where $v_t$ ∼N(0,r)
3.  Tunable reactivity via q/r: larger q → faster response; larger r → more smoothing.
4.  Innovation gating (optional): large standardized innovations ∣ $z_t$∣ imply regime transition uncertainty → clamp exposure to a floor (e.g., 30%) for that day.

Performance
| Metric | Num |
| --- | --- |
| Annual Sharpe | 2.12 |
| CAGR| 0.0838|
| Total Return | 0.93|
| Turnover| 0.110 |
| Maximum Drawdown| -0.062 |
| Max Weight| 0.00988 |

![alt text](https://github.com/spurs7216/US-equity/blob/master/image/kalman.png "PnL line")

