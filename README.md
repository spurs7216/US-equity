# US Equity Alpha Research

<p align="center">
  A U.S. equity research repo built around cross-sectional alpha design, causal regime filtering, and practical portfolio construction.
</p>

<p align="center">
  <a href="notebooks/alpha_mom.ipynb">Momentum Notebook</a> |
  <a href="notebooks/hmm_alpha_v2.ipynb">Regime Notebook v2</a> |
  <a href="notebooks/ma_crossover.ipynb">MA Crossover</a> |
  <a href="notebooks/function_sets.ipynb">Research Utilities</a>
</p>

## Research Thesis

The goal is not to maximize raw backtest return. The goal is to build a signal stack that still looks credible after neutralization, turnover control, regime scaling, and volatility management. This repo focuses on three questions:

- can a simple cross-sectional trend signal survive realistic portfolio construction?
- does causal regime filtering improve the timing of gross exposure?
- do layered risk controls improve the quality of the equity curve, not just the headline return?

## Current Public Result

This README reports only the cleaner `v2` regime path in [`notebooks/hmm_alpha_v2.ipynb`](notebooks/hmm_alpha_v2.ipynb). Earlier legacy regime notebooks are kept in the repo for internal reference, but their results are not used here because they relied on full-sample HMM fitting.

`notebooks/hmm_alpha_v2.ipynb` is the notebook worth opening first. `notebooks/hmm_alpha.ipynb` remains in the repo only as a legacy audit trail.

## Snapshot

| Item | Detail |
| --- | --- |
| Universe | top `~3000` U.S. common stocks |
| Coverage | `2015-01-01` to `2022-12-31` |
| Frequency | daily |
| Portfolio style | long-short, beta-aware, volatility-scaled |
| Core modules | EMA trend, HMM regime filter, Kalman filter overlay, drawdown cap |

## Representative v2 Results

Two useful reference points from the current notebook and saved search output:

| Configuration | Sharpe | CAGR | Total Return | Max Drawdown | Avg Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| HMM scale v2 + vol condition + beta neutral + DD cap | `1.81` | `20.7%` | `3.65x` | `-16.9%` | `0.277` |
| Best saved v2 grid result | `2.73` | `23.9%` | `4.76x` | `-11.2%` | `0.167` |

The saved best grid row uses a fast EMA trend signal (`4/9` spans), tighter tail selection (`top_frac=0.05`), dynamic regime mapping, causal Kalman filtering, and a volatility target. The improvement is not just a better regime model. It comes from the full stack working together.

## Equity Curves

The figures below are exported from the executed `v2` notebook and reflect the current public research path.

| Regime-Scaled Stack | Kalman Post-Scale | Best Saved v2 Grid Run |
| --- | --- | --- |
| ![Regime-Scaled Stack](artifacts/figures/regime_v2_stack.png) | ![Kalman Post-Scale](artifacts/figures/regime_v2_kalman.png) | ![Best Saved v2 Grid Run](artifacts/figures/regime_v2_best.png) |

## Why The Stack Helps

- **Cross-sectional EMA trend** keeps the base alpha simple and fast-moving. The signal only needs to rank stocks better than the cross-section, not predict exact returns.
- **Tail selection and robust scaling** reduce the impact of small noisy scores. Concentrating on the strongest names raises signal-to-noise.
- **HMM regime probabilities** translate market-state information into exposure control. The model is used as a sizing layer, not as a standalone directional forecast.
- **Kalman filtering** smooths noisy regime probabilities without using future data. In practice it helps reduce whipsaw, turnover, and abrupt leverage changes.
- **Vol targeting and drawdown caps** keep the strategy from overexposing itself exactly when the signal becomes unstable.

The main lesson from the current runs is that Kalman filtering is useful, but mostly as a stabilizer. The biggest performance lift appears when it is combined with a stronger alpha specification, dynamic regime scaling, and portfolio-level risk control.

## Research Workflow

1. Build daily panels from Yahoo Finance and reference files.
2. Generate a cross-sectional alpha from price or fundamentals.
3. Convert signals into neutralized portfolio weights.
4. Apply regime scaling, volatility targeting, and drawdown control.
5. Score the result with return, Sharpe, turnover, and concentration penalties.
6. Review notebook diagnostics before treating a result as a real research lead.

## Repo Map

- `src/us_equity/alphas/`: alpha and regime modules
- `src/us_equity/overlays/`: portfolio overlays and scaling logic
- `src/us_equity/search/`: grid-search and quality scoring utilities
- `notebooks/`: main research notebooks
- `data/reference/`: ticker and metadata inputs
- `outputs/regime_v2/`: saved results from the clean v2 regime search

## Run

```bash
python -m compileall src scripts
python scripts/yf_price_pipeline.py --tickers data/reference/tickers.csv --outdir data/market/yf_data
python scripts/yf_ohlcv_and_meta.py --tickers data/reference/tickers.csv --outdir data/market/yf_data --progress
python -m jupyter lab
```
