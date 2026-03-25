# US Equity Alpha Research

<p align="center">
  A long-short U.S. equity research sandbox focused on robust cross-sectional signals, regime-aware overlays, and cleaner risk-adjusted implementation.
</p>

<p align="center">
  <a href="notebooks/alpha_mom.ipynb">Momentum Notebook</a> |
  <a href="notebooks/hmm_alpha.ipynb">HMM Overlay</a> |
  <a href="notebooks/ma_crossover.ipynb">MA Crossover</a> |
  <a href="notebooks/function_sets.ipynb">Function Set</a>
</p>

## Research Thesis

The objective is not to maximize raw backtest return. The objective is to build equity signals that remain usable after neutralization, turnover control, and regime scaling. This repo therefore emphasizes:

- robust signal construction on noisy daily cross-sections
- exposure control through HMM and Kalman overlays
- practical portfolio formation rather than pure indicator ranking
- notebook diagnostics that make failure modes visible quickly

## Snapshot

| Item | Detail |
| --- | --- |
| Universe | top `~3000` U.S. common stocks |
| Coverage | `2015-01-01` to `2022-12-31` |
| Frequency | daily |
| Portfolio style | long-short, neutralized, risk-scaled |
| Core themes | EMA trend, HMM regime scaling, Kalman smoothing |

## Performance Snapshot

| Alpha | Annual Sharpe | CAGR | Total Return | Turnover | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA Crossover | `2.65` | `0.327` | `9.128` | `0.447` | `-0.159` |
| HMM Overlay | `2.11` | `0.254` | `5.3707` | `0.3095` | `-0.1686` |
| Kalman Overlay | `2.12` | `0.0838` | `0.93` | `0.110` | `-0.062` |

## Visual Diagnostics

| EMA | HMM | Kalman |
| --- | --- | --- |
| ![EMA](artifacts/figures/ema.png) | ![HMM](artifacts/figures/HMM.png) | ![Kalman](artifacts/figures/kalman.png) |

## Research Stack

| Layer | Paths |
| --- | --- |
| Signal modules | `src/us_equity/alphas/`, `src/us_equity/overlays/`, `src/us_equity/data/` |
| Search utilities | `src/us_equity/search/` |
| Main notebooks | [`notebooks/alpha_mom.ipynb`](notebooks/alpha_mom.ipynb), [`notebooks/alpha_mom_v2.ipynb`](notebooks/alpha_mom_v2.ipynb), [`notebooks/hmm_alpha.ipynb`](notebooks/hmm_alpha.ipynb), [`notebooks/ma_crossover.ipynb`](notebooks/ma_crossover.ipynb), [`notebooks/function_sets.ipynb`](notebooks/function_sets.ipynb) |
| Lightweight reference inputs | `data/reference/tickers.csv`, `data/reference/company_tickers.json` |
| Local research artifacts | `data/market/`, `data/archives/`, `outputs/grid_search/` |

## Pipeline

1. Build an alpha matrix from daily price, volume, or fundamentals.
2. Convert signals into neutralized portfolio weights.
3. Apply volatility targeting or regime overlays.
4. Backtest with transaction costs and score results with the kinked quality function.

## Working Repo

The repo is now structured more cleanly, but the structure is there to support the research rather than dominate it:

- reusable code lives under `src/us_equity/`
- notebooks stay under `notebooks/`
- local market data and grid-search outputs are kept out of the root folder

Run:

```bash
python -m compileall src scripts
python scripts/yf_price_pipeline.py --tickers data/reference/tickers.csv --outdir data/market/yf_data
python scripts/yf_ohlcv_and_meta.py --tickers data/reference/tickers.csv --outdir data/market/yf_data --progress
python -m jupyter lab
```
