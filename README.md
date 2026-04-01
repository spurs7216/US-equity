# US Equity Alpha Research

<p align="center">
  A U.S. equity research sandbox focused on cross-sectional signals, regime-aware overlays, and practical portfolio construction.
</p>

<p align="center">
  <a href="notebooks/alpha_mom.ipynb">Momentum Notebook</a> |
  <a href="notebooks/hmm_alpha.ipynb">HMM Overlay (Legacy)</a> |
  <a href="notebooks/hmm_alpha_v2.ipynb">HMM Overlay v2</a> |
  <a href="notebooks/ma_crossover.ipynb">MA Crossover</a> |
  <a href="notebooks/function_sets.ipynb">Function Set</a>
</p>

## Research Thesis

The objective is not to maximize raw backtest return. The objective is to build equity signals that remain usable after neutralization, turnover control, and regime scaling. This repo therefore emphasizes:

- robust signal construction on noisy daily cross-sections
- exposure control through regime probabilities and causal Kalman filtering
- practical portfolio formation rather than pure indicator ranking
- notebook diagnostics that make failure modes visible quickly
- versioned research paths when a notebook needs a cleaner rerun

## Snapshot

| Item | Detail |
| --- | --- |
| Universe | top `~3000` U.S. common stocks |
| Coverage | `2015-01-01` to `2022-12-31` |
| Frequency | daily |
| Portfolio style | long-short, neutralized, risk-scaled |
| Core themes | EMA trend, momentum, regime overlays |

## Research Status

- `notebooks/hmm_alpha.ipynb` is preserved as the original regime notebook for reference.
- `notebooks/hmm_alpha_v2.ipynb` is the cleaner rerun path with expanding or rolling HMM re-estimation before the Kalman filter step.
- Until the v2 notebook is rerun end-to-end, regime metrics in this repo should be treated as preliminary research rather than bias-free portfolio evidence.

## Visual Diagnostics

The regime figures below are legacy diagnostics kept for comparison while the v2 regime path is being rerun.

| EMA | HMM | Kalman |
| --- | --- | --- |
| ![EMA](artifacts/figures/ema.png) | ![HMM](artifacts/figures/HMM.png) | ![Kalman](artifacts/figures/kalman.png) |

## Research Stack

| Layer | Paths |
| --- | --- |
| Signal modules | `src/us_equity/alphas/`, `src/us_equity/overlays/`, `src/us_equity/data/` |
| Search utilities | `src/us_equity/search/` |
| Main notebooks | [`notebooks/alpha_mom.ipynb`](notebooks/alpha_mom.ipynb), [`notebooks/alpha_mom_v2.ipynb`](notebooks/alpha_mom_v2.ipynb), [`notebooks/hmm_alpha.ipynb`](notebooks/hmm_alpha.ipynb), [`notebooks/hmm_alpha_v2.ipynb`](notebooks/hmm_alpha_v2.ipynb), [`notebooks/ma_crossover.ipynb`](notebooks/ma_crossover.ipynb), [`notebooks/function_sets.ipynb`](notebooks/function_sets.ipynb) |
| Lightweight reference inputs | `data/reference/tickers.csv`, `data/reference/company_tickers.json` |
| Local research artifacts | `data/market/`, `data/archives/`, `outputs/grid_search/` |

## Pipeline

1. Build an alpha matrix from daily price, volume, or fundamentals.
2. Convert signals into neutralized portfolio weights.
3. Apply volatility targeting or regime overlays.
4. Backtest with transaction costs and score results with the kinked quality function.
5. Review diagnostics before treating any result as a credible research lead.

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
