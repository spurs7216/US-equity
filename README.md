# Long-short Neutralized Equity Alpha Research (US Top 3000)

by Joseph Shih, Sep 2025

This repository studies a daily, cross-sectional U.S. equity universe designed to cover large-, mid-, and small-cap names. The repo stays flat on purpose: the notebooks and Python modules reference one another and use local relative paths.

## Data Overview
- Universe: top ~3000 U.S. common stocks by float-adjusted market cap, with dynamic membership
- Time span: `2015-01-01` to `2022-12-31`
- Frequency: daily bars
- Core fields: `open`, `high`, `low`, `close`, `volume`, plus sector and industry mappings
- Derived fields: log returns, rolling robust volatility, and tradability masks

## Repository Layout
- `alpha_*.py`, `risk_overlays.py`, `kalman_overlays.py`, `simfin_loading.py`, `sector_loader.py`: core research modules
- `grid_search_*.py`: parameter search helpers
- `alpha_mom.ipynb`, `alpha_mom_v2.ipynb`, `hmm_alpha.ipynb`, `ma_crossover.ipynb`, `function_sets.ipynb`: main notebooks
- `image/`: checked-in plot assets
- `yf_data/`, `close.parquet`, `yf_data.zip`, `grid_results*.csv`: local data or generated artifacts excluded by `.gitignore`

## Pipeline
1. Build an alpha matrix.
2. Convert signals to neutralized portfolio weights.
3. Optionally apply vol targeting or regime overlays.
4. Backtest with costs and score results with the quality function.

## Main Alphas
### Alpha 1: EMA Crossover
Momentum signal on log prices with robust tail gating and inverse-vol scaling.

Performance:
- Annual Sharpe: `2.65`
- CAGR: `0.327`
- Total Return: `9.128`
- Turnover: `0.447`
- Maximum Drawdown: `-0.159`
- Max Weight: `0.0158`

![EMA PnL](https://github.com/spurs7216/US-equity/blob/master/image/ema.png)

### Alpha 2: Hidden Markov Model Overlay
Two-state Gaussian HMM on a robust market proxy; exposure is scaled rather than flipped in weak regimes.

Performance:
- Annual Sharpe: `2.11`
- CAGR: `0.254`
- Total Return: `5.3707`
- Turnover: `0.3095`
- Maximum Drawdown: `-0.1686`

![HMM PnL](https://github.com/spurs7216/US-equity/blob/master/image/HMM.png)

### Alpha 3: Kalman Filter Overlay
Forward-only Kalman smoothing on the HMM exposure scale to reduce regime jitter.

Performance:
- Annual Sharpe: `2.12`
- CAGR: `0.0838`
- Total Return: `0.93`
- Turnover: `0.110`
- Maximum Drawdown: `-0.062`
- Max Weight: `0.00988`

![Kalman PnL](https://github.com/spurs7216/US-equity/blob/master/image/kalman.png)

## Notes
- `ma_crossover.ipynb` was preserved from the existing repo because it differs from the old workspace copy and appears newer.
- `function_sets-Copy1.ipynb` is treated as local scratch and ignored.
- `tickers.csv` and `company_tickers.json` remain versioned because they are lightweight inputs rather than large generated outputs.
