# US Equity Alpha Research

<p align="center">
  A U.S. equity research repo focused on cross-sectional alpha design, causal regime filtering, and practical portfolio construction.
</p>

<p align="center">
  <a href="notebooks/alpha_mom.ipynb">Momentum Notebook</a> |
  <a href="notebooks/hmm_alpha_v2.ipynb">Regime Notebook v2</a> |
  <a href="notebooks/ma_crossover.ipynb">MA Crossover</a> |
  <a href="notebooks/function_sets.ipynb">Research Utilities</a>
</p>

## Research Thesis

This repository studies whether a simple cross-sectional trend signal can be made materially more robust through regime-aware exposure control and disciplined portfolio construction. The research is organized around three questions:

- can a simple cross-sectional trend signal survive realistic portfolio construction?
- does causal regime filtering improve exposure timing?
- do layered portfolio controls improve the quality of the equity curve, not only the headline return?

## Public Research Path

The published regime results in this repository are drawn from [`notebooks/hmm_alpha_v2.ipynb`](notebooks/hmm_alpha_v2.ipynb) and the saved outputs under `outputs/regime_v2/`. The workflow uses only causal HMM re-estimation and causal Kalman filtering, so the reported curves and tables reflect the current public research path.

## Snapshot

| Item | Detail |
| --- | --- |
| Universe | top `~3000` U.S. common stocks |
| Coverage | `2015-01-01` to `2022-12-31` |
| Frequency | daily |
| Portfolio style | long-short, beta-aware, volatility-scaled |
| Core modules | EMA trend, HMM regime filter, Kalman filter overlay, drawdown cap |

## Representative Results

Three reference points summarize the current `v2` research path:

| Configuration | Sharpe | CAGR | Total Return | Max Drawdown | Avg Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Representative v2 stack: regime scale + vol condition + beta neutralization + drawdown cap | `1.81` | `20.7%` | `3.65x` | `-16.9%` | `0.277` |
| Kalman post-scale overlay example | `1.44` | `9.5%` | `1.09x` | `-13.7%` | `0.184` |
| Best saved v2 configuration from grid search | `2.73` | `23.9%` | `4.76x` | `-11.2%` | `0.167` |

These three rows serve different purposes. The representative stack shows the core regime-aware pipeline. The Kalman post-scale example isolates how smoother exposure control can reduce turnover and drawdown, even if it does not maximize return on its own. The best saved configuration shows the full improvement once regime smoothing is combined with a stronger alpha specification, tighter tail selection, and post-weight volatility targeting.

## Equity Curves

The figures below are exported from the executed `v2` notebook and reflect the current public research path.

| Representative v2 Stack | Kalman Post-Scale Overlay | Best Saved v2 Configuration |
| --- | --- | --- |
| ![Representative v2 Stack](artifacts/figures/regime_v2_stack.png) | ![Kalman Post-Scale Overlay](artifacts/figures/regime_v2_kalman.png) | ![Best Saved v2 Configuration](artifacts/figures/regime_v2_best.png) |

## Why The Methods Help

- **Cross-sectional EMA trend** provides a simple base signal that reacts quickly to intermediate-horizon trend persistence. The objective is relative ranking across stocks, not point forecasting of returns.
- **Tail selection and robust scaling** concentrate capital in the strongest names and reduce the impact of small, noisy scores. This improves signal-to-noise and keeps weak convictions from dominating turnover.
- **HMM regime probabilities** convert aggregate market conditions into an exposure-scaling signal. In this framework the HMM is a sizing layer, not a standalone directional model.
- **Kalman filtering** smooths noisy regime probabilities without introducing forward-looking information. Its primary contribution is to reduce whipsaw, abrupt leverage changes, and unstable sizing decisions.
- **Volatility targeting, beta control, and drawdown caps** act at the portfolio level. These overlays are designed to keep exposure aligned with current signal quality and to prevent adverse environments from overwhelming the alpha.

The main empirical conclusion from the current runs is that Kalman filtering improves the stack primarily as a stabilizer rather than as a standalone source of alpha. The strongest results appear when regime smoothing is combined with a stronger alpha specification, dynamic regime scaling, and post-weight risk management.

## Methodology

The alpha construction process is organized as a sequence of explicit steps:

1. **Build the panel**  
   Load daily OHLCV data and reference files, align the stock panel, and remove obviously stale observations before signal construction.
2. **Form the base alpha**  
   Compute a cross-sectional EMA trend signal from price history, then normalize it with robust scaling so the signal reflects relative strength rather than raw price level.
3. **Select the tradeable tail**  
   Apply tradability filters and keep only the strongest part of the cross-section, which concentrates capital where the signal is most informative.
4. **Create initial portfolio weights**  
   Convert scores into capped, normalized long-short weights and neutralize broad market exposure so the portfolio reflects stock selection rather than simple market beta.
5. **Estimate the regime state**  
   Build a robust market-level return series, fit a two-state HMM using only information available at each date, and convert the filtered regime probability into an exposure signal.
6. **Stabilize exposure with Kalman filtering**  
   Smooth the raw regime scale with a causal Kalman filter and an innovation gate to avoid abrupt leverage changes during noisy state transitions.
7. **Apply portfolio-level risk controls**  
   Combine regime scaling with volatility conditioning, volatility targeting, beta control, and drawdown caps so exposure falls when the environment becomes less reliable.
8. **Evaluate and search**  
   Backtest with transaction costs, review equity-curve diagnostics, and use a quality function that penalizes turnover and concentration alongside return and Sharpe.

## Repository Structure

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
