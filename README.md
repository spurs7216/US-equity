# US Equity Alpha Research

<p align="center">
  A modular U.S. equity research repo for building long-short alphas, testing regime overlays, and keeping the research pipeline readable as it scales.
</p>

<p align="center">
  <a href="notebooks/ma_crossover.ipynb">MA Crossover</a> |
  <a href="notebooks/hmm_alpha.ipynb">HMM Overlay</a> |
  <a href="notebooks/alpha_mom.ipynb">Momentum Blend</a> |
  <a href="notebooks/interview_cheatsheet.ipynb">Interview Notes</a>
</p>

## Research Thesis

The repo is organized like a compact research platform rather than a scratch directory. Reusable signal logic lives in `src/`, notebooks stay in `notebooks/`, and local market artifacts are pushed into `data/`, `outputs/`, and `artifacts/`.

The aim is to study signals that still look credible after neutralization, turnover control, and regime-aware sizing. That bias shows up in the code structure: alpha generation, overlays, search logic, and diagnostics are separated so each layer can be changed independently.

## Repository Layout

```text
.
|- src/us_equity/
|  |- alphas/
|  |- overlays/
|  |- search/
|  \- data/
|- notebooks/
|- scripts/
|- data/
|  |- reference/
|  |- market/
|  \- archives/
|- outputs/grid_search/
|- artifacts/figures/
\- docs/
```

## Quick Start

```bash
python -m compileall src scripts
python scripts/yf_price_pipeline.py --tickers data/reference/tickers.csv --outdir data/market/yf_data
python scripts/yf_ohlcv_and_meta.py --tickers data/reference/tickers.csv --outdir data/market/yf_data --progress
python -m jupyter lab
```

Open notebooks from `notebooks/`. Each notebook adds `src/` to `sys.path` automatically, so imports keep working after the restructure.

## Research Stack

- `src/us_equity/alphas/`: crossover, HMM, and blended momentum signal code.
- `src/us_equity/overlays/`: Kalman and post-weight regime scaling, beta and drawdown controls.
- `src/us_equity/search/`: grid search utilities and the kinked objective.
- `src/us_equity/data/`: sector and SimFin loaders.
- `artifacts/figures/`: README figures.
- `docs/interview_cheatsheet.md`: narrative summary of the repo for interview prep.
