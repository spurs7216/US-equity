# Walk-Forward Regime Path

The original HMM and Kalman regime research is preserved for reference, but it uses full-sample HMM parameter estimation and should be treated as biased.

The separate walk-forward-safe path lives here:

- `src/us_equity/alphas/alpha_hmm_walkforward.py`
- `src/us_equity/alphas/alpha_hmm_v2_walkforward.py`
- `src/us_equity/overlays/kalman_overlays_walkforward.py`
- `src/us_equity/overlays/regime_post_scale_walkforward.py`
- `notebooks/hmm_alpha_walkforward.ipynb`

Design choices:

- existing files and old notebook outputs are untouched
- HMM parameters are re-estimated with expanding or rolling history only
- Kalman filtering stays causal
- dynamic threshold fallbacks avoid full-sample quantiles
- new notebook outputs should be written under `outputs/walkforward/`

This keeps the old research visible while providing a clean path for rerunning the regime work without future-data leakage.
