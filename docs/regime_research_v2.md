# Regime Research V2

The original HMM and Kalman regime notebooks are preserved for reference, but the original regime fit uses full-sample HMM parameter estimation and should be treated as preliminary research.

The separate v2 path lives here:

- `src/us_equity/alphas/alpha_hmm_v3.py`
- `src/us_equity/alphas/alpha_hmm_scale_v2.py`
- `src/us_equity/overlays/kalman_overlays_v2.py`
- `src/us_equity/overlays/regime_post_scale_v2.py`
- `notebooks/hmm_alpha_v2.ipynb`

Key differences in v2:

- HMM parameters are re-estimated with expanding or rolling history only.
- The Kalman step remains causal.
- Dynamic thresholds avoid full-sample quantile fallbacks.
- New outputs should be written under `outputs/regime_v2/`.

This keeps the legacy notebook available while providing a cleaner path for rerunning regime research without future-data leakage in the parameter fit.
