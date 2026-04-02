# Regime Research V2

This note summarizes the public regime workflow used in the repository.

## Core Files

- `src/us_equity/alphas/alpha_hmm_v3.py`
- `src/us_equity/alphas/alpha_hmm_scale_v2.py`
- `src/us_equity/overlays/kalman_overlays_v2.py`
- `src/us_equity/overlays/regime_post_scale_v2.py`
- `notebooks/hmm_alpha_v2.ipynb`

## Workflow Summary

1. Build a robust market series from winsorized cross-sectional log returns.
2. Re-estimate the two-state HMM on rolling or expanding history only.
3. Convert the filtered up-state probability into a dynamic exposure scale.
4. Smooth that scale with a causal Kalman filter and an innovation gate.
5. Apply the scale after weight construction so it interacts properly with portfolio-level controls.
6. Save public regime results under `outputs/regime_v2/`.

## Design Intent

The regime layer is used as an exposure-management tool rather than a standalone return forecast. Its role is to make the base cross-sectional alpha more stable, lower whipsaw, and improve the quality of the equity curve once combined with portfolio construction and risk controls.
