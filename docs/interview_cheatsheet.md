# US Equity Interview Cheat Sheet

## 30-Second Pitch

I built a modular long-short U.S. equity research stack that separates alpha generation, risk control, and model selection. The repo focuses on cross-sectional signals, regime-aware overlays, and explicit implementation penalties, so I am not just maximizing backtest return, I am optimizing for deployable signal quality.

Core files:
- `alpha_ma_crossover.py`
- `alpha_hmm.py`
- `alpha_fund_momentum.py`
- `risk_overlays.py`
- `kalman_overlays.py`
- `grid_search_pipeline.py`
- `quality_kinked.py`

## Research Pipeline

1. Build clean panel data from prices, volume, and fundamentals.
2. Generate a raw alpha matrix.
3. Cross-sectionally gate tails so only strong names are traded.
4. Risk-scale positions by volatility.
5. Neutralize exposures and optionally apply regime overlays.
6. Backtest net returns with turnover and drawdown tracking.
7. Grid-search parameters using a quality function that rewards Sharpe and return while penalizing turnover and concentration.

## Key Math

### 1. Moving-Average Crossover

\[
S_{i,t}=\frac{\text{MA}^{short}_{i,t}-\text{MA}^{long}_{i,t}}{|\text{MA}^{long}_{i,t}|}
\]

Tail gating:

\[
\mathcal{K}_t=\{i:\ |S_{i,t}|\ge q_t(\tau)\}
\]

Vol scaling:

\[
\alpha_{i,t}=\operatorname{sign}(S_{i,t})\cdot \frac{1}{\text{ATR}_{i,t}}
\]

### 2. Fundamental Momentum

Price momentum:

\[
Z^{price}_{i,t}=\frac{\log P_{i,t-S}-\log P_{i,t-L}}{\sigma^{EWMAD}_{i,t}}
\]

Fundamental blend:

\[
Z^{fund}_{i,t}=\frac{\sum_k w_k Z^{(k)}_{i,t}}{\sum_k |w_k|}
\]

Combined alpha:

\[
\alpha_{i,t}=Z^{price}_{i,t}\bigl(1+\lambda Z^{fund}_{i,t}\bigr)
\]

### 3. Robust Cross-Sectional Z-Score

\[
z_i=\frac{x_i-\operatorname{median}(x)}{1.4826\cdot \operatorname{MAD}(x)}
\]

This is important because equity cross-sections are heavy-tailed and outlier-sensitive.

### 4. HMM Regime Overlay

Robust market proxy:

\[
m_t=\operatorname{median}_i(\text{winsorized log return}_{i,t})
\]

Fit a 2-state Gaussian HMM by EM and compute:

\[
p_t=\Pr(s_t=\text{up}\mid m_{1:t})
\]

Map to regime score:

\[
r_t=2p_t-1
\]

Apply to base alpha:

\[
\alpha^{regime}_{i,t}=r_t\alpha^{base}_{i,t}
\]

### 5. Beta Neutralization

\[
w'_t=w_t-\frac{w_t^\top \beta_t}{\beta_t^\top \beta_t}\beta_t
\]

Then re-center to dollar neutrality and rescale gross exposure.

### 6. Volatility and Drawdown Overlays

Market-vol scaling:

\[
s_t=\operatorname{clip}(1-\gamma z_t^+, s_{\min}, 1)
\]

Drawdown-aware scaling:

\[
DD_t=\frac{EQ_t}{\max_{u\le t} EQ_u}-1
\]

Reduce exposure when drawdown breaches a threshold.

### 7. Kalman Filter Overlay

State:

\[
x_t=x_{t-1}+w_t
\]

Observation:

\[
y_t=x_t+v_t
\]

Used to apply a causal Kalman filter to the HMM-derived exposure scale and reduce regime jitter.

## Why This Looks Like Good Research

- Uses robust statistics: median, MAD, winsorization.
- Separates alpha generation from risk overlays and portfolio construction.
- Evaluates implementation metrics, not just return.
- Includes explicit penalties for turnover and concentration.
- Treats regime detection as a sizing problem, not only a directional forecast.

## Good Interview Lines

- I prefer robust estimators because equity cross-sections are heavy-tailed and naive moments can be dominated by outliers.
- I optimize for deployable alpha quality, not raw in-sample return.
- I treat regime models as exposure-scaling tools before I treat them as directional predictors.
- I explicitly penalize turnover and max weight because that is where many pretty backtests fail in production.

## Likely Questions

### Why HMM on the median market series?

Because it gives a low-noise market proxy and avoids fitting a high-dimensional latent-state model directly on the full panel.

### Why a kinked quality score instead of a simple Sharpe objective?

Because the loss is asymmetric. Falling below acceptable Sharpe or exceeding acceptable turnover should be punished much more than marginal improvement above target should be rewarded.

### What are the main weaknesses of this stack?

- Grid-search overfitting risk
- Regime instability
- Need for stricter point-in-time handling in fundamental features
- Further improvement possible in transaction-cost and borrow modeling

## Files to Mention in an Interview

- `alpha_ma_crossover.py`: signal construction and volatility scaling
- `alpha_hmm.py`: robust market series, HMM fit, regime modulation
- `alpha_fund_momentum.py`: blended price and fundamental alpha
- `risk_overlays.py`: market-vol, beta-neutral, drawdown scaling
- `kalman_overlays.py`: Kalman-filtered regime exposure
- `grid_search_pipeline.py`: modular parameter search and scoring
- `quality_kinked.py`: asymmetric objective design
