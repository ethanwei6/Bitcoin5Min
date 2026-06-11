# Modeling Stack

The first paper-trading version used simple explainable signals. The current
ensemble adds online approximations of well-known volatility, jump, and trend
models that can update every few seconds without heavyweight numerical
dependencies.

These are online, dependency-light approximations of research-backed model
families. They are not claimed to be state-of-the-art trained predictors. A
model earns live ensemble weight only through walk-forward tests on real crypto
prices and through paper-trading evidence.

## Models

- `distance_to_start_random_walk`: baseline probability from current distance to
  the interval start price.
- `short_momentum`: short-horizon drift extrapolation.
- `mean_reversion`: rolling z-score reversion toward recent local mean.
- `volatility_fade`: fades the latest fast move using realized volatility.
- `ewma_riskmetrics_volatility`: exponentially weighted volatility forecast.
- `garch_1_1`: Bollerslev-style conditional variance forecast.
- `gjr_threshold_garch`: asymmetric threshold GARCH-style volatility forecast.
- `student_t_garch`: fat-tailed GARCH probability model for crypto jump risk.
- `regime_switching_volatility`: two-regime volatility mixture for calm versus
  stressed intraday conditions.
- `har_realized_volatility`: Corsi-style heterogeneous realized volatility using
  short, medium, and longer intraday components.
- `empirical_interval_knn`: analog model using completed five-minute intervals
  with similar partial returns and volatility.
- `merton_jump_diffusion`: separates large standardized returns as jumps and
  forecasts a matched-moment jump-diffusion distribution.
- `kalman_local_trend`: local log-price trend estimate with state uncertainty.
- `polymarket_orderbook_imbalance`: market microstructure signal from Up/Down
  book depth. Treat this as market-awareness, not independent price discovery.

## Research grounding

The volatility models are grounded in established financial econometrics:

- ARCH/GARCH-style conditional volatility traces back to Engle's ARCH work and
  Bollerslev's GARCH generalization.
- GJR/threshold-style terms are used to capture asymmetric volatility response.
- HAR realized volatility follows the heterogeneous autoregressive realized
  volatility idea associated with Corsi-style realized-volatility forecasting.
- Jump-diffusion follows the Merton-style idea that returns can combine
  continuous diffusion and discontinuous jumps.

Deep sequence models such as DeepLOB, Temporal Fusion Transformer, N-BEATS,
PatchTST, TiDE, and TimeMixer are relevant future research directions. They are
not included in the live trader because this repo does not yet have a large
walk-forward training set of BTC 5m terminal labels plus crypto order-book
features. Adding them without that dataset would make the public repo look more
impressive but would not make the trader more honest.

## Why online approximations

The bot is built to paper trade continuously. Full MLE GARCH, stochastic
volatility, neural temporal fusion, or transformer models can be added later as
offline-trained adapters, but they should not be dropped into live execution
without:

- walk-forward training and validation,
- calibration checks for probability forecasts,
- latency measurement,
- overfit controls by market regime and time of day,
- settlement-source basis checks against the official Polymarket resolution
  source.

The current implementation is a stronger research baseline, not proof of alpha.
The right promotion path is to let it collect a few weeks of logs, then score
every model individually before moving beyond paper.

## Ensemble calibration

The ensemble no longer treats every model as an equally reliable independent
vote. It now combines probabilities through:

- volatility-core weighted probability pooling, so the best-calibrated
  underlying-price models drive the estimate;
- median and trimmed-mean blending, so one extreme model cannot dominate the
  probability estimate;
- live orderbook-implied Up probability as a market prior;
- disagreement shrinkage toward 50/50 when the model stack is dispersed;
- model-market gap shrinkage, because a model that claims 80-90% win
  probability against a much lower live market needs stronger empirical
  calibration before receiving Kelly-sized capital.
- source-basis uncertainty from the live spot-source spread, because
  Polymarket settles against the official Chainlink stream and a small
  exchange-median edge can be noise when Coinbase/Kraken/Gemini/Binance quotes
  are tens of dollars apart;
- horizon confidence shrinkage toward the live market prior, not toward a
  generic 50/50 coin flip. This prevents early uncertainty from manufacturing
  fake value in cheap contracts that the market is already pricing as unlikely.

The weights are deliberately modest. They express current research judgment
about correlated short-horizon models and should be re-estimated from a larger
walk-forward paper ledger before any real-money promotion.

## Research Principle

The objective is not to make the trader look good by filtering out bad model
outputs. The objective is to build probabilities accurate enough that repeated
Kelly-sized paper trades have positive expectancy after fees and realistic
execution latency. When the paper trader loses money, the first response should
be to improve model specification, calibration, regime awareness, and model
weights. Hard caps belong to production risk management, not to the default
research loop.

## Underlying-price backtests

Use `scripts/backtest_models.py` when the research question is model quality,
not bot execution quality. It fetches real exchange candles for the current
Polymarket 5M crypto universe:

- BTC, ETH, SOL, BNB, XRP, and DOGE from Binance spot klines.
- HYPE from Hyperliquid candles.

The default `--interval 1m` is useful for cross-asset scans. BTC-specific
calibration can also use `--interval 1s`, which is slower but gives better
coverage of first-30-second and market-age effects. Do not treat tiny `1s`
smoke runs as accuracy evidence; they only verify the data path.

The backtester now reports chronological train/test confidence calibration. It
learns reliability buckets on the train window by model, market age, and
reported confidence, then scores raw and calibrated probabilities on the
held-out test window. This is the correct evidence source for answering "when a
model says 70%, how often does it actually win?"

Recent real BTC 7-day, 1-minute train/test evidence showed volatility-family
models remained the most reliable raw predictors by Brier score, while simple
bucket calibration helped weaker or overconfident models more than it helped
already-calibrated volatility models. That means calibration tables should be
used as diagnostics and model-weight inputs until longer walk-forward evidence
justifies feeding them directly into Kelly sizing.

## References

- Engle, "Autoregressive Conditional Heteroskedasticity with Estimates of the
  Variance of United Kingdom Inflation", Econometrica, 1982.
- Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity",
  Journal of Econometrics, 1986. DOI:
  `10.1016/0304-4076(86)90063-1`.
- Glosten, Jagannathan, and Runkle, "On the Relation between the Expected Value
  and the Volatility of the Nominal Excess Return on Stocks", Journal of
  Finance, 1993.
- Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility",
  Journal of Financial Econometrics, 2009.
- Merton, "Option Pricing When Underlying Stock Returns Are Discontinuous",
  Journal of Financial Economics, 1976.
- Zhang, Zohren, and Roberts, "DeepLOB: Deep Convolutional Neural Networks for
  Limit Order Books", arXiv: `1808.03668`.
- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
  Time Series Forecasting", arXiv: `1912.09363`.
- Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable
  time series forecasting", arXiv: `1905.10437`.
- Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers", arXiv: `2211.14730`.
