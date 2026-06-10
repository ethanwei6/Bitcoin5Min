# Modeling Stack

The first paper-trading version used simple explainable signals. The current
ensemble adds online approximations of well-known volatility, jump, and trend
models that can update every few seconds without heavyweight numerical
dependencies.

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

## Why online approximations

The bot is built to paper trade continuously. Full MLE GARCH, stochastic
volatility, neural temporal fusion, or transformer models can be added later as
offline-trained adapters, but they should not be dropped into live execution
without:

- walk-forward training and validation,
- calibration checks for probability forecasts,
- latency measurement,
- overfit controls by market regime and time of day,
- settlement-source basis checks against Chainlink BTC/USD.

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
- disagreement shrinkage toward 50/50 when the model stack is dispersed.

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
not bot execution quality. It fetches one-minute exchange candles for the
current Polymarket 5M crypto universe:

- BTC, ETH, SOL, BNB, XRP, and DOGE from Binance spot klines.
- HYPE from Hyperliquid candles.

The 7-day cross-asset run on June 9, 2026 ranked the volatility-family models
highest by Brier score: GJR threshold GARCH, GARCH(1,1), HAR realized
volatility, and EWMA. A follow-up smoke run showed Student-t GARCH and
regime-switching volatility in the same top calibration band, while empirical
interval KNN was useful but weaker. The ensemble is therefore volatility-core
weighted. Mean reversion, short momentum, KNN, and volatility fade remain
available as auxiliary views, but they are not the center of the probability
estimate.
