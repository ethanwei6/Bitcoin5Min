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
- `har_realized_volatility`: Corsi-style heterogeneous realized volatility using
  short, medium, and longer intraday components.
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
every model individually before assigning weights or moving beyond paper.

