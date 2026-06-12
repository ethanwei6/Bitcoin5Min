# Research Postmortem

This project treats losses as research data. The first long paper run was
valuable because it exposed where the naive version of the strategy stopped
being a trader and became an overtrading machine.

## First Long Run

Ledger: 263 settled paper trades across BTC five-minute Polymarket markets.

| Metric | Value |
|---|---:|
| Final realized PnL | -$363.70 |
| Peak PnL | +$288.80 |
| Peak trade | 44 |
| Max drawdown from peak | -$669.73 |
| Settled-trade win rate | 49.4% |

The early run looked attractive, but the later run exposed a regime failure:
the ensemble continued to generate high-confidence signals after realized PnL
and model calibration had deteriorated.

## What Failed

The core failure was not simply variance. Three issues compounded:

1. The bot kept trading after the realized PnL regime flipped.
2. The official-settlement daily loss check was brittle because reconciled
   settlement rows did not always include `written_at`.
3. The strategy repeatedly entered the same five-minute market and sometimes
   bought expensive binary contracts where one missed forecast erased multiple
   small wins.

Model-level diagnostics were also sobering: traded-signal model direction
accuracy clustered around roughly 49-53%, which is not enough to justify taker
fees, latency, and 65c+ entries.

## Counterfactual Risk Controls

The strongest simple counterfactual was a peak-to-trough drawdown stop.

| Rule | Kept Trades | Counterfactual PnL |
|---|---:|---:|
| $50 drawdown stop | 9 | +$42.41 |
| $75 drawdown stop | 34 | +$66.79 |
| $100 drawdown stop | 76 | +$183.51 |
| $125 drawdown stop | 82 | +$162.91 |
| $150 drawdown stop | 89 | +$114.22 |
| $300 drawdown stop | 103 | -$21.46 |

This does not prove the strategy has alpha. It shows the original risk system
was too permissive and that realized-regime controls should be part of the
research loop.

## Changes Made

The current implementation now includes:

- Daily PnL and drawdown calculation from paired trades and settlements.
- Optional drawdown, consecutive-loss, per-market entry, max-trade, and
  per-market exposure brakes. The default research config disables these with
  `0` so model quality is tested directly before production-style brakes are
  added back.
- `max_contract_entry_price` can be used as a price cap, but the research
  config sets it to `1.0` to avoid an arbitrary price cutoff.
- Calibrated ensemble aggregation with volatility-core weighted probability pooling.
- Market-prior anchoring from the live Up/Down order books.
- Disagreement shrinkage when model probabilities are dispersed.
- Horizon-confidence shrinkage so early-interval signals are less confident
  than otherwise identical near-resolution signals.
- `scripts/backtest_models.py` for underlying-price model validation across
  current Polymarket 5M crypto assets.
- Chronological train/test reliability buckets from real crypto candles, so
  reported model confidence can be compared with held-out win rates by market
  age.
- Post-execution risk check before a simulated fill is recorded as a paper trade.
- Clean output-directory override for separate strategy iterations.
- `scripts/analyze_paper_run.py` for repeatable postmortem generation.

## Research Interpretation

The alpha, if it exists, is not "the ensemble is always smarter than the
market." The more realistic hypothesis is narrower:

- There may be short time windows where Polymarket's five-minute crypto markets
  lag realized spot movement.
- The strategy needs to avoid paying up for already-consensus contracts.
- Model weights and calibration should adapt when realized probabilities turn
  out to be stale or regime-dependent.
- Evaluation should focus on net PnL after taker fees, marketable-order delay,
  fill slippage, and official-resolution basis.

The next research step is another clean paper ledger using the underlying-tested
volatility-core ensemble plus longer real-price confidence calibration, not live
trading.

## June 11 Model Regression

The later two-hour paper run was a useful failure case:

| Metric | Value |
|---|---:|
| Settled trades | 40 |
| Realized PnL | -$289.75 |
| Win rate | 20.0% |
| Max drawdown | -$309.31 |

The headline comparison to the prior +$2.8k one-hour run was misleading. The
profitable run used far more effective turnover and repeated entries in the same
markets, including a single market drawdown around -$675. The later loss was
smaller in exposure, but it exposed the model bug more clearly: the ensemble
kept treating cheap UP contracts as recovery value when the live market and the
official-resolution basis disagreed with the exchange-median signal.

The key model fix was not a hard entry cutoff. The ensemble now:

- allows zero-weight models to contribute nothing to probability pooling;
- heavily downweights KNN, orderbook imbalance, Merton jump diffusion, Kalman,
  mean-reversion, and volatility-fade signals until they show out-of-sample
  reliability;
- shrinks horizon uncertainty toward the market prior instead of 50/50;
- adds source-spread basis variance to the volatility models;
- increases market-prior anchoring when source-basis noise is large relative to
  distance from the interval start.

On the saved bad-run snapshot replay, these changes reduced the comparable
replay loss from roughly -$304 to about -$27. That is not proof of alpha; it is
evidence that the worst regression was calibration and oracle-basis handling,
not simply unlucky variance.

## June 11 Follow-Up Run

A later two-hour paper run with the calibrated stack finished positive, but the
path was still too concentrated to treat as validated edge:

| Metric | Value |
|---|---:|
| Settled trades | 32 |
| Realized PnL | +$100.69 |
| Win rate | 37.5% |
| Max drawdown | -$45.83 |

The largest trade made +$121.07 after buying a late DOWN contract at 16c. The
trade was not a high-confidence directional forecast; it was a cheap-underdog
dislocation where the model assigned roughly 28% probability to a contract the
book briefly offered near 16c. Excluding that one trade, the run would have been
negative.

This is an important distinction for future research. The current bot can be
profitable in a value-betting sense while still losing more trades than it wins.
For a >50% win-rate strategy, directional trades must be evaluated separately
from underdog dislocation trades. The next reports should break out those
cohorts explicitly instead of judging the entire ledger as one homogeneous
strategy.

The follow-up implementation adds that separation directly to the risk engine
and reports. Future trades carry `raw_probability`, haircut-adjusted
`probability`, `kelly_scale`, and `trade_cohort` fields. Underdog value trades
receive small adverse-selection haircuts and smaller Kelly sizing instead of
being hard-filtered out. This preserves the core positive-expectancy objective
while reducing dependence on one large late-underdog hit.

## Three-hour calibrated run

The next three-hour run finished positive, but it exposed two cleaner research
issues than the headline PnL:

- Realized PnL was +$26.66 on 9 settled trades, with a 33.3% trade win rate.
- One market produced four same-side DOWN entries and a -$9.02 market loss,
  which showed that repeated same-market signals were still too correlated to
  size as fresh independent evidence.
- Correcting calibration to score all proxy-resolved observed markets, not only
  traded markets, improved the ensemble Brier read to 0.1742 across 2,064
  signal ticks and showed the volatility/HAR/GARCH core remained stronger than
  the traded-only subset had suggested.

The implementation response was model-first: fix the volatility-fade model's
time scaling, keep the volatility-core weights dominant, and add market-level
calibration to the evidence report. The trader response was a soft same-market
reentry haircut and Kelly decay rather than a hard entry cap.
