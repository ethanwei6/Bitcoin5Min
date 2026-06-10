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
- `scripts/backtest_models.py` for underlying-price model validation across
  current Polymarket 5M crypto assets.
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
volatility-core ensemble, not live trading.
