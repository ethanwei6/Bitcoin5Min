# Bitcoin5Min

Execution-aware paper-trading research harness for Polymarket's Bitcoin
five-minute Up/Down markets.

The project continuously discovers the active BTC 5m market, polls live
Polymarket CLOB books and exchange spot feeds, runs an online ensemble of
short-horizon price models, sizes eligible trades with Kelly, and
journals every signal, paper execution, fill, settlement, and postmortem.

This is a paper-trading and research system only. It does not sign orders, post
orders, or require wallet credentials.

## Why This Exists

Five-minute binary markets are a clean testbed for market microstructure and
short-horizon forecasting. The hard part is not writing a model that sometimes
looks right; it is building the machinery to answer:

- Did the model see the correct market and correct underlying price?
- Would the quoted Polymarket book still have filled after realistic latency?
- Did PnL come from a persistent edge or from a short lucky regime?
- Which controls stop a working strategy from turning into overtrading?

This repo is designed around those questions.

## Highlights

- Live market discovery for `btc-updown-5m-{start_epoch}` Polymarket events.
- Multi-source BTC spot aggregation from Coinbase, Binance US, Kraken, and
  Gemini with source-spread diagnostics.
- Online model ensemble with random-walk, momentum, mean-reversion, EWMA,
  GARCH(1,1), GJR threshold-GARCH, Student-t GARCH, regime-switching
  volatility, HAR realized volatility, empirical interval KNN, Merton
  jump-diffusion, Kalman-style local trend, and orderbook imbalance signals.
- Weighted-majority model gate plus edge, confidence, Kelly sizing,
  refreshed-book execution checks, and optional opt-in risk brakes.
- Paper CLOB execution simulator that rechecks the live order book, honors
  market-specific `itode` taker delay metadata, applies FOK/FAK fill logic, and
  records latency/slippage/rejection evidence.
- JSONL research ledger and scripts for official settlement reconciliation,
  trade audit, evidence reports, and strategy postmortems.
- macOS LaunchAgent and VPS/systemd operating notes for long-running shadow
  trading.

## Architecture

```text
spot APIs + Polymarket Gamma/CLOB
              |
              v
     market + orderbook snapshots
              |
              v
      online model ensemble
              |
              v
 majority / edge / Kelly risk engine
              |
              v
 refreshed-book paper execution simulation
              |
              v
 trades, settlements, audits, postmortems
```

Core package: `src/poly_5m_bot/`

- `market.py`: active BTC 5m market discovery.
- `spot.py`: exchange quote aggregation and quality checks.
- `models.py`: online forecasting ensemble.
- `risk.py`: weighted majority, edge, Kelly sizing, and optional risk brakes.
- `execution.py`: paper CLOB execution simulator.
- `paper.py`: durable cash/position ledger.
- `bot.py`: continuous runner and drain-mode shutdown behavior.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
make test
```

Run a clean paper-trading ledger:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/run_paper_bot.py \
  --config config/paper_btc_5m.json \
  --output-dir outputs/paper_trader_v2
```

Run a bounded one-week paper test:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/run_paper_bot.py \
  --config config/paper_btc_5m.json \
  --output-dir outputs/paper_trader_v2 \
  --duration-seconds 604800 \
  --drain-before-stop-seconds 600
```

The bot stops opening new positions during drain mode and waits for open
positions to resolve before exiting.

## Evidence Workflow

Generate a full evidence report:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/generate_evidence_report.py \
  --output-dir outputs/paper_trader_v2 \
  --report-dir outputs/paper_trader_v2/reports
```

Run reconciliation, audit, performance, and evidence maintenance:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/run_daily_maintenance.py \
  --output-dir outputs/paper_trader_v2 \
  --report-dir outputs/paper_trader_v2/reports
```

Generate a strategy postmortem:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/analyze_paper_run.py \
  --output-dir outputs/paper_trader_v2 \
  --report-dir outputs/paper_trader_v2/reports
```

Backtest the prediction models on actual underlying crypto prices:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/backtest_models.py --days 7 --reports-dir reports/model_backtests
```

This uses the current Polymarket 5M crypto universe and tests the models against
exchange spot candles, not against the bot's historical trade logs.

For BTC-specific confidence calibration, run a chronological train/test replay:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/backtest_models.py \
  --asset BTC \
  --days 30 \
  --interval 1m \
  --train-fraction 0.70 \
  --min-calibration-bucket-count 50
```

For finer market-age research, `--interval 1s` is supported for Binance symbols.
Second-level runs are slower and should be used for offline calibration studies,
not as a quick CI check. Tiny second-level smoke runs verify plumbing only; they
are not enough evidence to claim model accuracy.

Runtime ledgers are intentionally ignored by git. The reports are reproducible
from the JSONL outputs generated by a run.

## Execution Simulation

Paper fills do not blindly use the first quoted ask. When a trade passes the
risk engine, the simulator:

1. Builds a paper marketable BUY intent with side, token, shares, reference
   price, and worst accepted price.
2. Checks the current CLOB market metadata for Polymarket's `itode` taker delay
   flag.
3. Waits the documented delay only when the live market reports it.
4. Re-fetches the live CLOB book.
5. Walks refreshed asks under FOK/FAK rules.
6. Records latency, book hashes, consumed levels, fees, and rejection reason.

This mirrors the pre-submit execution path while remaining paper-only.

## Model-First Research Mode

The first long paper run peaked early and then failed because the strategy kept
trading after the realized PnL regime flipped. The next run was profitable but
showed a deeper modeling leak: the raw ensemble could become confidently wrong
when several correlated models agreed on the same stale short-horizon move. The
current default emphasizes probability calibration over hard filtering:

- volatility-core weighted probability pooling instead of a plain average
- robust median/trimmed-mean blending to reduce outlier model influence
- live orderbook-implied probability as a market prior
- disagreement-based shrinkage toward 50/50 when the model stack is unstable
- model-market gap shrinkage, so a large edge claim must survive calibration
  before Kelly sees it
- horizon confidence shrinkage, so early-interval forecasts are pulled closer
  to 50/50 than forecasts made near resolution

The default research config avoids arbitrary hard strategy filters. A value of
`0` disables optional brakes such as max trade size, per-market exposure, daily
loss, drawdown, consecutive-loss, and per-market entry caps. It now uses
quarter-Kelly sizing (`kelly_fraction: 0.25`) and `min_edge: 0.0`, so a trade is
funded only when the model probability beats the executable cost after fees.
Kelly also targets total exposure in the current market instead of re-sizing
from scratch on every tick, which keeps the run model-driven without repeatedly
stacking the same correlated bet.
The live decision is therefore driven by model probability, weighted majority,
available cash, current market exposure, top-of-book liquidity, and calibrated
Kelly sizing.

Operational guards remain in place for data integrity: the bot still requires
enough models, a captured interval start, positive Kelly spend after fees, and
enough time before resolution for the execution simulation to be meaningful.

The point is not to claim alpha from one run, from synthetic tests, or from a
single lucky PnL curve. The point is to make every failure measurable, then
improve the model stack, calibration, weights, and sizing until the
fractional-Kelly paper strategy has repeatable positive expectancy on held-out
real crypto prices and live paper-trading ledgers.

## Documentation

- [Modeling stack](docs/modeling_stack.md)
- [Operating logistics](docs/operating_logistics.md)
- [Research postmortem](docs/research_postmortem.md)

## Safety

This repository is not financial advice and is not a live trading bot. It is a
paper-trading research system for testing whether a short-horizon strategy has
repeatable edge after fees, latency, slippage, and market-resolution risk.
