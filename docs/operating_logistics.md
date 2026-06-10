# Operating Logistics

## What the bot is trading

Polymarket's BTC 5m markets are binary Up/Down contracts. The relevant market
question is whether BTC finishes the interval greater than or equal to the
start price. The market page and rules point to Chainlink BTC/USD as the
resolution source, so any exchange-based spot feed is only a proxy. That basis
risk is part of the paper test and should be measured explicitly.

## Data path

1. Generate the current market slug from the five-minute UTC start epoch:
   `btc-updown-5m-{start_epoch}`.
2. Fetch Gamma event metadata and extract the tradable market, outcomes, token
   IDs, start/end time, and CLOB fields.
3. Fetch CLOB order books for Up and Down token IDs.
4. Fetch BTC spot from Coinbase, Binance US, and Kraken, then use the median
   valid price as the primary spot proxy.
5. Persist every snapshot before creating a signal.
6. If the signal passes risk gates, create a paper CLOB BUY intent and re-fetch
   the current CLOB book before filling. This recheck measures actual request
   latency and catches book moves between signal time and order-intent time.

## Modeling path

The current ensemble uses explainable short-horizon models plus online
approximations of established econometric price and volatility models:

- Distance-to-start random walk: estimates the chance current BTC remains above
  the interval start price, volatility adjusted.
- Short momentum, local trend, mean reversion, and volatility fade: capture
  fast intraday drift and reversal effects.
- EWMA/RiskMetrics volatility, GARCH(1,1), and GJR threshold-GARCH: estimate
  conditional volatility for the next few minutes.
- HAR realized volatility: combines short, medium, and longer intraday
  realized-volatility components.
- Merton jump-diffusion: separates large standardized returns as jumps and
  forecasts a matched-moment jump-diffusion distribution.
- Polymarket orderbook imbalance: market microstructure signal from Up/Down
  book depth. Treat this as market-awareness, not independent price discovery.

The bot records individual model probabilities so the evidence report can score
calibration and direction accuracy per model rather than only reporting blended
PnL.

## Decision gate

A trade is eligible only when:

- Enough models produce probabilities.
- A strict weighted majority points to the same side, using the configured
  calibration-first model weights.
- Configured confidence and edge thresholds clear. In the default model-first
  research config both are set to `0`, so positive Kelly spend after fees is
  the effective edge test.
- Optional risk brakes have not tripped. These are disabled with `0` in the
  default research config, including per-market entry caps, max trade size,
  per-market exposure, daily loss, drawdown, and consecutive-loss limits.
- The market is not too close to start or resolution.
- Available cash and top-of-book liquidity can support the Kelly-sized order.

## Kelly sizing

For a binary contract with win probability `p` and executable price `c`, the
full Kelly fraction of bankroll to spend is:

```text
f = (p - c) / (1 - c)
```

The default research config uses quarter Kelly (`kelly_fraction: 0.25`) and
targets total current-market exposure before adding to a position. This means a
fresh signal can still receive meaningful size, but repeated ticks in the same
market do not keep re-spending as though no position already exists. Optional
`max_trade_usd` and `max_position_usd_per_market` limits can still be turned
back on for production-style risk testing by setting them above `0`.

## Execution simulation

Polling the book and immediately marking a paper fill is too optimistic for a
real strategy. The paper broker now runs a stricter CLOB execution simulation:

1. Signal and risk checks produce a proposed side, price, shares, and spend.
2. The bot writes the signal.
3. It builds a paper BUY intent matching Polymarket's market-order shape: a
   marketable limit order with `FOK` or `FAK` behavior and a worst-price limit.
4. It checks `GET /clob-markets/{condition_id}` for Polymarket's `itode` taker
   delay flag and applies the documented crypto/finance hold only when the
   current market reports it.
5. It re-fetches the public CLOB book for the selected outcome token.
6. It measures elapsed latency from intent creation through refreshed-book
   receipt.
7. It walks refreshed asks up to `execution_max_slippage_ticks`.
8. It fills, partially fills, or rejects the paper order according to
   `execution_order_type` and `execution_min_fill_ratio`.
9. Accepted fills must still pass the post-execution risk check before the
   paper broker records a position.

This does not sign or post any order. It mirrors the real pre-submit path while
remaining paper-only. If a real adapter is added later, it should use the same
intent fields plus EIP-712 signing and `postOrder`/`createAndPostMarketOrder`.

To probe the live execution path without opening a model-triggered paper
position, run:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=work/pycache \
python3 scripts/run_execution_dry_run.py --config config/paper_btc_5m.json --side AUTO --spend-usd 5
```

The dry run writes `dry_run: true` to `execution_simulations.jsonl`, including
book hashes, top-ask drift, latency, filled shares, consumed levels, and
fill/rejection reason. If the current market reports `itode: true`, the
reported latency includes the documented 250 ms taker hold.

## Hosting

Minimum setup:

- 1 vCPU, 1 GB RAM VPS
- Python 3.10+
- Persistent disk for JSONL data
- `systemd` service with automatic restart
- Daily archive/sync of `outputs/paper_trader`

For a paper-only run, there are no wallet keys. The service only needs outbound
HTTPS to Polymarket and spot-price APIs. Keep the process boring: one service,
one persistent output directory, restart on crash, and daily log rotation or
object-storage sync.

Concrete setup:

1. Clone or copy this folder to `/srv/poly-5m-paper-trader`.
2. Create a virtualenv and install the package.
3. Start with `--once` to verify API access.
4. Install the `systemd` service below.
5. Check `signals.jsonl` and `official_settlements.jsonl` daily.
6. After 2-4 weeks, evaluate model-level calibration and PnL, then change model
   weights or model definitions before considering production-style risk caps.

Suggested `systemd` unit:

```ini
[Unit]
Description=Polymarket BTC 5m paper trader
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/srv/poly-5m-paper-trader
ExecStart=/srv/poly-5m-paper-trader/.venv/bin/poly-5m-paper --config config/paper_btc_5m.json
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

## Local macOS background run

For this workspace, LaunchAgent files are generated by
`scripts/install_local_launch_agents.sh`. The installer syncs the current code
into `~/Library/Application Support/poly5m-paper-trader` because macOS
LaunchAgents may not be allowed to execute directly from `~/Documents`.

Install and start the paper bot plus daily evidence maintenance:

```bash
scripts/install_local_launch_agents.sh
```

Check status:

```bash
launchctl list | grep com.ethan.poly5m
```

Useful logs:

```bash
tail -f "$HOME/Library/Application Support/poly5m-paper-trader/outputs/paper_trader/launchd.out.log"
tail -f "$HOME/Library/Application Support/poly5m-paper-trader/outputs/paper_trader/launchd.err.log"
tail -f "$HOME/Library/Application Support/poly5m-paper-trader/outputs/paper_trader/maintenance.err.log"
```

Uninstall:

```bash
scripts/uninstall_local_launch_agents.sh
```

Daily reports are written to
`~/Library/Application Support/poly5m-paper-trader/outputs/paper_trader/reports/`.

## Shutdown behavior

Stopping should never strand the paper ledger mid-market. The runner has a
drain phase:

- It stops opening new positions before the planned stop time.
- It continues polling spot and Gamma.
- It writes only snapshots and official settlements.
- It exits only when there are no open positions.

For a bounded run, use `--duration-seconds` plus either the configured
`drain_before_stop_seconds` or a CLI override:

```bash
python scripts/run_paper_bot.py --config config/paper_btc_5m.json --duration-seconds 3600 --drain-before-stop-seconds 420
```

For manual stop signals, the first `SIGINT`/`SIGTERM` enters the same drain mode
instead of immediately exiting.

## Promotion criteria

Do not promote to live trading until the paper ledger demonstrates:

- Positive net expected value after taker fees and spread.
- Stable performance across time of day and volatility regimes.
- No dependence on stale/missing market discovery.
- Conservative drawdown under simulated worse fills.
- Agreement between paper settlement and actual Polymarket resolution.
- Zero entry-audit failures for captured top-of-book fills.
- Low rejected-intent rate after refreshed-book execution simulation.
- Calibration that is good enough to justify Kelly sizing, not just lucky PnL.
- Positive PnL under 1c, 2c, and 3c adverse-fill stress tests.

## Current risk lessons

The first 263-trade paper ledger peaked early and then failed because the bot
kept trading through a dead realized-PnL regime. The next longer ledger was
positive, but it showed that the real leak was model calibration: correlated
short-horizon models could agree with each other while still being stale versus
the live market. The current model stack therefore uses volatility-core weighted
probability pooling, robust median/trimmed blending, orderbook-implied prior
anchoring, disagreement shrinkage, and model-market gap shrinkage before Kelly
sizing sees a probability.

The current research config disables the old hard brakes by setting them to
`0`, sets `max_contract_entry_price` to `1.0`, uses quarter Kelly, and sets
`min_edge` to `0.0`. The sizing layer now treats Kelly as a target exposure for
the active market, so repeated entries are allowed only when the current market
position is still below the model-implied fractional-Kelly target. These are
not proof of alpha; they are a cleaner way to test whether the model estimates
themselves are good enough before adding production-style risk brakes back in.
