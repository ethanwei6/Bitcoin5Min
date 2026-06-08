from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .bot import PaperTradingBot, run_once
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket BTC 5m paper trader")
    parser.add_argument("--config", default="config/paper_btc_5m.json")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir for a clean run")
    parser.add_argument("--once", action="store_true", help="Run one sample and exit")
    parser.add_argument("--settle-only", action="store_true", help="Settle due positions without opening new trades")
    parser.add_argument("--duration-seconds", type=float, default=None, help="Run for a bounded number of seconds")
    parser.add_argument(
        "--drain-before-stop-seconds",
        type=float,
        default=None,
        help="For bounded runs, stop opening new trades this many seconds before stop and wait for open positions to resolve",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))
    if args.once:
        run_once(config, settle_only=args.settle_only)
    else:
        PaperTradingBot(config).run_forever(
            duration_seconds=args.duration_seconds,
            settle_only=args.settle_only,
            drain_before_stop_seconds=args.drain_before_stop_seconds,
        )


if __name__ == "__main__":
    main()
