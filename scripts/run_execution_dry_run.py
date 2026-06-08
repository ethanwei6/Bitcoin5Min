#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from poly_5m_bot.config import load_config
from poly_5m_bot.execution import PaperExecutionSimulator
from poly_5m_bot.journal import JsonlJournal
from poly_5m_bot.market import PolymarketDiscovery
from poly_5m_bot.orderbook import ClobOrderBookClient, OrderBook
from poly_5m_bot.risk import TradeDecision, taker_fee_per_share


def choose_side(up_book: OrderBook, down_book: OrderBook, requested: str) -> str:
    if requested != "AUTO":
        return requested
    up_ask = up_book.best_ask.price if up_book.best_ask else 1.0
    down_ask = down_book.best_ask.price if down_book.best_ask else 1.0
    return "UP" if up_ask <= down_ask else "DOWN"


def dry_run_decision(side: str, book: OrderBook, spend_usd: float, fee_rate: float) -> TradeDecision:
    if book.best_ask is None:
        raise RuntimeError(f"No executable ask for {side}")
    price = book.best_ask.price
    fee_per_share = taker_fee_per_share(price, fee_rate)
    effective_cost = price + fee_per_share
    shares = min(spend_usd / effective_cost, book.best_ask.size)
    return TradeDecision(
        should_trade=True,
        side=side,
        probability=0.0,
        executable_price=price,
        effective_cost=effective_cost,
        edge=0.0,
        shares=shares,
        spend_usd=shares * effective_cost,
        kelly_fraction_full=0.0,
        reason="manual execution dry-run",
    )


def market_taker_delay(books: ClobOrderBookClient, condition_id: str) -> tuple[float, str]:
    try:
        market_info = books.get_market_info(condition_id)
    except Exception:
        return 0.0, "clob_market_info_unavailable"
    return (250.0, "clob_market_info_itode") if bool(market_info.get("itode")) else (0.0, "clob_market_info_itode")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-only live CLOB execution dry run")
    parser.add_argument("--config", default="config/paper_btc_5m.json")
    parser.add_argument("--side", choices=["AUTO", "UP", "DOWN"], default="AUTO")
    parser.add_argument("--spend-usd", type=float, default=5.0)
    args = parser.parse_args()

    config = load_config(args.config)
    discovery = PolymarketDiscovery(config.polymarket, config.market_interval_seconds)
    books = ClobOrderBookClient(config.polymarket)
    market = discovery.current_market(time.time())
    if market is None:
        raise RuntimeError("No current BTC five-minute Polymarket market found")

    up_book = books.get_book(market.tokens.up)
    down_book = books.get_book(market.tokens.down)
    side = choose_side(up_book, down_book, args.side)
    book = up_book if side == "UP" else down_book
    token_id = market.tokens.up if side == "UP" else market.tokens.down
    decision = dry_run_decision(side, book, args.spend_usd, config.fee_rate)
    simulator = PaperExecutionSimulator(
        enabled=config.simulate_execution_latency,
        order_type=config.execution_order_type,
        max_slippage_ticks=config.execution_max_slippage_ticks,
        min_fill_ratio=config.execution_min_fill_ratio,
        fee_rate=config.fee_rate,
    )
    delay_ms, delay_source = market_taker_delay(books, market.condition_id)
    execution = simulator.simulate_buy(
        decision=decision,
        token_id=token_id,
        book_at_signal=book,
        fetch_current_book=books.get_book,
        market_taker_delay_ms=delay_ms,
        market_taker_delay_source=delay_source,
    )
    record = {
        "timestamp": time.time(),
        "dry_run": True,
        "market_slug": market.slug,
        "market": asdict(market),
        "decision": decision,
        "execution": execution,
    }
    JsonlJournal(Path(config.output_dir)).append("execution_simulations.jsonl", record)
    print(json.dumps(record, default=lambda value: asdict(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
