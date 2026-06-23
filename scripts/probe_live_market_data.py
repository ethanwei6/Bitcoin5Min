#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poly_5m_bot.config import load_config
from poly_5m_bot.market import PolymarketDiscovery
from poly_5m_bot.models import market_implied_up_probability
from poly_5m_bot.orderbook import ClobOrderBookClient, OrderBook
from poly_5m_bot.spot import SpotPriceClient, SpotSnapshot


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def top_depth(book: OrderBook, side: str, levels: int = 3) -> float:
    rows = book.bids if side == "bid" else book.asks
    return sum(level.size for level in rows[:levels])


def book_digest(book: OrderBook | None) -> dict[str, Any] | None:
    if book is None:
        return None
    return {
        "hash": book.hash,
        "best_bid": asdict(book.best_bid) if book.best_bid is not None else None,
        "best_ask": asdict(book.best_ask) if book.best_ask is not None else None,
        "top3_bid_depth": top_depth(book, "bid"),
        "top3_ask_depth": top_depth(book, "ask"),
        "bid_levels": len(book.bids),
        "ask_levels": len(book.asks),
        "tick_size": book.tick_size,
        "min_order_size": book.min_order_size,
    }


def spot_digest(spot: SpotSnapshot | None) -> dict[str, Any] | None:
    if spot is None:
        return None
    return {
        "median_price": spot.median_price,
        "source_spread_usd": spot.source_spread_usd,
        "quotes": [asdict(quote) for quote in spot.quotes],
        "failed_sources": spot.failed_sources,
        "warnings": spot.warnings,
    }


def median_or_zero(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def change_count(values: list[Any]) -> int:
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def longest_same_run(values: list[Any]) -> int:
    longest = 0
    current = 0
    previous = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error")]
    ok_rows = [row for row in rows if not row.get("error")]
    loop_intervals = [
        right["wall_time"] - left["wall_time"]
        for left, right in zip(rows, rows[1:])
        if right.get("wall_time") is not None and left.get("wall_time") is not None
    ]
    up_hashes = [
        (((row.get("up_book") or {}).get("hash")))
        for row in ok_rows
        if row.get("up_book") is not None
    ]
    down_hashes = [
        (((row.get("down_book") or {}).get("hash")))
        for row in ok_rows
        if row.get("down_book") is not None
    ]
    up_asks = [
        ((((row.get("up_book") or {}).get("best_ask") or {}).get("price")))
        for row in ok_rows
        if row.get("up_book") is not None
    ]
    down_asks = [
        ((((row.get("down_book") or {}).get("best_ask") or {}).get("price")))
        for row in ok_rows
        if row.get("down_book") is not None
    ]
    spot_prices = [
        ((row.get("spot") or {}).get("median_price"))
        for row in ok_rows
        if row.get("spot") is not None
    ]
    latencies_by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in (row.get("latencies_ms") or {}).items():
            latencies_by_key.setdefault(key, []).append(float(value))
    return {
        "samples": len(rows),
        "successful_samples": len(ok_rows),
        "error_samples": len(errors),
        "markets_seen": sorted({str((row.get("market") or {}).get("slug")) for row in ok_rows if row.get("market")}),
        "median_loop_interval_seconds": median_or_zero(loop_intervals),
        "p95_loop_interval_seconds": percentile(loop_intervals, 95),
        "distinct_up_book_hashes": len(set(up_hashes)),
        "distinct_down_book_hashes": len(set(down_hashes)),
        "up_top_ask_changes": change_count(up_asks),
        "down_top_ask_changes": change_count(down_asks),
        "spot_price_changes": change_count(spot_prices),
        "longest_same_up_book_hash_run": longest_same_run(up_hashes),
        "longest_same_down_book_hash_run": longest_same_run(down_hashes),
        "latencies_ms": {
            key: {
                "median": median_or_zero(values),
                "p95": percentile(values, 95),
                "max": max(values) if values else 0.0,
            }
            for key, values in sorted(latencies_by_key.items())
        },
        "errors": [row.get("error") for row in errors],
    }


def write_markdown(summary: dict[str, Any], path: Path, raw_path: Path) -> None:
    lines = [
        "# Live Polymarket Data Probe",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"Raw samples: `{raw_path}`",
        "",
        "## Summary",
        "",
        f"- Samples: `{summary['samples']}` successful: `{summary['successful_samples']}` errors: `{summary['error_samples']}`",
        f"- Markets seen: `{', '.join(summary['markets_seen']) or 'none'}`",
        f"- Median loop interval: `{summary['median_loop_interval_seconds']:.3f}s`",
        f"- P95 loop interval: `{summary['p95_loop_interval_seconds']:.3f}s`",
        f"- Distinct UP book hashes: `{summary['distinct_up_book_hashes']}`",
        f"- Distinct DOWN book hashes: `{summary['distinct_down_book_hashes']}`",
        f"- UP top-ask changes: `{summary['up_top_ask_changes']}`",
        f"- DOWN top-ask changes: `{summary['down_top_ask_changes']}`",
        f"- Spot median price changes: `{summary['spot_price_changes']}`",
        "",
        "## Latency",
        "",
        "| Component | Median ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|",
    ]
    for key, stats in summary["latencies_ms"].items():
        lines.append(
            f"| `{key}` | {stats['median']:.1f} | {stats['p95']:.1f} | {stats['max']:.1f} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in summary["errors"]:
            lines.append(f"- `{error}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timed_call(label: str, func):
    started = time.perf_counter()
    value = func()
    return value, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe live Polymarket/spot data cadence")
    parser.add_argument("--config", default="config/paper_btc_5m.json")
    parser.add_argument("--output-dir", default="reports/live_data_probes")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    raw_path = output_dir / f"live_data_probe_{stamp}.jsonl"
    summary_json_path = output_dir / f"live_data_probe_{stamp}.json"
    summary_md_path = output_dir / f"live_data_probe_{stamp}.md"

    discovery = PolymarketDiscovery(config.polymarket, config.market_interval_seconds)
    books = ClobOrderBookClient(config.polymarket)
    spot_client = SpotPriceClient(config.spot_sources, config.max_spot_source_spread_usd)

    rows: list[dict[str, Any]] = []
    stop_at = time.time() + max(args.duration_seconds, 0.0)
    with raw_path.open("w", encoding="utf-8") as handle:
        while time.time() < stop_at:
            wall_time = time.time()
            row: dict[str, Any] = {
                "wall_time": wall_time,
                "written_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "latencies_ms": {},
            }
            started = time.perf_counter()
            try:
                market, row["latencies_ms"]["market_discovery"] = timed_call(
                    "market_discovery",
                    lambda: discovery.current_market(wall_time),
                )
                row["market"] = asdict(market) if market is not None else None
                if market is not None:
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        spot_future = executor.submit(
                            timed_call,
                            "spot_snapshot",
                            spot_client.snapshot,
                        )
                        up_future = executor.submit(
                            timed_call,
                            "up_book",
                            lambda: books.get_book(market.tokens.up),
                        )
                        down_future = executor.submit(
                            timed_call,
                            "down_book",
                            lambda: books.get_book(market.tokens.down),
                        )
                        spot, row["latencies_ms"]["spot_snapshot"] = spot_future.result()
                        up_book, row["latencies_ms"]["up_book"] = up_future.result()
                        down_book, row["latencies_ms"]["down_book"] = down_future.result()
                    row["spot"] = spot_digest(spot)
                    row["up_book"] = book_digest(up_book)
                    row["down_book"] = book_digest(down_book)
                    row["market_prior_p_up"] = market_implied_up_probability(up_book, down_book)
                else:
                    spot, row["latencies_ms"]["spot_snapshot"] = timed_call(
                        "spot_snapshot",
                        spot_client.snapshot,
                    )
                    row["spot"] = spot_digest(spot)
                row["latencies_ms"]["total_sample"] = (time.perf_counter() - started) * 1000.0
            except Exception as exc:
                row["error"] = repr(exc)
                row["latencies_ms"]["total_sample"] = (time.perf_counter() - started) * 1000.0
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            rows.append(row)
            elapsed = time.time() - wall_time
            time.sleep(max(0.0, args.interval_seconds - elapsed))

    summary = summarize(rows)
    summary["raw_path"] = str(raw_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, summary_md_path, raw_path)
    print(json.dumps({"raw": str(raw_path), "json": str(summary_json_path), "markdown": str(summary_md_path)}, indent=2))


if __name__ == "__main__":
    main()
