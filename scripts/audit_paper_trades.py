#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import argparse
from pathlib import Path


OUTPUT_DIR = Path("outputs/paper_trader")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def best_ask(snapshot: dict, side: str) -> tuple[float | None, float | None]:
    book_key = "up_book" if side == "UP" else "down_book"
    book = snapshot.get(book_key) or {}
    asks = book.get("asks") or []
    if not asks:
        return None, None
    return float(asks[0]["price"]), float(asks[0]["size"])


def nearest_snapshot(snapshots: list[dict], market_slug: str, timestamp: float) -> dict | None:
    candidates = [
        snap
        for snap in snapshots
        if (snap.get("market") or {}).get("slug") == market_slug
        and float(snap.get("timestamp", 0)) <= timestamp
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda snap: float(snap.get("timestamp", 0)))


def first_spot_for_market(snapshots: list[dict], market_slug: str) -> float | None:
    for snap in snapshots:
        if (snap.get("market") or {}).get("slug") == market_slug:
            spot = snap.get("spot") or {}
            if spot.get("median_price") is not None:
                return float(spot["median_price"])
    return None


def first_spot_after(snapshots: list[dict], end_epoch: int) -> float | None:
    for snap in snapshots:
        if float(snap.get("timestamp", 0)) >= end_epoch:
            spot = snap.get("spot") or {}
            if spot.get("median_price") is not None:
                return float(spot["median_price"])
    return None


def consumed_level_capacity(levels: list[dict]) -> float:
    return sum(float(level.get("shares", 0.0)) for level in levels)


def execution_entry_check(position: dict, decision: dict, execution: dict) -> dict:
    consumed = execution.get("consumed_levels") or []
    fill_price = execution.get("fill_price")
    fill_shares = execution.get("fill_shares")
    return {
        "market_slug": position["market_slug"],
        "side": position["side"],
        "trade_price": float(position["price"]),
        "decision_price": float(decision["executable_price"]),
        "execution_fill_price": fill_price,
        "execution_top_ask_at_signal": execution.get("top_ask_at_signal"),
        "execution_top_ask_at_submit": execution.get("top_ask_at_submit"),
        "execution_latency_ms": execution.get("latency_ms"),
        "execution_book_hash_at_signal": (execution.get("intent") or {}).get("book_hash_at_signal"),
        "execution_book_hash_at_submit": execution.get("book_hash_at_submit"),
        "shares": float(position["shares"]),
        "execution_fill_shares": fill_shares,
        "execution_accepted": bool(execution.get("accepted")),
        "price_matches_decision": abs(float(position["price"]) - float(decision["executable_price"])) < 1e-9,
        "price_matches_execution_fill": fill_price is not None
        and abs(float(position["price"]) - float(fill_price)) < 1e-9,
        "shares_match_execution_fill": fill_shares is not None
        and abs(float(position["shares"]) - float(fill_shares)) < 1e-9,
        "fits_top_ask_size": fill_shares is not None
        and consumed_level_capacity(consumed) + 1e-9 >= float(fill_shares),
        "entry_evidence_source": "execution_simulation",
    }


def spot_quality(snapshots: list[dict]) -> dict:
    spreads = []
    bad_medians = 0
    source_counts = []
    failures = 0
    warning_count = 0
    for snap in snapshots:
        spot = snap.get("spot") or {}
        quotes = spot.get("quotes") or []
        prices = [float(item["price"]) for item in quotes if item.get("price") is not None]
        if prices:
            source_counts.append(len(prices))
            spreads.append(max(prices) - min(prices))
            if abs(float(spot["median_price"]) - statistics.median(prices)) > 1e-9:
                bad_medians += 1
        failures += len(spot.get("failed_sources") or [])
        warning_count += len(spot.get("warnings") or [])
    return {
        "snapshots_with_spot": len(source_counts),
        "median_source_count": statistics.median(source_counts) if source_counts else 0,
        "max_cross_source_spread_usd": max(spreads) if spreads else 0,
        "median_cross_source_spread_usd": statistics.median(spreads) if spreads else 0,
        "bad_median_count": bad_medians,
        "failed_source_count": failures,
        "warning_count": warning_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paper-trading ledger consistency")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    snapshots = load_jsonl(output_dir / "snapshots.jsonl")
    trades = load_jsonl(output_dir / "trades.jsonl")
    official_path = output_dir / "official_settlements.jsonl"
    settlement_path = official_path if official_path.exists() else output_dir / "settlements.jsonl"
    settlements = load_jsonl(settlement_path)

    entry_checks = []
    for trade in trades:
        position = trade["position"]
        decision = trade["decision"]
        execution = trade.get("execution")
        if execution:
            entry_checks.append(execution_entry_check(position, decision, execution))
            continue
        opened_at = float(position["opened_at"])
        snapshot = nearest_snapshot(snapshots, position["market_slug"], opened_at)
        ask_price, ask_size = best_ask(snapshot or {}, position["side"])
        entry_checks.append(
            {
                "market_slug": position["market_slug"],
                "side": position["side"],
                "trade_price": float(position["price"]),
                "decision_price": float(decision["executable_price"]),
                "snapshot_best_ask": ask_price,
                "snapshot_best_ask_size": ask_size,
                "shares": float(position["shares"]),
                "price_matches_decision": abs(float(position["price"]) - float(decision["executable_price"])) < 1e-9,
                "price_matches_snapshot_top_ask": ask_price is not None and abs(float(position["price"]) - ask_price) < 1e-9,
                "fits_top_ask_size": ask_size is not None and float(position["shares"]) <= ask_size + 1e-9,
                "entry_evidence_source": "signal_snapshot",
            }
        )

    settlement_checks = []
    by_market = {}
    for trade in trades:
        position = trade["position"]
        by_market[position["market_slug"]] = int(position["market_end_epoch"])
    for settlement in settlements:
        market_slug = settlement["market_slug"]
        expected_start = first_spot_for_market(snapshots, market_slug)
        expected_end = first_spot_after(snapshots, by_market[market_slug])
        start = settlement.get("start_price_proxy")
        end = settlement.get("end_price_proxy")
        if start is not None and end is not None:
            start = float(start)
            end = float(end)
            expected_side = "UP" if end >= start else "DOWN"
        else:
            expected_side = settlement["winning_side"]
        settlement_checks.append(
            {
                "market_slug": market_slug,
                "side": settlement["side"],
                "winning_side": settlement["winning_side"],
                "start_price_proxy": start,
                "first_snapshot_start_proxy": expected_start,
                "end_price_proxy": end,
                "first_snapshot_after_end_proxy": expected_end,
                "winning_side_matches_proxy": settlement["winning_side"] == expected_side,
                "start_matches_snapshot": start is None or (expected_start is not None and abs(start - expected_start) < 1e-9),
                "end_matches_snapshot": end is None or (expected_end is not None and abs(end - expected_end) < 1e-9),
            }
        )

    report = {
        "counts": {
            "snapshots": len(snapshots),
            "trades": len(trades),
            "settlements": len(settlements),
            "unsettled_trades": len(trades) - len(settlements),
            "settlement_file": settlement_path.name,
        },
        "entry_checks": entry_checks,
        "settlement_checks": settlement_checks,
        "spot_quality": spot_quality(snapshots),
        "all_entry_prices_match_top_ask": all(
            item.get("price_matches_snapshot_top_ask", True) for item in entry_checks
        ),
        "all_entry_prices_match_execution_or_top_ask": all(
            item.get("price_matches_execution_fill", item.get("price_matches_snapshot_top_ask", False))
            for item in entry_checks
        ),
        "all_entry_shares_match_execution": all(
            item.get("shares_match_execution_fill", True) for item in entry_checks
        ),
        "all_entries_fit_top_ask_size": all(item["fits_top_ask_size"] for item in entry_checks),
        "all_settlements_match_proxy_snapshots": all(
            item["winning_side_matches_proxy"] and item["start_matches_snapshot"] and item["end_matches_snapshot"]
            for item in settlement_checks
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
