#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def row_time(row: dict[str, Any]) -> datetime | None:
    value = row.get("written_at")
    if not value:
        return None
    return parse_time(str(value))


def in_window(row: dict[str, Any], since: datetime | None, until: datetime | None) -> bool:
    timestamp = row_time(row)
    if timestamp is None:
        return True
    if since is not None and timestamp < since:
        return False
    if until is not None and timestamp >= until:
        return False
    return True


def iso_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


def market_window(position: dict[str, Any]) -> str:
    start = datetime.fromtimestamp(float(position["market_start_epoch"]), timezone.utc).strftime("%H:%M")
    end = datetime.fromtimestamp(float(position["market_end_epoch"]), timezone.utc).strftime("%H:%M")
    return f"{start}-{end}"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def spot_quality(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    spreads: list[float] = []
    source_counts: list[int] = []
    failures = 0
    warnings = 0
    bad_medians = 0
    for snapshot in snapshots:
        spot = snapshot.get("spot") or {}
        quotes = spot.get("quotes") or []
        prices = [float(item["price"]) for item in quotes if item.get("price") is not None]
        if prices:
            source_counts.append(len(prices))
            spreads.append(max(prices) - min(prices))
            if abs(float(spot["median_price"]) - statistics.median(prices)) > 1e-9:
                bad_medians += 1
        failures += len(spot.get("failed_sources") or [])
        warnings += len(spot.get("warnings") or [])
    return {
        "snapshots_with_spot": len(source_counts),
        "median_source_count": statistics.median(source_counts) if source_counts else 0,
        "median_cross_source_spread_usd": statistics.median(spreads) if spreads else 0.0,
        "p95_cross_source_spread_usd": percentile(spreads, 95),
        "max_cross_source_spread_usd": max(spreads) if spreads else 0.0,
        "failed_source_count": failures,
        "warning_count": warnings,
        "bad_median_count": bad_medians,
    }


def book_digest(snapshot: dict[str, Any], key: str) -> dict[str, Any] | None:
    book = snapshot.get(key)
    return book if isinstance(book, dict) else None


def best_ask_price(snapshot: dict[str, Any], key: str) -> float | None:
    book = book_digest(snapshot, key)
    if not book:
        return None
    ask = book.get("best_ask") or {}
    value = ask.get("price")
    return float(value) if value is not None else None


def book_hash(snapshot: dict[str, Any], key: str) -> str | None:
    book = book_digest(snapshot, key)
    if not book:
        return None
    value = book.get("hash")
    return str(value) if value else None


def change_count(values: list[Any]) -> int:
    return sum(1 for left, right in zip(values, values[1:]) if left != right)


def latest_snapshot_lags(signals: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> list[float]:
    by_market: dict[str, list[float]] = defaultdict(list)
    for snapshot in snapshots:
        market = snapshot.get("market") or {}
        slug = market.get("slug")
        if slug is None:
            continue
        by_market[str(slug)].append(float(snapshot.get("timestamp", 0.0)))
    for rows in by_market.values():
        rows.sort()

    lags = []
    for signal in signals:
        slug = signal.get("market_slug")
        if slug is None:
            continue
        timestamp = float(signal.get("timestamp", 0.0))
        candidates = [item for item in by_market.get(str(slug), []) if item <= timestamp]
        if not candidates:
            continue
        lags.append(timestamp - candidates[-1])
    return lags


def data_freshness(snapshots: list[dict[str, Any]], signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(snapshots, key=lambda row: float(row.get("timestamp", 0.0)))
    market_snapshots = [row for row in ordered if row.get("market")]
    decision_snapshots = [
        row for row in market_snapshots if not bool(row.get("settle_only"))
    ]
    intervals = [
        float(right.get("timestamp", 0.0)) - float(left.get("timestamp", 0.0))
        for left, right in zip(ordered, ordered[1:])
        if float(right.get("timestamp", 0.0)) >= float(left.get("timestamp", 0.0))
    ]
    up_hashes = [book_hash(row, "up_book") for row in decision_snapshots if book_hash(row, "up_book")]
    down_hashes = [book_hash(row, "down_book") for row in decision_snapshots if book_hash(row, "down_book")]
    up_asks = [best_ask_price(row, "up_book") for row in decision_snapshots if best_ask_price(row, "up_book") is not None]
    down_asks = [
        best_ask_price(row, "down_book")
        for row in decision_snapshots
        if best_ask_price(row, "down_book") is not None
    ]
    lags = latest_snapshot_lags(signals, ordered)
    missing_up_book = sum(1 for row in decision_snapshots if not book_digest(row, "up_book"))
    missing_down_book = sum(1 for row in decision_snapshots if not book_digest(row, "down_book"))
    stale_signal_lags = sum(1 for lag in lags if lag > 7.5)
    return {
        "snapshots": len(ordered),
        "market_snapshots": len(market_snapshots),
        "decision_snapshots": len(decision_snapshots),
        "signals": len(signals),
        "settle_only_snapshots": sum(1 for row in market_snapshots if bool(row.get("settle_only"))),
        "median_snapshot_interval_seconds": statistics.median(intervals) if intervals else 0.0,
        "p95_snapshot_interval_seconds": percentile(intervals, 95),
        "max_snapshot_interval_seconds": max(intervals) if intervals else 0.0,
        "missing_up_book_snapshots": missing_up_book,
        "missing_down_book_snapshots": missing_down_book,
        "up_book_hash_changes": change_count(up_hashes),
        "down_book_hash_changes": change_count(down_hashes),
        "distinct_up_book_hashes": len(set(up_hashes)),
        "distinct_down_book_hashes": len(set(down_hashes)),
        "up_top_ask_changes": change_count(up_asks),
        "down_top_ask_changes": change_count(down_asks),
        "median_signal_snapshot_lag_seconds": statistics.median(lags) if lags else 0.0,
        "p95_signal_snapshot_lag_seconds": percentile(lags, 95),
        "max_signal_snapshot_lag_seconds": max(lags) if lags else 0.0,
        "stale_signal_snapshot_lags_over_7_5s": stale_signal_lags,
    }


def settlement_key_from_position(position: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(position["market_slug"]),
        str(position["side"]),
        f"{float(position['cost_usd']):.12f}",
        f"{float(position['shares']):.12f}",
    )


def settlement_key(settlement: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(settlement["market_slug"]),
        str(settlement["side"]),
        f"{float(settlement['cost_usd']):.12f}",
        f"{float(settlement['shares']):.12f}",
    )


def settlement_by_trade(trades: list[dict[str, Any]], settlements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for settlement in settlements:
        by_key[settlement_key(settlement)].append(settlement)
    paired = []
    for trade in trades:
        key = settlement_key_from_position(trade["position"])
        settlement = by_key[key].pop(0) if by_key.get(key) else None
        paired.append({"trade": trade, "settlement": settlement})
    return paired


def group_snapshots_by_market(snapshots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        market = snapshot.get("market") or {}
        slug = market.get("slug")
        if slug is not None:
            grouped[str(slug)].append(snapshot)
    return grouped


def nearest_snapshot(
    snapshots_by_market: dict[str, list[dict[str, Any]]],
    market_slug: str,
    timestamp: float,
) -> dict[str, Any] | None:
    rows = snapshots_by_market.get(market_slug) or []
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row.get("timestamp", 0.0)) - timestamp))


def snapshot_spot_price(snapshot: dict[str, Any] | None) -> float | None:
    if snapshot is None:
        return None
    spot = snapshot.get("spot") or {}
    value = spot.get("median_price")
    return float(value) if value is not None else None


def proxy_winners_from_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    max_start_lag_seconds: float = 15.0,
    max_end_lag_seconds: float = 30.0,
) -> dict[str, str]:
    """Infer interval winners from observed spot snapshots for model scoring.

    Trade PnL still uses official Polymarket settlements when available. This
    proxy map lets calibration score all observed completed markets, not just
    the subset the trader happened to enter.
    """
    ordered = sorted(snapshots, key=lambda row: float(row.get("timestamp", 0.0)))
    markets: dict[str, dict[str, Any]] = {}
    grouped = group_snapshots_by_market(ordered)
    for snapshot in ordered:
        market = snapshot.get("market") or {}
        slug = market.get("slug")
        if slug is None:
            continue
        markets[str(slug)] = market

    winners: dict[str, str] = {}
    for slug, market in markets.items():
        start_epoch = float(market.get("start_epoch", 0.0))
        end_epoch = float(market.get("end_epoch", 0.0))
        if start_epoch <= 0.0 or end_epoch <= start_epoch:
            continue
        market_rows = [
            row
            for row in grouped.get(slug, [])
            if snapshot_spot_price(row) is not None
            and float(row.get("timestamp", 0.0)) >= start_epoch
        ]
        if not market_rows:
            continue
        start_row = min(market_rows, key=lambda row: float(row.get("timestamp", 0.0)))
        if float(start_row.get("timestamp", 0.0)) - start_epoch > max_start_lag_seconds:
            continue
        end_rows = [
            row
            for row in ordered
            if snapshot_spot_price(row) is not None
            and float(row.get("timestamp", 0.0)) >= end_epoch
        ]
        if not end_rows:
            continue
        end_row = min(end_rows, key=lambda row: float(row.get("timestamp", 0.0)))
        if float(end_row.get("timestamp", 0.0)) - end_epoch > max_end_lag_seconds:
            continue
        start_price = snapshot_spot_price(start_row)
        end_price = snapshot_spot_price(end_row)
        if start_price is None or end_price is None:
            continue
        winners[slug] = "UP" if end_price >= start_price else "DOWN"
    return winners


def reconstructed_trade_cohort(
    pair: dict[str, Any],
    snapshots_by_market: dict[str, list[dict[str, Any]]],
) -> str:
    trade = pair["trade"]
    position = trade["position"]
    decision = trade.get("decision") or {}
    explicit = decision.get("trade_cohort")
    if explicit and explicit != "none":
        return str(explicit)

    side = str(position.get("side", ""))
    raw_probability = float(decision.get("raw_probability", decision.get("probability", 0.5)))
    fill_price = float(position.get("price", 0.0))
    if raw_probability >= 0.5:
        return "directional_confidence"
    if fill_price >= 0.5:
        return "low_confidence_full_price"

    opened_at = float(position["opened_at"])
    seconds_to_end = float(position["market_end_epoch"]) - opened_at
    start_price = position.get("start_price_proxy")
    snapshot = nearest_snapshot(
        snapshots_by_market,
        str(position["market_slug"]),
        opened_at,
    )
    spot_price = None
    if snapshot is not None:
        spot = snapshot.get("spot") or {}
        if spot.get("median_price") is not None:
            spot_price = float(spot["median_price"])

    rebound = False
    if start_price is not None and spot_price is not None:
        distance = spot_price - float(start_price)
        rebound = (side == "UP" and distance < 0.0) or (side == "DOWN" and distance > 0.0)
    base = "underdog_rebound" if rebound else "underdog_continuation"
    return f"late_{base}" if seconds_to_end <= 60.0 else base


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(pairs)
    pnl_values = [
        float(pair["settlement"]["pnl_usd"])
        for pair in pairs
        if pair.get("settlement") is not None
    ]
    costs = [float(pair["trade"]["position"]["cost_usd"]) for pair in pairs]
    fill_prices = [float(pair["trade"]["position"]["price"]) for pair in pairs]
    probabilities = [
        float((pair["trade"].get("decision") or {}).get("probability", 0.5))
        for pair in pairs
    ]
    raw_probabilities = [
        float(
            (pair["trade"].get("decision") or {}).get(
                "raw_probability",
                (pair["trade"].get("decision") or {}).get("probability", 0.5),
            )
        )
        for pair in pairs
    ]
    seconds_to_end = [
        float(pair["trade"]["position"]["market_end_epoch"])
        - float(pair["trade"]["position"]["opened_at"])
        for pair in pairs
    ]
    return {
        "trades": count,
        "settled_trades": len(pnl_values),
        "pnl_usd": sum(pnl_values),
        "cost_usd": sum(costs),
        "win_rate": (
            sum(1 for pnl in pnl_values if pnl > 0.0) / len(pnl_values)
            if pnl_values
            else 0.0
        ),
        "avg_fill_price": statistics.mean(fill_prices) if fill_prices else 0.0,
        "avg_probability": statistics.mean(probabilities) if probabilities else 0.0,
        "avg_raw_probability": statistics.mean(raw_probabilities) if raw_probabilities else 0.0,
        "avg_seconds_to_end": statistics.mean(seconds_to_end) if seconds_to_end else 0.0,
    }


def cohort_diagnostics(
    pairs: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshots_by_market = group_snapshots_by_market(snapshots)
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    price_buckets: dict[str, list[dict[str, Any]]] = {
        "fill_below_20c": [],
        "fill_below_35c": [],
        "fill_below_50c": [],
        "fill_50c_or_higher": [],
    }
    for pair in pairs:
        position = pair["trade"]["position"]
        fill = float(position["price"])
        cohort = reconstructed_trade_cohort(pair, snapshots_by_market)
        by_cohort[cohort].append(pair)
        if fill < 0.20:
            price_buckets["fill_below_20c"].append(pair)
        if fill < 0.35:
            price_buckets["fill_below_35c"].append(pair)
        if fill < 0.50:
            price_buckets["fill_below_50c"].append(pair)
        else:
            price_buckets["fill_50c_or_higher"].append(pair)
    return {
        "trade_cohorts": {
            name: summarize_pairs(rows)
            for name, rows in sorted(by_cohort.items())
        },
        "price_buckets": {
            name: summarize_pairs(rows)
            for name, rows in price_buckets.items()
        },
    }


def suppressed_edge_diagnostics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    executable_rows = []
    for signal in signals:
        decision = signal.get("decision") or {}
        effective_cost = float(decision.get("effective_cost") or 0.0)
        if effective_cost <= 0.0:
            continue
        probability = float(decision.get("probability", 0.5))
        raw_probability = float(decision.get("raw_probability", probability))
        executable_rows.append(
            {
                "cohort": str(decision.get("trade_cohort") or "none"),
                "reason": str(decision.get("reason") or ""),
                "raw_edge": raw_probability - effective_cost,
                "adjusted_edge": probability - effective_cost,
                "haircut": float(decision.get("probability_haircut") or 0.0),
                "effective_cost": effective_cost,
                "executable_price": float(decision.get("executable_price") or 0.0),
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        raw_edges = [float(row["raw_edge"]) for row in rows]
        adjusted_edges = [float(row["adjusted_edge"]) for row in rows]
        haircuts = [float(row["haircut"]) for row in rows]
        prices = [float(row["executable_price"]) for row in rows]
        return {
            "signals": len(rows),
            "raw_positive_edges": sum(1 for edge in raw_edges if edge >= 0.0),
            "adjusted_positive_edges": sum(1 for edge in adjusted_edges if edge >= 0.0),
            "suppressed_raw_positive_edges": sum(
                1
                for row in rows
                if float(row["raw_edge"]) >= 0.0 and float(row["adjusted_edge"]) < 0.0
            ),
            "near_raw_edges": sum(1 for edge in raw_edges if edge >= -0.005),
            "avg_raw_edge": statistics.mean(raw_edges) if raw_edges else 0.0,
            "avg_adjusted_edge": statistics.mean(adjusted_edges) if adjusted_edges else 0.0,
            "avg_probability_haircut": statistics.mean(haircuts) if haircuts else 0.0,
            "avg_executable_price": statistics.mean(prices) if prices else 0.0,
        }

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in executable_rows:
        by_cohort[str(row["cohort"])].append(row)
    return {
        "overall": summarize(executable_rows),
        "by_cohort": {
            cohort: summarize(rows)
            for cohort, rows in sorted(by_cohort.items())
        },
    }


def calibration(signals: list[dict[str, Any]], winning_by_market: dict[str, str]) -> dict[str, Any]:
    ensemble_errors: list[float] = []
    bucketed: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_probability": 0.0, "wins": 0})
    model_errors: dict[str, list[float]] = defaultdict(list)
    model_direction: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "correct": 0})
    for signal in signals:
        slug = signal.get("market_slug")
        winner = winning_by_market.get(slug)
        if winner is None:
            continue
        outcome_up = 1.0 if winner == "UP" else 0.0
        forecast = signal.get("forecast") or {}
        p_up = float(forecast.get("p_up", 0.5))
        ensemble_errors.append((p_up - outcome_up) ** 2)
        bucket_floor = int(min(9, max(0, math.floor(p_up * 10))))
        bucket = f"{bucket_floor / 10:.1f}-{(bucket_floor + 1) / 10:.1f}"
        bucketed[bucket]["count"] += 1
        bucketed[bucket]["sum_probability"] += p_up
        bucketed[bucket]["wins"] += outcome_up
        for model in forecast.get("forecasts") or []:
            name = str(model.get("name"))
            mp = float(model.get("p_up", 0.5))
            model_errors[name].append((mp - outcome_up) ** 2)
            model_direction[name]["count"] += 1
            predicted = "UP" if mp >= 0.5 else "DOWN"
            if predicted == winner:
                model_direction[name]["correct"] += 1
    return {
        "ensemble_brier": statistics.mean(ensemble_errors) if ensemble_errors else None,
        "settled_signal_count": len(ensemble_errors),
        "buckets": {
            bucket: {
                "count": int(data["count"]),
                "avg_probability": data["sum_probability"] / data["count"],
                "actual_up_rate": data["wins"] / data["count"],
            }
            for bucket, data in sorted(bucketed.items())
            if data["count"]
        },
        "models": {
            name: {
                "brier": statistics.mean(errors),
                "count": len(errors),
                "direction_accuracy": (
                    model_direction[name]["correct"] / model_direction[name]["count"]
                    if model_direction[name]["count"]
                    else None
                ),
            }
            for name, errors in sorted(model_errors.items())
        },
    }


def market_level_calibration(signals: list[dict[str, Any]], winning_by_market: dict[str, str]) -> dict[str, Any]:
    latest_by_market: dict[str, dict[str, Any]] = {}
    for signal in signals:
        slug = signal.get("market_slug")
        if slug not in winning_by_market:
            continue
        if not signal.get("forecast"):
            continue
        timestamp = float(signal.get("timestamp", 0.0))
        previous = latest_by_market.get(str(slug))
        if previous is None or timestamp >= float(previous.get("timestamp", 0.0)):
            latest_by_market[str(slug)] = signal
    result = calibration(list(latest_by_market.values()), winning_by_market)
    result["settled_market_count"] = result.pop("settled_signal_count")
    return result


def execution_quality(executions: list[dict[str, Any]]) -> dict[str, Any]:
    dry_runs = [row for row in executions if row.get("dry_run")]
    trade_intents = [row for row in executions if not row.get("dry_run")]
    sims = [row.get("execution") or {} for row in trade_intents]
    dry_sims = [row.get("execution") or {} for row in dry_runs]
    accepted = [item for item in sims if item.get("accepted")]
    rejected = [item for item in sims if item and not item.get("accepted")]
    latencies = [float(item.get("latency_ms", 0.0)) for item in sims if item]
    drifts = []
    for item in sims:
        signal_ask = item.get("top_ask_at_signal")
        submit_ask = item.get("top_ask_at_submit")
        if signal_ask is not None and submit_ask is not None:
            drifts.append(float(submit_ask) - float(signal_ask))
    return {
        "intents": len(sims),
        "dry_run_intents": len(dry_sims),
        "dry_run_accepted": sum(1 for item in dry_sims if item.get("accepted")),
        "dry_run_rejected": sum(1 for item in dry_sims if item and not item.get("accepted")),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(
            sorted(
                {
                    reason: sum(1 for item in rejected if item.get("reason") == reason)
                    for reason in {item.get("reason") for item in rejected}
                }.items()
            )
        ),
        "latency_ms_median": statistics.median(latencies) if latencies else 0.0,
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_max": max(latencies) if latencies else 0.0,
        "submit_top_ask_drift_median": statistics.median(drifts) if drifts else 0.0,
        "submit_top_ask_drift_max": max(drifts) if drifts else 0.0,
    }


def trading_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    markets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "cost_usd": 0.0, "payout_usd": 0.0, "pnl_usd": 0.0, "sides": set(), "winner": None}
    )
    pnl_sequence = []
    stress = {1: 0.0, 2: 0.0, 3: 0.0}
    for pair in pairs:
        trade = pair["trade"]
        settlement = pair["settlement"]
        position = trade["position"]
        slug = position["market_slug"]
        row = markets[slug]
        row["trades"] += 1
        row["cost_usd"] += float(position["cost_usd"])
        row["sides"].add(position["side"])
        for cents in stress:
            stress[cents] -= float(position["shares"]) * cents / 100.0
        if settlement is not None:
            row["payout_usd"] += float(settlement["payout_usd"])
            row["pnl_usd"] += float(settlement["pnl_usd"])
            row["winner"] = settlement["winning_side"]
            pnl_sequence.append(float(settlement["pnl_usd"]))
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    losing_streak = 0
    max_losing_streak = 0
    for pnl in pnl_sequence:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = min(max_drawdown, cumulative - peak)
        if pnl < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0
    total_pnl = sum(pnl_sequence)
    return {
        "total_trades": len(pairs),
        "settled_trades": len(pnl_sequence),
        "realized_pnl_usd": total_pnl,
        "max_drawdown_usd": max_drawdown,
        "max_losing_streak": max_losing_streak,
        "win_rate": sum(1 for pnl in pnl_sequence if pnl > 0) / len(pnl_sequence) if pnl_sequence else 0.0,
        "adverse_fill_stress_pnl_delta": {f"{cents}c": delta for cents, delta in stress.items()},
        "markets": {
            slug: {
                "trades": row["trades"],
                "cost_usd": row["cost_usd"],
                "payout_usd": row["payout_usd"],
                "pnl_usd": row["pnl_usd"],
                "sides": sorted(row["sides"]),
                "winner": row["winner"],
            }
            for slug, row in sorted(markets.items())
        },
    }


def markdown_report(report: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    lines = [
        "# Polymarket BTC 5m Evidence Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"Window: `{report['window']['since'] or 'beginning'}` to `{report['window']['until'] or 'now'}`",
        "",
        "## Summary",
        "",
        f"- Trades: `{report['trading']['total_trades']}`",
        f"- Settled trades: `{report['trading']['settled_trades']}`",
        f"- Realized PnL: `${report['trading']['realized_pnl_usd']:.2f}`",
        f"- Max drawdown: `${report['trading']['max_drawdown_usd']:.2f}`",
        f"- Win rate by trade: `{report['trading']['win_rate']:.3f}`",
        f"- Trades below 50c: `{report['cohorts']['price_buckets']['fill_below_50c']['trades']}`",
        f"- Execution intents: `{report['execution']['intents']}` accepted / rejected: `{report['execution']['accepted']}` / `{report['execution']['rejected']}`",
        f"- Manual execution dry runs: `{report['execution']['dry_run_intents']}` accepted / rejected: `{report['execution']['dry_run_accepted']}` / `{report['execution']['dry_run_rejected']}`",
        f"- Median execution recheck latency: `{report['execution']['latency_ms_median']:.1f} ms`",
        f"- Ensemble Brier score: `{report['calibration']['ensemble_brier']}`",
        f"- Resolved markets for calibration: `{report['counts']['calibration_markets']}`",
        "",
        "## Trades",
        "",
        "| # | Opened UTC | Market | Window | Side | Entry | Shares | Cost | Winner | Payout | PnL |",
        "|---:|---|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for index, pair in enumerate(pairs, 1):
        trade = pair["trade"]
        settlement = pair["settlement"] or {}
        position = trade["position"]
        lines.append(
            "| "
            f"{index} | {iso_from_epoch(float(position['opened_at']))} | `{position['market_slug']}` | "
            f"{market_window(position)} | {position['side']} | {float(position['price']):.4f} | "
            f"{float(position['shares']):.4f} | ${float(position['cost_usd']):.2f} | "
            f"{settlement.get('winning_side', 'OPEN')} | ${float(settlement.get('payout_usd', 0.0)):.2f} | "
            f"${float(settlement.get('pnl_usd', 0.0)):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Market PnL",
            "",
            "| Market | Sides | Winner | Trades | Cost | Payout | PnL |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for slug, row in report["trading"]["markets"].items():
        lines.append(
            f"| `{slug}` | {','.join(row['sides'])} | {row['winner']} | {row['trades']} | "
            f"${row['cost_usd']:.2f} | ${row['payout_usd']:.2f} | ${row['pnl_usd']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Trade Cohorts",
            "",
            "| Cohort | Trades | Win Rate | Cost | PnL | Avg Fill | Avg p | Avg Raw p | Avg Seconds To End |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["cohorts"]["trade_cohorts"].items():
        lines.append(
            f"| `{name}` | {row['trades']} | {row['win_rate']:.3f} | "
            f"${row['cost_usd']:.2f} | ${row['pnl_usd']:.2f} | "
            f"{row['avg_fill_price']:.3f} | {row['avg_probability']:.3f} | "
            f"{row['avg_raw_probability']:.3f} | {row['avg_seconds_to_end']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Price Buckets",
            "",
            "| Bucket | Trades | Win Rate | Cost | PnL | Avg Fill | Avg p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["cohorts"]["price_buckets"].items():
        lines.append(
            f"| `{name}` | {row['trades']} | {row['win_rate']:.3f} | "
            f"${row['cost_usd']:.2f} | ${row['pnl_usd']:.2f} | "
            f"{row['avg_fill_price']:.3f} | {row['avg_probability']:.3f} |"
        )
    edge_overall = report["suppressed_edges"]["overall"]
    lines.extend(
        [
            "",
            "## Raw Edge Diagnostics",
            "",
            f"- Executable signal ticks: `{edge_overall['signals']}`",
            f"- Raw positive-edge ticks: `{edge_overall['raw_positive_edges']}`",
            f"- Adjusted positive-edge ticks: `{edge_overall['adjusted_positive_edges']}`",
            f"- Raw positive edges suppressed by calibration: `{edge_overall['suppressed_raw_positive_edges']}`",
            f"- Near-edge raw ticks within 0.5c: `{edge_overall['near_raw_edges']}`",
            "",
            "| Cohort | Signals | Raw +Edge | Adjusted +Edge | Suppressed Raw +Edge | Avg Raw Edge | Avg Adjusted Edge | Avg Haircut | Avg Price |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["suppressed_edges"]["by_cohort"].items():
        lines.append(
            f"| `{name}` | {row['signals']} | {row['raw_positive_edges']} | "
            f"{row['adjusted_positive_edges']} | {row['suppressed_raw_positive_edges']} | "
            f"{row['avg_raw_edge']:.4f} | {row['avg_adjusted_edge']:.4f} | "
            f"{row['avg_probability_haircut']:.4f} | {row['avg_executable_price']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Execution Quality",
            "",
            f"- Rejection reasons: `{json.dumps(report['execution']['rejection_reasons'], sort_keys=True)}`",
            f"- P95 latency: `{report['execution']['latency_ms_p95']:.1f} ms`",
            f"- Max latency: `{report['execution']['latency_ms_max']:.1f} ms`",
            f"- Median submit top-ask drift: `{report['execution']['submit_top_ask_drift_median']:.4f}`",
            f"- Max submit top-ask drift: `{report['execution']['submit_top_ask_drift_max']:.4f}`",
            "",
            "## Data Freshness",
            "",
            f"- Market snapshots: `{report['data_freshness']['market_snapshots']}`",
            f"- Decision-capable snapshots: `{report['data_freshness']['decision_snapshots']}`",
            f"- Settle-only snapshots: `{report['data_freshness']['settle_only_snapshots']}`",
            f"- Median snapshot interval: `{report['data_freshness']['median_snapshot_interval_seconds']:.2f}s`",
            f"- P95 snapshot interval: `{report['data_freshness']['p95_snapshot_interval_seconds']:.2f}s`",
            f"- Max snapshot interval: `{report['data_freshness']['max_snapshot_interval_seconds']:.2f}s`",
            f"- Missing UP/DOWN books on decision-capable snapshots: `{report['data_freshness']['missing_up_book_snapshots']}` / `{report['data_freshness']['missing_down_book_snapshots']}`",
            f"- UP/DOWN book hash changes: `{report['data_freshness']['up_book_hash_changes']}` / `{report['data_freshness']['down_book_hash_changes']}`",
            f"- UP/DOWN top-ask changes: `{report['data_freshness']['up_top_ask_changes']}` / `{report['data_freshness']['down_top_ask_changes']}`",
            f"- Median signal-to-snapshot lag: `{report['data_freshness']['median_signal_snapshot_lag_seconds']:.3f}s`",
            f"- P95 signal-to-snapshot lag: `{report['data_freshness']['p95_signal_snapshot_lag_seconds']:.3f}s`",
            f"- Signal lags over 7.5s: `{report['data_freshness']['stale_signal_snapshot_lags_over_7_5s']}`",
            "",
            "## Spot Quality",
            "",
            f"- Snapshots with spot: `{report['spot_quality']['snapshots_with_spot']}`",
            f"- Median source count: `{report['spot_quality']['median_source_count']}`",
            f"- Median cross-source spread: `${report['spot_quality']['median_cross_source_spread_usd']:.2f}`",
            f"- P95 cross-source spread: `${report['spot_quality']['p95_cross_source_spread_usd']:.2f}`",
            f"- Failed source count: `{report['spot_quality']['failed_source_count']}`",
            f"- Warning count: `{report['spot_quality']['warning_count']}`",
            "",
            "## Calibration",
            "",
            f"Signal-tick samples: `{report['calibration']['settled_signal_count']}`",
            "",
            "| Bucket | Count | Avg p(UP) | Actual UP rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket, row in report["calibration"]["buckets"].items():
        lines.append(
            f"| {bucket} | {row['count']} | {row['avg_probability']:.3f} | {row['actual_up_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Market-Level Calibration",
            "",
            f"Last-signal market samples: `{report['market_level_calibration']['settled_market_count']}`",
            "",
            "| Bucket | Count | Avg p(UP) | Actual UP rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket, row in report["market_level_calibration"]["buckets"].items():
        lines.append(
            f"| {bucket} | {row['count']} | {row['avg_probability']:.3f} | {row['actual_up_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Market-Level Model Scores",
            "",
            "| Model | Markets | Brier | Direction Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["market_level_calibration"]["models"].items():
        direction = row["direction_accuracy"]
        lines.append(
            f"| `{name}` | {row['count']} | {row['brier']:.4f} | "
            f"{direction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Model Scores",
            "",
            "| Model | Count | Brier | Direction Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in report["calibration"]["models"].items():
        direction = row["direction_accuracy"]
        lines.append(
            f"| `{name}` | {row['count']} | {row['brier']:.4f} | "
            f"{direction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adverse Fill Stress",
            "",
            "PnL delta if every filled buy were worse by this many cents per share:",
            "",
        ]
    )
    for label, delta in report["trading"]["adverse_fill_stress_pnl_delta"].items():
        lines.append(f"- `{label}`: `${delta:.2f}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-trading evidence report")
    parser.add_argument("--output-dir", default="outputs/paper_trader")
    parser.add_argument("--since", default=None, help="Inclusive UTC ISO timestamp")
    parser.add_argument("--until", default=None, help="Exclusive UTC ISO timestamp")
    parser.add_argument("--report-dir", default="outputs/paper_trader/reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    since = parse_time(args.since)
    until = parse_time(args.until)
    trades = [row for row in load_jsonl(output_dir / "trades.jsonl") if in_window(row, since, until)]
    all_settlements = load_jsonl(output_dir / "official_settlements.jsonl")
    settlements = [row for row in all_settlements if in_window(row, since, until)]
    signals = [row for row in load_jsonl(output_dir / "signals.jsonl") if in_window(row, since, until)]
    snapshots = [row for row in load_jsonl(output_dir / "snapshots.jsonl") if in_window(row, since, until)]
    executions = [row for row in load_jsonl(output_dir / "execution_simulations.jsonl") if in_window(row, since, until)]
    pairs = settlement_by_trade(trades, all_settlements)
    winning_by_market = proxy_winners_from_snapshots(snapshots)
    winning_by_market.update({row["market_slug"]: row["winning_side"] for row in all_settlements})
    report = {
        "window": {"since": args.since, "until": args.until},
        "trading": trading_summary(pairs),
        "cohorts": cohort_diagnostics(pairs, snapshots),
        "suppressed_edges": suppressed_edge_diagnostics(signals),
        "execution": execution_quality(executions),
        "data_freshness": data_freshness(snapshots, signals),
        "spot_quality": spot_quality(snapshots),
        "calibration": calibration(signals, winning_by_market),
        "market_level_calibration": market_level_calibration(signals, winning_by_market),
        "counts": {
            "snapshots": len(snapshots),
            "signals": len(signals),
            "trades": len(trades),
            "settlements": len(settlements),
            "execution_simulations": len(executions),
            "calibration_markets": len(winning_by_market),
        },
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"evidence_{stamp}.json"
    md_path = report_dir / f"evidence_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(report, pairs), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
