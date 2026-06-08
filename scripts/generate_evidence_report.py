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
        f"- Execution intents: `{report['execution']['intents']}` accepted / rejected: `{report['execution']['accepted']}` / `{report['execution']['rejected']}`",
        f"- Manual execution dry runs: `{report['execution']['dry_run_intents']}` accepted / rejected: `{report['execution']['dry_run_accepted']}` / `{report['execution']['dry_run_rejected']}`",
        f"- Median execution recheck latency: `{report['execution']['latency_ms_median']:.1f} ms`",
        f"- Ensemble Brier score: `{report['calibration']['ensemble_brier']}`",
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
            "## Execution Quality",
            "",
            f"- Rejection reasons: `{json.dumps(report['execution']['rejection_reasons'], sort_keys=True)}`",
            f"- P95 latency: `{report['execution']['latency_ms_p95']:.1f} ms`",
            f"- Max latency: `{report['execution']['latency_ms_max']:.1f} ms`",
            f"- Median submit top-ask drift: `{report['execution']['submit_top_ask_drift_median']:.4f}`",
            f"- Max submit top-ask drift: `{report['execution']['submit_top_ask_drift_max']:.4f}`",
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
    winning_by_market = {row["market_slug"]: row["winning_side"] for row in all_settlements}
    report = {
        "window": {"since": args.since, "until": args.until},
        "trading": trading_summary(pairs),
        "execution": execution_quality(executions),
        "spot_quality": spot_quality(snapshots),
        "calibration": calibration(signals, winning_by_market),
        "counts": {
            "snapshots": len(snapshots),
            "signals": len(signals),
            "trades": len(trades),
            "settlements": len(settlements),
            "execution_simulations": len(executions),
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
