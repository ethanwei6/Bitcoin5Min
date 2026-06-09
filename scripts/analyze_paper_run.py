#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def pair_trades(output_dir: Path) -> list[dict[str, Any]]:
    trades = load_jsonl(output_dir / "trades.jsonl")
    official = load_jsonl(output_dir / "official_settlements.jsonl")
    proxy = load_jsonl(output_dir / "settlements.jsonl")
    settlements = official if official else proxy
    rows = []
    for index, trade in enumerate(trades):
        settlement = settlements[index] if index < len(settlements) else None
        position = trade["position"]
        pnl = float(settlement["pnl_usd"]) if settlement else None
        rows.append(
            {
                "index": index + 1,
                "trade": trade,
                "settlement": settlement,
                "market_slug": position["market_slug"],
                "side": position["side"],
                "price": float(position["price"]),
                "cost_usd": float(position["cost_usd"]),
                "opened_at": float(position["opened_at"]),
                "opened_utc": datetime.fromtimestamp(float(position["opened_at"]), timezone.utc),
                "seconds_from_start": float(position["opened_at"])
                - float(position["market_start_epoch"]),
                "executed_edge": (
                    float(trade["decision"]["probability"])
                    - (
                        float((trade.get("execution") or {}).get("fill_cost_usd", 0.0))
                        / float((trade.get("execution") or {}).get("fill_shares", 1.0))
                    )
                    if (trade.get("execution") or {}).get("fill_shares")
                    else float(trade["decision"].get("edge", 0.0))
                ),
                "pnl_usd": pnl,
            }
        )
    return rows


def pnl_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cum = 0.0
    peak = 0.0
    max_drawdown = 0.0
    peak_trade = 0
    drawdown_trade = 0
    curve = []
    for row in rows:
        if row["pnl_usd"] is None:
            continue
        cum += row["pnl_usd"]
        if cum > peak:
            peak = cum
            peak_trade = row["index"]
        drawdown = peak - cum
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            drawdown_trade = row["index"]
        curve.append({"trade": row["index"], "opened_utc": row["opened_utc"].isoformat(), "cum_pnl": cum})
    settled = [row for row in rows if row["pnl_usd"] is not None]
    return {
        "trades": len(rows),
        "settled_trades": len(settled),
        "realized_pnl_usd": cum,
        "peak_pnl_usd": peak,
        "peak_trade": peak_trade,
        "max_drawdown_usd": max_drawdown,
        "drawdown_trade": drawdown_trade,
        "win_rate": (
            sum(1 for row in settled if row["pnl_usd"] > 0) / len(settled)
            if settled
            else None
        ),
        "curve": curve,
    }


def grouped(rows: list[dict[str, Any]], key_fn) -> dict[Any, dict[str, Any]]:
    out: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        if row["pnl_usd"] is not None:
            out[key_fn(row)].append(row["pnl_usd"])
    return {
        key: {
            "trades": len(values),
            "pnl_usd": sum(values),
            "win_rate": sum(1 for value in values if value > 0) / len(values),
            "avg_pnl_usd": statistics.fmean(values),
        }
        for key, values in sorted(out.items(), key=lambda item: item[0])
    }


def market_table(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    markets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        markets[row["market_slug"]].append(row)
    ranked = sorted(
        markets.items(),
        key=lambda item: sum(row["pnl_usd"] or 0.0 for row in item[1]),
        reverse=reverse,
    )
    table = []
    for slug, market_rows in ranked[:10]:
        table.append(
            {
                "market_slug": slug,
                "trades": len(market_rows),
                "side": ",".join(sorted({row["side"] for row in market_rows})),
                "winner": (
                    market_rows[0]["settlement"].get("winning_side")
                    if market_rows[0]["settlement"]
                    else "OPEN"
                ),
                "pnl_usd": sum(row["pnl_usd"] or 0.0 for row in market_rows),
            }
        )
    return table


def counterfactuals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for limit in [50, 75, 100, 125, 150, 200, 300]:
        cum = 0.0
        peak = 0.0
        kept = 0
        stopped_at = None
        for row in rows:
            if row["pnl_usd"] is None or stopped_at is not None:
                continue
            kept += 1
            cum += row["pnl_usd"]
            peak = max(peak, cum)
            if peak - cum >= limit:
                stopped_at = row["index"]
        results[f"drawdown_stop_{limit}"] = {
            "kept_trades": kept,
            "pnl_usd": cum,
            "stopped_at_trade": stopped_at,
        }
    for cap in [0.55, 0.60, 0.65, 0.70]:
        kept_rows = [row for row in rows if row["pnl_usd"] is not None and row["price"] <= cap]
        results[f"price_cap_{cap:.2f}"] = {
            "kept_trades": len(kept_rows),
            "pnl_usd": sum(row["pnl_usd"] for row in kept_rows),
        }
    for cap in [0.15, 0.20, 0.25]:
        kept_rows = [
            row
            for row in rows
            if row["pnl_usd"] is not None and row["executed_edge"] <= cap
        ]
        results[f"edge_dislocation_cap_{cap:.2f}"] = {
            "kept_trades": len(kept_rows),
            "pnl_usd": sum(row["pnl_usd"] for row in kept_rows),
        }
    for start, end in [(15, 180), (15, 210), (30, 210), (45, 210)]:
        kept_rows = [
            row
            for row in rows
            if row["pnl_usd"] is not None and start <= row["seconds_from_start"] <= end
        ]
        results[f"entry_window_{start}_{end}s"] = {
            "kept_trades": len(kept_rows),
            "pnl_usd": sum(row["pnl_usd"] for row in kept_rows),
        }
    return results


def key_finding(report: dict[str, Any]) -> str:
    baseline = report["summary"]["realized_pnl_usd"]
    improvements = [
        (name, row)
        for name, row in report["counterfactuals"].items()
        if row["kept_trades"] >= 10 and row["pnl_usd"] > baseline
    ]
    if not improvements:
        return (
            "No simple single-rule counterfactual beat the ledger by enough to justify "
            "tightening production rules from this sample alone."
        )
    name, row = max(improvements, key=lambda item: item[1]["pnl_usd"])
    if name.startswith("edge_dislocation_cap"):
        return (
            f"The best simple improvement was `{name}`: it kept `{row['kept_trades']}` "
            f"trades and would have produced `${row['pnl_usd']:.2f}`. This points to "
            "overconfident model-market dislocations, not low confidence, as the main leak."
        )
    if name.startswith("entry_window"):
        return (
            f"The best simple improvement was `{name}`: it kept `{row['kept_trades']}` "
            f"trades and would have produced `${row['pnl_usd']:.2f}`. This points to "
            "opening/late-contract timing risk as the main leak."
        )
    return (
        f"The best simple improvement was `{name}`: it kept `{row['kept_trades']}` "
        f"trades and would have produced `${row['pnl_usd']:.2f}`."
    )


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# BTC 5m Paper Run Postmortem",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        f"- Trades: `{summary['trades']}` settled: `{summary['settled_trades']}`",
        f"- Realized PnL: `${summary['realized_pnl_usd']:.2f}`",
        f"- Peak PnL: `${summary['peak_pnl_usd']:.2f}` at trade `{summary['peak_trade']}`",
        f"- Max drawdown: `${summary['max_drawdown_usd']:.2f}` at trade `{summary['drawdown_trade']}`",
        f"- Win rate: `{summary['win_rate']:.3f}`",
        "",
        "## Key Failure",
        "",
        key_finding(report),
        "",
        "## Counterfactuals",
        "",
        "| Rule | Kept Trades | PnL | Stop Trade |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["counterfactuals"].items():
        lines.append(
            f"| `{name}` | {row['kept_trades']} | ${row['pnl_usd']:.2f} | {row.get('stopped_at_trade') or ''} |"
        )
    lines.extend(["", "## Hourly PnL UTC", "", "| Hour | Trades | PnL | Win Rate |", "|---:|---:|---:|---:|"])
    for hour, row in report["by_hour_utc"].items():
        lines.append(f"| {hour} | {row['trades']} | ${row['pnl_usd']:.2f} | {row['win_rate']:.3f} |")
    lines.extend(["", "## PnL By Side", "", "| Side | Trades | PnL | Win Rate |", "|---|---:|---:|---:|"])
    for side, row in report["by_side"].items():
        lines.append(f"| {side} | {row['trades']} | ${row['pnl_usd']:.2f} | {row['win_rate']:.3f} |")
    lines.extend(["", "## PnL By Entry Timing", "", "| Window | Trades | PnL | Win Rate |", "|---|---:|---:|---:|"])
    for window, row in report["by_entry_timing"].items():
        lines.append(f"| {window} | {row['trades']} | ${row['pnl_usd']:.2f} | {row['win_rate']:.3f} |")
    lines.extend(["", "## Worst Markets", "", "| Market | Trades | Side | Winner | PnL |", "|---|---:|---|---|---:|"])
    for row in report["worst_markets"]:
        lines.append(
            f"| `{row['market_slug']}` | {row['trades']} | {row['side']} | {row['winner']} | ${row['pnl_usd']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paper run PnL and failure regimes")
    parser.add_argument("--output-dir", default="outputs/paper_trader")
    parser.add_argument("--report-dir", default="outputs/paper_trader/reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = pair_trades(output_dir)
    report = {
        "summary": pnl_summary(rows),
        "by_hour_utc": grouped(rows, lambda row: row["opened_utc"].hour),
        "by_side": grouped(rows, lambda row: row["side"]),
        "by_entry_price": grouped(
            rows,
            lambda row: "cheap<=0.35" if row["price"] <= 0.35 else "expensive>=0.65" if row["price"] >= 0.65 else "mid",
        ),
        "by_entry_timing": grouped(
            rows,
            lambda row: (
                "<30s"
                if row["seconds_from_start"] < 30
                else "30-60s"
                if row["seconds_from_start"] < 60
                else "60-120s"
                if row["seconds_from_start"] < 120
                else "120-220s"
                if row["seconds_from_start"] < 220
                else "220s+"
            ),
        ),
        "worst_markets": market_table(rows, reverse=False),
        "best_markets": market_table(rows, reverse=True),
        "counterfactuals": counterfactuals(rows),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"postmortem_{stamp}.json"
    md_path = report_dir / f"postmortem_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
