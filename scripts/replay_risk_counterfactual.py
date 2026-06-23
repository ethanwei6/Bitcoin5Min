#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poly_5m_bot.config import BotConfig, load_config
from poly_5m_bot.models import EnsembleForecast, ModelForecast
from poly_5m_bot.orderbook import BookLevel, OrderBook
from poly_5m_bot.risk import RiskEngine


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def pair_trades(output_dir: Path) -> list[dict[str, Any]]:
    trades = load_jsonl(output_dir / "trades.jsonl")
    official = load_jsonl(output_dir / "official_settlements.jsonl")
    proxy = load_jsonl(output_dir / "settlements.jsonl")
    settlements = official if official else proxy
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for settlement in settlements:
        by_key[settlement_key(settlement)].append(settlement)
    paired = []
    for trade in trades:
        position = trade["position"]
        settlement = by_key[settlement_key_from_position(position)].pop(0)
        paired.append({"trade": trade, "settlement": settlement})
    return sorted(paired, key=lambda pair: float(pair["trade"]["position"]["opened_at"]))


def synthetic_distance(side: str, cohort: str) -> float:
    if "rebound" in cohort:
        return -1.0 if side == "UP" else 1.0
    if "underdog_continuation" in cohort:
        return 1.0 if side == "UP" else -1.0
    return 0.0


def synthetic_forecast(pair: dict[str, Any], config: BotConfig) -> EnsembleForecast:
    trade = pair["trade"]
    position = trade["position"]
    decision = trade["decision"]
    side = str(position["side"])
    raw_probability = float(decision.get("raw_probability", decision.get("probability", 0.5)))
    p_up = raw_probability if side == "UP" else 1.0 - raw_probability
    forecasts = [
        ModelForecast(
            name=f"logged_signal_{index}",
            p_up=p_up,
            expected_end_price=0.0,
            confidence=abs(p_up - 0.5) * 2.0,
            reason="replayed from logged paper-trade decision",
        )
        for index in range(max(config.min_models_required, 1))
    ]
    total_weight = float(max(config.min_models_required, 1))
    return EnsembleForecast(
        p_up=p_up,
        expected_end_price=0.0,
        confidence=abs(p_up - 0.5) * 2.0,
        forecasts=forecasts,
        majority_side=side,
        majority_count=len(forecasts),
        majority_weight=total_weight,
        total_weight=total_weight,
        raw_p_up=p_up,
        spot_distance_from_start=synthetic_distance(side, str(decision.get("trade_cohort", ""))),
    )


def synthetic_books(pair: dict[str, Any]) -> tuple[OrderBook, OrderBook]:
    trade = pair["trade"]
    position = trade["position"]
    decision = trade["decision"]
    execution = trade.get("execution") or {}
    side = str(position["side"])
    executable_price = float(decision.get("executable_price", position["price"]))
    fill_shares = float(execution.get("fill_shares", position["shares"]))
    up_price = executable_price if side == "UP" else max(0.01, 1.0 - executable_price)
    down_price = executable_price if side == "DOWN" else max(0.01, 1.0 - executable_price)
    up_book = OrderBook(
        token_id="up-token",
        bids=[BookLevel(price=max(up_price - 0.01, 0.01), size=fill_shares)],
        asks=[BookLevel(price=up_price, size=fill_shares)],
        tick_size="0.01",
        min_order_size=0.0,
        hash="logged-fill",
    )
    down_book = OrderBook(
        token_id="down-token",
        bids=[BookLevel(price=max(down_price - 0.01, 0.01), size=fill_shares)],
        asks=[BookLevel(price=down_price, size=fill_shares)],
        tick_size="0.01",
        min_order_size=0.0,
        hash="logged-fill",
    )
    return up_book, down_book


def settle_open_positions(
    open_positions: list[dict[str, Any]],
    *,
    now: float,
) -> tuple[list[dict[str, Any]], float]:
    remaining = []
    pnl = 0.0
    for position in open_positions:
        if float(position["market_end_epoch"]) > now:
            remaining.append(position)
            continue
        won = position["side"] == position["winning_side"]
        payout = float(position["shares"]) if won else 0.0
        pnl += payout - float(position["cost_usd"])
    return remaining, pnl


def market_exposure(open_positions: list[dict[str, Any]], market_slug: str) -> float:
    return sum(
        float(position["cost_usd"])
        for position in open_positions
        if position["market_slug"] == market_slug
    )


def market_entry_counts(open_positions: list[dict[str, Any]], rows: list[dict[str, Any]], market_slug: str) -> tuple[int, int, int]:
    retained = [
        row for row in rows
        if row["retained"] and row["market_slug"] == market_slug
    ]
    up = sum(1 for row in retained if row["side"] == "UP")
    down = sum(1 for row in retained if row["side"] == "DOWN")
    return len(retained), up, down


def replay(output_dir: Path, config: BotConfig) -> dict[str, Any]:
    pairs = pair_trades(output_dir)
    risk = RiskEngine(config)
    cash = float(config.paper_starting_cash_usd)
    realized_pnl = 0.0
    peak_pnl = 0.0
    max_drawdown = 0.0
    open_positions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for pair in pairs:
        trade = pair["trade"]
        position = trade["position"]
        settlement = pair["settlement"]
        opened_at = float(position["opened_at"])
        open_positions, newly_realized = settle_open_positions(open_positions, now=opened_at)
        realized_pnl += newly_realized
        cash += newly_realized
        peak_pnl = max(peak_pnl, realized_pnl)
        max_drawdown = max(max_drawdown, peak_pnl - realized_pnl)

        market_slug = str(position["market_slug"])
        entries, up_entries, down_entries = market_entry_counts(open_positions, rows, market_slug)
        forecast = synthetic_forecast(pair, config)
        up_book, down_book = synthetic_books(pair)
        decision = risk.decide(
            forecast=forecast,
            up_book=up_book,
            down_book=down_book,
            cash_usd=cash,
            market_exposure_usd=market_exposure(open_positions, market_slug),
            market_entries=entries,
            daily_pnl_usd=realized_pnl,
            daily_drawdown_usd=max_drawdown,
            consecutive_losses=0,
            seconds_from_start=opened_at - float(position["market_start_epoch"]),
            seconds_to_end=float(position["market_end_epoch"]) - opened_at,
            up_entries=up_entries,
            down_entries=down_entries,
        )

        original_cost = float(position["cost_usd"])
        retained_cost = 0.0
        retained_shares = 0.0
        retained = False
        retained_pnl = 0.0
        if decision.should_trade:
            execution = trade.get("execution") or {}
            fill_cost = float(execution.get("fill_cost_usd", original_cost))
            fill_shares = float(execution.get("fill_shares", position["shares"]))
            effective_cost = fill_cost / fill_shares
            retained_cost = min(float(decision.spend_usd), fill_cost, cash)
            if retained_cost > 0.0:
                retained = True
                retained_shares = retained_cost / effective_cost
                cash -= retained_cost
                open_positions.append(
                    {
                        "market_slug": market_slug,
                        "market_end_epoch": float(position["market_end_epoch"]),
                        "side": str(position["side"]),
                        "winning_side": str(settlement["winning_side"]),
                        "cost_usd": retained_cost,
                        "shares": retained_shares,
                    }
                )
                retained_pnl = (
                    retained_shares - retained_cost
                    if position["side"] == settlement["winning_side"]
                    else -retained_cost
                )

        rows.append(
            {
                "market_slug": market_slug,
                "side": str(position["side"]),
                "winner": str(settlement["winning_side"]),
                "cohort": str(trade["decision"].get("trade_cohort", "none")),
                "opened_at": opened_at,
                "entry_price": float(position["price"]),
                "original_cost_usd": original_cost,
                "original_pnl_usd": float(settlement["pnl_usd"]),
                "retained": retained,
                "retained_cost_usd": retained_cost,
                "retained_shares": retained_shares,
                "retained_pnl_usd": retained_pnl,
                "new_decision": asdict(decision),
                "original_decision": trade["decision"],
            }
        )

    open_positions, newly_realized = settle_open_positions(open_positions, now=float("inf"))
    realized_pnl += newly_realized
    peak_pnl = max(peak_pnl, realized_pnl)
    max_drawdown = max(max_drawdown, peak_pnl - realized_pnl)
    original_pnl = sum(row["original_pnl_usd"] for row in rows)
    retained_pnl = sum(row["retained_pnl_usd"] for row in rows)
    return {
        "summary": {
            "original_trades": len(rows),
            "retained_trades": sum(1 for row in rows if row["retained"]),
            "rejected_trades": sum(1 for row in rows if not row["retained"]),
            "original_cost_usd": sum(row["original_cost_usd"] for row in rows),
            "retained_cost_usd": sum(row["retained_cost_usd"] for row in rows),
            "original_pnl_usd": original_pnl,
            "retained_pnl_usd": retained_pnl,
            "pnl_delta_usd": retained_pnl - original_pnl,
            "retained_win_rate": (
                sum(1 for row in rows if row["retained"] and row["retained_pnl_usd"] > 0)
                / max(1, sum(1 for row in rows if row["retained"]))
            ),
            "max_drawdown_usd": max_drawdown,
        },
        "by_cohort": summarize_groups(rows, "cohort"),
        "by_reject_reason": dict(Counter(row["new_decision"]["reason"] for row in rows if not row["retained"])),
        "rows": rows,
        "scope_note": (
            "Same-fill retention counterfactual: it only resizes or rejects fills "
            "the historical paper trader actually took. It does not invent new "
            "opportunities from signals the original strategy skipped."
        ),
    }


def summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out = {}
    for group, group_rows in sorted(grouped.items()):
        retained = [row for row in group_rows if row["retained"]]
        out[group] = {
            "original_trades": len(group_rows),
            "retained_trades": len(retained),
            "original_cost_usd": sum(row["original_cost_usd"] for row in group_rows),
            "retained_cost_usd": sum(row["retained_cost_usd"] for row in group_rows),
            "original_pnl_usd": sum(row["original_pnl_usd"] for row in group_rows),
            "retained_pnl_usd": sum(row["retained_pnl_usd"] for row in group_rows),
            "retained_win_rate": (
                sum(1 for row in retained if row["retained_pnl_usd"] > 0) / len(retained)
                if retained else 0.0
            ),
        }
    return out


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Risk Calibration Counterfactual",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        report["scope_note"],
        "",
        "## Summary",
        "",
        f"- Original trades: `{summary['original_trades']}`",
        f"- Retained trades: `{summary['retained_trades']}` rejected: `{summary['rejected_trades']}`",
        f"- Original cost: `${summary['original_cost_usd']:.2f}` retained cost: `${summary['retained_cost_usd']:.2f}`",
        f"- Original PnL: `${summary['original_pnl_usd']:.2f}`",
        f"- Counterfactual retained PnL: `${summary['retained_pnl_usd']:.2f}`",
        f"- PnL delta: `${summary['pnl_delta_usd']:.2f}`",
        f"- Retained win rate: `{summary['retained_win_rate']:.3f}`",
        f"- Max drawdown on retained fills: `${summary['max_drawdown_usd']:.2f}`",
        "",
        "## By Cohort",
        "",
        "| Cohort | Original Trades | Retained Trades | Original Cost | Retained Cost | Original PnL | Retained PnL | Retained Win Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort, row in report["by_cohort"].items():
        lines.append(
            f"| `{cohort}` | {row['original_trades']} | {row['retained_trades']} | "
            f"${row['original_cost_usd']:.2f} | ${row['retained_cost_usd']:.2f} | "
            f"${row['original_pnl_usd']:.2f} | ${row['retained_pnl_usd']:.2f} | "
            f"{row['retained_win_rate']:.3f} |"
        )
    lines.extend(["", "## Reject Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted(report["by_reject_reason"].items()):
        lines.append(f"| `{reason}` | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay logged paper fills under the current risk config")
    parser.add_argument("--output-dir", default="outputs/paper_trader")
    parser.add_argument("--config", default="config/paper_btc_5m.json")
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    report = replay(output_dir, config)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"risk_counterfactual_{stamp}.json"
    md_path = report_dir / f"risk_counterfactual_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
