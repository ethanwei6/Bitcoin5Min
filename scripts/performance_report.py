#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path("outputs/paper_trader")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-trading performance")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    trades = load_jsonl(output_dir / "trades.jsonl")
    official_path = output_dir / "official_settlements.jsonl"
    settlement_path = official_path if official_path.exists() else output_dir / "settlements.jsonl"
    settlements = load_jsonl(settlement_path)
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    by_market: dict[str, dict[str, float]] = defaultdict(
        lambda: {"trades": 0, "cost_usd": 0.0, "payout_usd": 0.0, "pnl_usd": 0.0}
    )
    for trade in trades:
        position = trade["position"]
        row = by_market[position["market_slug"]]
        row["trades"] += 1
        row["cost_usd"] += float(position["cost_usd"])
    for settlement in settlements:
        row = by_market[settlement["market_slug"]]
        row["payout_usd"] += float(settlement["payout_usd"])
        row["pnl_usd"] += float(settlement["pnl_usd"])

    open_positions = [
        position
        for position in state.get("positions", [])
        if not position.get("settled", False)
    ]
    report = {
        "starting_cash_usd": 1000.0,
        "cash_usd": state.get("cash_usd"),
        "realized_pnl_usd": state.get("realized_pnl_usd", 0.0),
        "total_trades": len(trades),
        "settled_trades": len(settlements),
        "open_trades": len(open_positions),
        "open_cost_usd": sum(float(position["cost_usd"]) for position in open_positions),
        "settlement_file": settlement_path.name,
        "markets": by_market,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
