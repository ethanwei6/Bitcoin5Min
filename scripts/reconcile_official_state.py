#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

from poly_5m_bot.config import load_config
from poly_5m_bot.resolution import GammaResolutionClient


OUTPUT_DIR = Path("outputs/paper_trader")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile paper trades against official market outcomes")
    parser.add_argument("--config", default="config/paper_btc_5m.json")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    resolution_client = GammaResolutionClient(config.polymarket)
    trades = load_jsonl(output_dir / "trades.jsonl")

    resolutions = {}
    for trade in trades:
        slug = trade["position"]["market_slug"]
        if slug not in resolutions:
            resolutions[slug] = resolution_client.get_resolution(slug)

    cash = config.paper_starting_cash_usd
    realized_pnl = 0.0
    positions = []
    official_settlements = []
    for trade in trades:
        position = dict(trade["position"])
        slug = position["market_slug"]
        cost = float(position["cost_usd"])
        cash -= cost
        resolution = resolutions.get(slug)
        settled = bool(resolution and resolution.closed and resolution.winning_side)
        position["settled"] = settled
        if settled:
            winning_side = resolution.winning_side
            payout = float(position["shares"]) if position["side"] == winning_side else 0.0
            pnl = payout - cost
            cash += payout
            realized_pnl += pnl
            official_settlements.append(
                {
                    "market_slug": slug,
                    "side": position["side"],
                    "winning_side": winning_side,
                    "shares": float(position["shares"]),
                    "cost_usd": cost,
                    "payout_usd": payout,
                    "pnl_usd": pnl,
                    "settlement_source": "gamma_official_outcome",
                    "resolution_source": resolution.source,
                }
            )
        positions.append(position)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_settlements.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in official_settlements),
        encoding="utf-8",
    )
    (output_dir / "state.json").write_text(
        json.dumps(
            {
                "cash_usd": cash,
                "positions": positions,
                "realized_pnl_usd": realized_pnl,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "trades": len(trades),
                "official_settlements": len(official_settlements),
                "open_trades": len(trades) - len(official_settlements),
                "cash_usd": cash,
                "realized_pnl_usd": realized_pnl,
                "resolutions": {
                    slug: (
                        {
                            "closed": resolution.closed,
                            "winning_side": resolution.winning_side,
                            "outcome_prices": resolution.outcome_prices,
                        }
                        if resolution
                        else None
                    )
                    for slug, resolution in resolutions.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
