from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_analyze_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_paper_run.py"
    spec = importlib.util.spec_from_file_location("analyze_paper_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_pair_trades_matches_official_settlements_by_position_key(tmp_path: Path) -> None:
    analyze = load_analyze_module()
    trades = [
        {
            "decision": {"probability": 0.55, "edge": 0.03},
            "execution": {"fill_cost_usd": 1.0, "fill_shares": 2.0},
            "position": {
                "cost_usd": 1.0,
                "market_slug": "market-a",
                "market_start_epoch": 1000,
                "opened_at": 1010,
                "price": 0.50,
                "shares": 2.0,
                "side": "UP",
            },
        },
        {
            "decision": {"probability": 0.55, "edge": 0.03},
            "execution": {"fill_cost_usd": 2.0, "fill_shares": 4.0},
            "position": {
                "cost_usd": 2.0,
                "market_slug": "market-b",
                "market_start_epoch": 1000,
                "opened_at": 1020,
                "price": 0.50,
                "shares": 4.0,
                "side": "DOWN",
            },
        },
    ]
    settlements = [
        {
            "cost_usd": 2.0,
            "market_slug": "market-b",
            "pnl_usd": -2.0,
            "shares": 4.0,
            "side": "DOWN",
            "winning_side": "UP",
        },
        {
            "cost_usd": 1.0,
            "market_slug": "market-a",
            "pnl_usd": 1.0,
            "shares": 2.0,
            "side": "UP",
            "winning_side": "UP",
        },
    ]
    write_jsonl(tmp_path / "trades.jsonl", trades)
    write_jsonl(tmp_path / "official_settlements.jsonl", settlements)

    rows = analyze.pair_trades(tmp_path)

    assert rows[0]["market_slug"] == "market-a"
    assert rows[0]["pnl_usd"] == 1.0
    assert rows[1]["market_slug"] == "market-b"
    assert rows[1]["pnl_usd"] == -2.0
