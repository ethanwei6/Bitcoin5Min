from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from test_risk import make_config


def load_counterfactual_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_risk_counterfactual.py"
    spec = importlib.util.spec_from_file_location("replay_risk_counterfactual", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_counterfactual_rejects_recalibrated_cheap_continuation_fill(tmp_path: Path) -> None:
    replay = load_counterfactual_module()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    fill_price = 0.30
    shares = 10.0
    fee = 0.07 * fill_price * (1.0 - fill_price)
    cost = shares * (fill_price + fee)
    trade = {
        "decision": {
            "executable_price": fill_price,
            "probability": 0.33,
            "raw_probability": 0.33,
            "edge": 0.0153,
            "should_trade": True,
            "side": "UP",
            "spend_usd": cost,
            "trade_cohort": "underdog_continuation",
        },
        "execution": {
            "accepted": True,
            "fill_cost_usd": cost,
            "fill_price": fill_price,
            "fill_shares": shares,
        },
        "position": {
            "cost_usd": cost,
            "market_end_epoch": 1300,
            "market_slug": "btc-updown-5m-1000",
            "market_start_epoch": 1000,
            "opened_at": 1100,
            "price": fill_price,
            "shares": shares,
            "side": "UP",
        },
    }
    settlement = {
        "cost_usd": cost,
        "market_slug": "btc-updown-5m-1000",
        "payout_usd": 0.0,
        "pnl_usd": -cost,
        "shares": shares,
        "side": "UP",
        "winning_side": "DOWN",
    }
    write_jsonl(output_dir / "trades.jsonl", [trade])
    write_jsonl(output_dir / "official_settlements.jsonl", [settlement])
    config = replace(
        make_config(),
        min_edge=0.0,
        min_confidence=0.0,
        max_contract_entry_price=1.0,
        max_seconds_after_market_start=0,
        max_entries_per_market=0,
        max_trade_usd=0.0,
        max_position_usd_per_market=0.0,
        underdog_probability_haircut=0.0,
        underdog_continuation_probability_haircut=0.08,
        underdog_kelly_scale=1.0,
        underdog_continuation_kelly_scale=1.0,
    )

    report = replay.replay(output_dir, config)

    assert report["summary"]["original_trades"] == 1
    assert report["summary"]["retained_trades"] == 0
    assert report["summary"]["pnl_delta_usd"] == cost
    assert report["by_reject_reason"] == {"edge below threshold": 1}
