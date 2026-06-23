from __future__ import annotations

import json
import time
from pathlib import Path

from poly_5m_bot.bot import PaperTradingBot
from poly_5m_bot.config import BotConfig, PolymarketConfig
from poly_5m_bot.paper import PaperPosition


def make_config(output_dir: Path) -> BotConfig:
    return BotConfig(
        asset="BTC",
        market_interval_seconds=300,
        sample_interval_seconds=0,
        decision_cutoff_seconds_before_end=20,
        paper_starting_cash_usd=1000.0,
        trade_only_if_majority=True,
        min_models_required=3,
        min_edge=0.03,
        min_confidence=0.55,
        kelly_fraction=0.25,
        max_trade_usd=25.0,
        max_position_usd_per_market=50.0,
        max_daily_loss_usd=100.0,
        max_daily_drawdown_usd=100.0,
        max_consecutive_losses=6,
        max_entries_per_market=2,
        min_contract_entry_price=0.0,
        max_contract_entry_price=0.60,
        max_edge=0.20,
        fee_rate=0.07,
        min_seconds_after_market_start=15,
        max_seconds_after_market_start=210,
        max_start_capture_lag_seconds=10,
        drain_before_stop_seconds=360,
        max_spot_source_spread_usd=150.0,
        model_weights={},
        market_prior_weight=0.25,
        disagreement_shrink=0.25,
        horizon_confidence_min_multiplier=0.25,
        horizon_confidence_power=0.65,
        simulate_execution_latency=True,
        execution_order_type="FOK",
        execution_max_slippage_ticks=1,
        execution_min_fill_ratio=0.999,
        output_dir=output_dir,
        polymarket=PolymarketConfig(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            event_slug_template="btc-updown-5m-{start_epoch}",
            slug_search_radius=4,
        ),
        spot_sources=[],
    )


def test_bounded_run_drains_open_positions_without_new_trade_ticks(tmp_path: Path) -> None:
    bot = PaperTradingBot(make_config(tmp_path))
    bot.broker.state.positions = [
        PaperPosition(
            market_slug="btc-updown-5m-1",
            market_start_epoch=1,
            market_end_epoch=301,
            side="DOWN",
            shares=1.0,
            price=0.5,
            fee_usd=0.0,
            cost_usd=0.5,
            opened_at=1.0,
            settled=False,
        )
    ]
    settle_only_flags: list[bool] = []

    def fake_tick(settle_only: bool = False) -> None:
        settle_only_flags.append(settle_only)
        bot.broker.state.positions[0].settled = True

    bot.tick = fake_tick  # type: ignore[method-assign]
    bot.run_forever(duration_seconds=0, drain_before_stop_seconds=120)

    assert settle_only_flags == [True]
    assert not bot.broker.has_open_positions()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_daily_risk_stats_match_settlements_by_position_key_not_file_order(tmp_path: Path) -> None:
    bot = PaperTradingBot(make_config(tmp_path))
    opened_at = time.time()
    trades = [
        {
            "position": {
                "market_slug": "btc-updown-5m-a",
                "market_start_epoch": int(opened_at) - 30,
                "market_end_epoch": int(opened_at) + 270,
                "side": "UP",
                "shares": 10.0,
                "cost_usd": 5.0,
                "opened_at": opened_at,
            }
        },
        {
            "position": {
                "market_slug": "btc-updown-5m-b",
                "market_start_epoch": int(opened_at) - 30,
                "market_end_epoch": int(opened_at) + 270,
                "side": "DOWN",
                "shares": 14.0,
                "cost_usd": 7.0,
                "opened_at": opened_at + 1.0,
            }
        },
    ]
    settlements = [
        {
            "market_slug": "btc-updown-5m-b",
            "side": "DOWN",
            "shares": 14.0,
            "cost_usd": 7.0,
            "pnl_usd": -7.0,
        },
        {
            "market_slug": "btc-updown-5m-a",
            "side": "UP",
            "shares": 10.0,
            "cost_usd": 5.0,
            "pnl_usd": 5.0,
        },
    ]
    write_jsonl(tmp_path / "trades.jsonl", trades)
    write_jsonl(tmp_path / "official_settlements.jsonl", settlements)

    stats = bot._daily_risk_stats()

    assert stats.realized_pnl_usd == -2.0
    assert stats.drawdown_usd == 7.0
    assert stats.consecutive_losses == 1
