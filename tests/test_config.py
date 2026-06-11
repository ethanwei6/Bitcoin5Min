from __future__ import annotations

from pathlib import Path

from poly_5m_bot.config import load_config


def test_default_config_is_model_first_fractional_kelly_research_mode() -> None:
    config = load_config(Path("config/paper_btc_5m.json"))

    assert config.kelly_fraction == 0.25
    assert config.market_prior_weight == 0.75
    assert config.disagreement_shrink == 0.85
    assert config.horizon_confidence_min_multiplier == 0.25
    assert config.horizon_confidence_power == 0.65
    assert config.min_edge == 0.0
    assert config.min_confidence == 0.0
    assert config.max_trade_usd == 0.0
    assert config.max_position_usd_per_market == 0.0
    assert config.max_daily_loss_usd == 0.0
    assert config.max_daily_drawdown_usd == 0.0
    assert config.max_consecutive_losses == 0
    assert config.max_entries_per_market == 0
    assert config.max_contract_entry_price == 1.0
