from __future__ import annotations

from pathlib import Path

from poly_5m_bot.config import load_config


def test_default_config_is_value_seeking_fractional_kelly_research_mode() -> None:
    config = load_config(Path("config/paper_btc_5m.json"))

    assert config.kelly_fraction == 0.35
    assert config.market_prior_weight == 0.55
    assert config.disagreement_shrink == 0.65
    assert config.horizon_confidence_min_multiplier == 0.35
    assert config.horizon_confidence_power == 0.65
    assert config.underdog_probability_haircut == 0.003
    assert config.rebound_probability_haircut == 0.0
    assert config.late_underdog_probability_haircut == 0.002
    assert config.underdog_continuation_probability_haircut == 0.024
    assert config.underdog_kelly_scale == 1.0
    assert config.underdog_continuation_kelly_scale == 0.55
    assert config.rebound_kelly_scale == 1.05
    assert config.late_underdog_kelly_scale == 0.95
    assert config.same_market_reentry_probability_haircut == 0.002
    assert config.same_market_reentry_kelly_decay == 0.90
    assert config.same_market_opposite_side_probability_haircut == 0.008
    assert config.same_market_opposite_side_kelly_decay == 0.70
    assert config.same_market_late_reentry_probability_haircut == 0.004
    assert config.same_market_late_reentry_kelly_decay == 0.85
    assert config.model_weights["short_momentum"] == 0.38
    assert config.model_weights["mean_reversion"] == 0.96
    assert config.model_weights["volatility_fade"] == 0.77
    assert config.model_weights["merton_jump_diffusion"] == 1.40
    assert config.model_weights["empirical_interval_knn"] == 1.00
    assert config.model_weights["gjr_threshold_garch"] == 1.52
    assert config.min_edge == 0.0
    assert config.min_confidence == 0.0
    assert config.max_trade_usd == 0.0
    assert config.max_position_usd_per_market == 0.0
    assert config.max_daily_loss_usd == 0.0
    assert config.max_daily_drawdown_usd == 0.0
    assert config.max_consecutive_losses == 0
    assert config.max_entries_per_market == 0
    assert config.max_contract_entry_price == 1.0
