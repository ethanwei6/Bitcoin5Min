#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import test_bot_drain
import test_backtest_models
import test_config
import test_execution
import test_market
import test_models
import test_risk


def main() -> None:
    test_backtest_models.test_interval_ms_supports_second_and_minute_candles()
    test_backtest_models.test_train_test_split_is_chronological_before_asset_name()
    test_backtest_models.test_confidence_calibration_preserves_probability_side()
    test_config.test_default_config_is_model_first_fractional_kelly_research_mode()
    test_risk.test_fee_per_share_matches_protocol_shape()
    test_risk.test_kelly_fraction_positive_only_when_probability_beats_cost()
    test_risk.test_risk_engine_trades_when_majority_edge_and_caps_pass()
    test_risk.test_risk_engine_uses_weighted_majority_for_side()
    test_risk.test_zero_market_entry_and_position_caps_disable_hard_limits()
    test_risk.test_fractional_kelly_targets_total_market_exposure_not_each_tick()
    test_risk.test_risk_engine_rejects_after_daily_drawdown_limit()
    test_risk.test_risk_engine_rejects_expensive_contracts()
    test_risk.test_risk_engine_rejects_late_contract_entries()
    test_risk.test_risk_engine_rejects_overconfident_dislocations()
    test_risk.test_execution_fill_must_still_clear_edge_threshold()
    test_execution.test_execution_uses_refreshed_book_and_records_latency()
    test_execution.test_fok_rejects_when_refreshed_book_cannot_fill_inside_worst_price()
    test_models.test_window_only_captures_interval_start_when_seen_early()
    test_models.test_garch_model_emits_bounded_forecast()
    test_models.test_merton_jump_diffusion_model_emits_bounded_forecast()
    test_models.test_default_ensemble_has_multiple_econometric_forecasts()
    test_models.test_new_interval_models_emit_bounded_forecasts()
    test_models.test_ensemble_shrinks_extreme_models_toward_market_prior()
    test_models.test_ensemble_calibrates_large_model_market_dislocation()
    test_market.test_market_from_event_uses_slug_epoch_as_interval_start()
    test_bot_drain.test_bounded_run_drains_open_positions_without_new_trade_ticks(
        Path("work/test-drain")
    )
    print("manual pytest-style tests passed")


if __name__ == "__main__":
    main()
