#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import test_bot_drain
import test_backtest_models
import test_analyze_paper_run
import test_config
import test_execution
import test_evidence_report
import test_market
import test_models
import test_risk
import test_risk_counterfactual
import test_spot


def main() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    test_backtest_models.test_interval_ms_supports_second_and_minute_candles()
    test_backtest_models.test_train_test_split_is_chronological_before_asset_name()
    test_backtest_models.test_confidence_calibration_preserves_probability_side()
    test_backtest_models.test_model_rows_for_examples_uses_asset_timestamp_keys()
    test_backtest_models.test_model_rows_for_asset_keeps_only_requested_asset()
    test_backtest_models.test_candidate_weight_profiles_include_current_config_when_provided()
    test_backtest_models.test_weight_profile_reports_proxy_trade_buckets()
    test_backtest_models.test_data_quality_report_flags_minute_candles_as_research_only()
    test_backtest_models.test_data_quality_report_flags_requested_interval_mismatch()
    test_backtest_models.test_markdown_discloses_exchange_data_not_polymarket_replay()
    with TemporaryDirectory() as tmp_dir:
        test_analyze_paper_run.test_pair_trades_matches_official_settlements_by_position_key(
            Path(tmp_dir)
        )
    with TemporaryDirectory() as tmp_dir:
        test_risk_counterfactual.test_counterfactual_rejects_recalibrated_cheap_continuation_fill(
            Path(tmp_dir)
        )
    test_config.test_default_config_is_value_seeking_fractional_kelly_research_mode()
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
    test_risk.test_underdog_value_trade_gets_probability_haircut()
    test_risk.test_late_underdog_value_trade_is_scaled_not_hard_blocked()
    test_risk.test_continuation_underdog_is_softly_calibrated_separately_from_rebound()
    test_risk.test_same_market_reentry_is_softly_penalized_not_capped()
    test_risk.test_execution_fill_must_still_clear_edge_threshold()
    test_execution.test_execution_uses_refreshed_book_and_records_latency()
    test_execution.test_fok_rejects_when_refreshed_book_cannot_fill_inside_worst_price()
    test_spot.test_spot_snapshot_fetches_sources_concurrently()
    test_evidence_report.test_proxy_winners_use_all_observed_completed_markets()
    test_evidence_report.test_market_level_calibration_uses_one_latest_signal_per_market()
    test_evidence_report.test_suppressed_edge_diagnostics_show_raw_positive_edges_cut_by_calibration()
    test_evidence_report.test_data_freshness_reports_sampling_and_book_churn()
    test_models.test_window_only_captures_interval_start_when_seen_early()
    test_models.test_population_variance_matches_population_definition()
    test_models.test_garch_model_emits_bounded_forecast()
    test_models.test_basis_spread_reduces_near_threshold_confidence()
    test_models.test_merton_jump_diffusion_model_emits_bounded_forecast()
    test_models.test_volatility_fade_uses_horizon_scaled_volatility()
    test_models.test_default_ensemble_has_multiple_econometric_forecasts()
    test_models.test_new_interval_models_emit_bounded_forecasts()
    test_models.test_ensemble_shrinks_extreme_models_toward_market_prior()
    test_models.test_ensemble_calibrates_large_model_market_dislocation()
    test_models.test_zero_weight_models_do_not_influence_probability_pool()
    test_models.test_marketless_ensemble_keeps_model_probability_for_underlying_backtests()
    test_models.test_horizon_shrink_targets_market_prior_for_large_dislocations()
    test_models.test_ensemble_calibrates_false_recovery_value_against_extreme_market_prior()
    test_models.test_ensemble_shrinks_cheap_continuation_value_more_than_rebound_value()
    test_market.test_market_from_event_uses_slug_epoch_as_interval_start()
    test_market.test_discovery_reuses_cached_market_inside_active_interval()
    test_bot_drain.test_bounded_run_drains_open_positions_without_new_trade_ticks(Path("work/test-drain"))
    test_bot_drain.test_daily_risk_stats_match_settlements_by_position_key_not_file_order(Path("work/test-risk-stats"))
    print("manual pytest-style tests passed")


if __name__ == "__main__":
    main()
