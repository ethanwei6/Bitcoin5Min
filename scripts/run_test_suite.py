#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import test_bot_drain
import test_execution
import test_market
import test_models
import test_risk


def main() -> None:
    test_risk.test_fee_per_share_matches_protocol_shape()
    test_risk.test_kelly_fraction_positive_only_when_probability_beats_cost()
    test_risk.test_risk_engine_trades_when_majority_edge_and_caps_pass()
    test_risk.test_risk_engine_rejects_after_daily_drawdown_limit()
    test_risk.test_risk_engine_rejects_expensive_contracts()
    test_execution.test_execution_uses_refreshed_book_and_records_latency()
    test_execution.test_fok_rejects_when_refreshed_book_cannot_fill_inside_worst_price()
    test_models.test_window_only_captures_interval_start_when_seen_early()
    test_models.test_garch_model_emits_bounded_forecast()
    test_models.test_merton_jump_diffusion_model_emits_bounded_forecast()
    test_models.test_default_ensemble_has_multiple_econometric_forecasts()
    test_market.test_market_from_event_uses_slug_epoch_as_interval_start()
    test_bot_drain.test_bounded_run_drains_open_positions_without_new_trade_ticks(
        Path("work/test-drain")
    )
    print("manual pytest-style tests passed")


if __name__ == "__main__":
    main()
