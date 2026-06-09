from __future__ import annotations

from pathlib import Path

from poly_5m_bot.config import BotConfig, PolymarketConfig
from poly_5m_bot.models import EnsembleForecast, ModelForecast
from poly_5m_bot.orderbook import BookLevel, OrderBook
from poly_5m_bot.risk import TradeDecision, RiskEngine, full_kelly_fraction, taker_fee_per_share


def make_config() -> BotConfig:
    return BotConfig(
        asset="BTC",
        market_interval_seconds=300,
        sample_interval_seconds=5,
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
        simulate_execution_latency=True,
        execution_order_type="FOK",
        execution_max_slippage_ticks=1,
        execution_min_fill_ratio=0.999,
        output_dir=Path("outputs/test"),
        polymarket=PolymarketConfig(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            event_slug_template="btc-updown-5m-{start_epoch}",
            slug_search_radius=4,
        ),
        spot_sources=[],
    )


def make_book(price: float, size: float = 100.0) -> OrderBook:
    return OrderBook(
        token_id="token",
        bids=[BookLevel(price=max(price - 0.02, 0.01), size=size)],
        asks=[BookLevel(price=price, size=size)],
        tick_size="0.01",
        min_order_size=5.0,
        hash="hash",
    )


def test_fee_per_share_matches_protocol_shape() -> None:
    assert round(taker_fee_per_share(0.5, 0.07), 5) == 0.0175
    assert round(taker_fee_per_share(0.3, 0.07), 5) == round(taker_fee_per_share(0.7, 0.07), 5)


def test_kelly_fraction_positive_only_when_probability_beats_cost() -> None:
    assert full_kelly_fraction(0.62, 0.52) > 0
    assert full_kelly_fraction(0.50, 0.52) == 0


def test_risk_engine_trades_when_majority_edge_and_caps_pass() -> None:
    forecasts = [
        ModelForecast("a", 0.66, 101.0, 0.2, "x"),
        ModelForecast("b", 0.64, 101.0, 0.2, "x"),
        ModelForecast("c", 0.48, 99.0, 0.1, "x"),
    ]
    ensemble = EnsembleForecast(
        p_up=0.63,
        expected_end_price=100.5,
        confidence=0.26,
        forecasts=forecasts,
        majority_side="UP",
        majority_count=2,
    )
    decision = RiskEngine(make_config()).decide(
        forecast=ensemble,
        up_book=make_book(0.50),
        down_book=make_book(0.50),
        cash_usd=1000.0,
        market_exposure_usd=0.0,
        market_entries=0,
        daily_pnl_usd=0.0,
        daily_drawdown_usd=0.0,
        consecutive_losses=0,
        seconds_from_start=30.0,
        seconds_to_end=120.0,
    )
    assert decision.should_trade
    assert decision.side == "UP"
    assert decision.spend_usd <= 25.0


def test_risk_engine_rejects_after_daily_drawdown_limit() -> None:
    forecasts = [
        ModelForecast("a", 0.66, 101.0, 0.2, "x"),
        ModelForecast("b", 0.64, 101.0, 0.2, "x"),
        ModelForecast("c", 0.62, 101.0, 0.2, "x"),
        ModelForecast("d", 0.61, 101.0, 0.2, "x"),
    ]
    ensemble = EnsembleForecast(
        p_up=0.63,
        expected_end_price=100.5,
        confidence=0.26,
        forecasts=forecasts,
        majority_side="UP",
        majority_count=4,
    )
    decision = RiskEngine(make_config()).decide(
        forecast=ensemble,
        up_book=make_book(0.50),
        down_book=make_book(0.50),
        cash_usd=1000.0,
        market_exposure_usd=0.0,
        market_entries=0,
        daily_pnl_usd=50.0,
        daily_drawdown_usd=100.0,
        consecutive_losses=0,
        seconds_from_start=30.0,
        seconds_to_end=120.0,
    )
    assert not decision.should_trade
    assert decision.reason == "daily drawdown limit reached"


def test_risk_engine_rejects_expensive_contracts() -> None:
    forecasts = [
        ModelForecast("a", 0.90, 101.0, 0.8, "x"),
        ModelForecast("b", 0.88, 101.0, 0.8, "x"),
        ModelForecast("c", 0.86, 101.0, 0.8, "x"),
        ModelForecast("d", 0.84, 101.0, 0.8, "x"),
    ]
    ensemble = EnsembleForecast(
        p_up=0.87,
        expected_end_price=100.5,
        confidence=0.74,
        forecasts=forecasts,
        majority_side="UP",
        majority_count=4,
    )
    decision = RiskEngine(make_config()).decide(
        forecast=ensemble,
        up_book=make_book(0.70),
        down_book=make_book(0.30),
        cash_usd=1000.0,
        market_exposure_usd=0.0,
        market_entries=0,
        daily_pnl_usd=0.0,
        daily_drawdown_usd=0.0,
        consecutive_losses=0,
        seconds_from_start=30.0,
        seconds_to_end=120.0,
    )
    assert not decision.should_trade
    assert decision.reason == "contract price above risk cap"


def test_risk_engine_rejects_late_contract_entries() -> None:
    forecasts = [
        ModelForecast("a", 0.66, 101.0, 0.2, "x"),
        ModelForecast("b", 0.64, 101.0, 0.2, "x"),
        ModelForecast("c", 0.62, 101.0, 0.2, "x"),
    ]
    ensemble = EnsembleForecast(
        p_up=0.64,
        expected_end_price=100.5,
        confidence=0.28,
        forecasts=forecasts,
        majority_side="UP",
        majority_count=3,
    )
    decision = RiskEngine(make_config()).decide(
        forecast=ensemble,
        up_book=make_book(0.50),
        down_book=make_book(0.50),
        cash_usd=1000.0,
        market_exposure_usd=0.0,
        market_entries=0,
        daily_pnl_usd=0.0,
        daily_drawdown_usd=0.0,
        consecutive_losses=0,
        seconds_from_start=240.0,
        seconds_to_end=60.0,
    )
    assert not decision.should_trade
    assert decision.reason == "too late after market start"


def test_risk_engine_rejects_overconfident_dislocations() -> None:
    forecasts = [
        ModelForecast("a", 0.78, 101.0, 0.5, "x"),
        ModelForecast("b", 0.76, 101.0, 0.5, "x"),
        ModelForecast("c", 0.74, 101.0, 0.5, "x"),
    ]
    ensemble = EnsembleForecast(
        p_up=0.76,
        expected_end_price=100.5,
        confidence=0.52,
        forecasts=forecasts,
        majority_side="UP",
        majority_count=3,
    )
    decision = RiskEngine(make_config()).decide(
        forecast=ensemble,
        up_book=make_book(0.50),
        down_book=make_book(0.50),
        cash_usd=1000.0,
        market_exposure_usd=0.0,
        market_entries=0,
        daily_pnl_usd=0.0,
        daily_drawdown_usd=0.0,
        consecutive_losses=0,
        seconds_from_start=60.0,
        seconds_to_end=180.0,
    )
    assert not decision.should_trade
    assert decision.reason == "edge above dislocation cap"


def test_execution_fill_must_still_clear_edge_threshold() -> None:
    decision = TradeDecision(
        should_trade=True,
        side="UP",
        probability=0.56,
        executable_price=0.50,
        effective_cost=0.5175,
        edge=0.0425,
        shares=10.0,
        spend_usd=5.175,
        kelly_fraction_full=0.1,
        reason="trade eligible",
    )
    check = RiskEngine(make_config()).check_execution_fill(
        decision,
        fill_price=0.52,
        fill_cost_usd=5.4,
        fill_shares=10.0,
    )
    assert not check["accepted"]
    assert check["reason"] == "executed edge below threshold"
