from __future__ import annotations

import math

from poly_5m_bot.models import (
    DistanceToStartModel,
    Ensemble,
    EmpiricalIntervalKnnModel,
    ForecastModel,
    GarchVolatilityModel,
    MertonJumpDiffusionModel,
    ModelForecast,
    PriceObservation,
    RegimeSwitchingVolatilityModel,
    RollingPriceWindow,
    StudentTGarchModel,
    VolatilityFadeModel,
)
from poly_5m_bot.orderbook import BookLevel, OrderBook


class StaticModel(ForecastModel):
    def __init__(self, name: str, p_up: float):
        self.name = name
        self.p_up = p_up

    def forecast(self, window, observation, up_book, down_book):
        return ModelForecast(
            name=self.name,
            p_up=self.p_up,
            expected_end_price=observation.spot_price,
            confidence=abs(self.p_up - 0.5) * 2,
            reason="static test forecast",
        )


def make_book(bid: float, ask: float) -> OrderBook:
    return OrderBook(
        token_id="token",
        bids=[BookLevel(price=bid, size=100.0)],
        asks=[BookLevel(price=ask, size=100.0)],
        tick_size="0.01",
        min_order_size=5.0,
        hash="hash",
    )


def test_window_only_captures_interval_start_when_seen_early() -> None:
    window = RollingPriceWindow(max_start_capture_lag_seconds=10)
    window.append(
        PriceObservation(
            timestamp=1030,
            market_start_epoch=1000,
            market_end_epoch=1300,
            spot_price=100.0,
        )
    )
    assert window.current_market_start_price(1000) is None

    window.append(
        PriceObservation(
            timestamp=1305,
            market_start_epoch=1300,
            market_end_epoch=1600,
            spot_price=101.0,
        )
    )
    assert window.current_market_start_price(1300) == 101.0


def populated_window() -> tuple[RollingPriceWindow, PriceObservation]:
    window = RollingPriceWindow(max_start_capture_lag_seconds=10)
    start = 1000.0
    price = 100.0
    latest = None
    for index in range(80):
        timestamp = start + index * 5
        price *= math.exp(0.00008 * math.sin(index / 4) + 0.00003)
        if index == 45:
            price *= math.exp(0.004)
        latest = PriceObservation(
            timestamp=timestamp,
            market_start_epoch=1000,
            market_end_epoch=1600,
            spot_price=price,
        )
        window.append(latest)
    assert latest is not None
    return window, latest


def minute_window() -> tuple[RollingPriceWindow, PriceObservation]:
    window = RollingPriceWindow(maxlen=360, max_start_capture_lag_seconds=65)
    price = 100.0
    latest = None
    start = 1000
    for index in range(180):
        timestamp = start + index * 60
        market_start = timestamp - (timestamp % 300)
        price *= math.exp(0.0009 * math.sin(index / 5) + 0.0003 * math.cos(index / 13))
        latest = PriceObservation(
            timestamp=timestamp,
            market_start_epoch=market_start,
            market_end_epoch=market_start + 300,
            spot_price=price,
        )
        window.append(latest)
    assert latest is not None
    return window, latest


def test_garch_model_emits_bounded_forecast() -> None:
    window, observation = populated_window()
    forecast = GarchVolatilityModel().forecast(window, observation, None, None)
    assert forecast is not None
    assert 0.01 <= forecast.p_up <= 0.99
    assert forecast.expected_end_price > 0


def test_basis_spread_reduces_near_threshold_confidence() -> None:
    low_spread_window = RollingPriceWindow(max_start_capture_lag_seconds=10)
    high_spread_window = RollingPriceWindow(max_start_capture_lag_seconds=10)
    low_start = PriceObservation(
        timestamp=1002,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
        source_spread_usd=0.0,
    )
    high_start = PriceObservation(
        timestamp=1002,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
        source_spread_usd=120.0,
    )
    low_spread_window.append(low_start)
    high_spread_window.append(high_start)
    low_observation = PriceObservation(
        timestamp=1120,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.08,
        source_spread_usd=0.0,
    )
    high_observation = PriceObservation(
        timestamp=1120,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.08,
        source_spread_usd=120.0,
    )

    low_forecast = DistanceToStartModel().forecast(
        low_spread_window,
        low_observation,
        None,
        None,
    )
    high_forecast = DistanceToStartModel().forecast(
        high_spread_window,
        high_observation,
        None,
        None,
    )

    assert low_forecast is not None
    assert high_forecast is not None
    assert abs(high_forecast.p_up - 0.5) < abs(low_forecast.p_up - 0.5)


def test_merton_jump_diffusion_model_emits_bounded_forecast() -> None:
    window, observation = populated_window()
    forecast = MertonJumpDiffusionModel().forecast(window, observation, None, None)
    assert forecast is not None
    assert 0.01 <= forecast.p_up <= 0.99
    assert "jumps=" in forecast.reason


def test_volatility_fade_uses_horizon_scaled_volatility() -> None:
    window, observation = populated_window()
    forecast = VolatilityFadeModel().forecast(window, observation, None, None)

    assert forecast is not None
    assert 0.01 <= forecast.p_up <= 0.99
    assert "vol_per_second=" in forecast.reason


def test_default_ensemble_has_multiple_econometric_forecasts() -> None:
    window, observation = populated_window()
    forecast = Ensemble().forecast(window, observation, None, None)
    assert forecast is not None
    names = {item.name for item in forecast.forecasts}
    assert "garch_1_1" in names
    assert "merton_jump_diffusion" in names
    assert "har_realized_volatility" in names
    assert "student_t_garch" in names
    assert "regime_switching_volatility" in names


def test_new_interval_models_emit_bounded_forecasts() -> None:
    window, observation = minute_window()
    for model in [
        StudentTGarchModel(),
        RegimeSwitchingVolatilityModel(),
        EmpiricalIntervalKnnModel(),
    ]:
        forecast = model.forecast(window, observation, None, None)
        assert forecast is not None, model.name
        assert 0.01 <= forecast.p_up <= 0.99
        assert forecast.expected_end_price > 0


def test_ensemble_shrinks_extreme_models_toward_market_prior() -> None:
    observation = PriceObservation(
        timestamp=1010,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("a", 0.92),
            StaticModel("b", 0.88),
            StaticModel("c", 0.84),
        ],
        market_prior_weight=0.40,
    )
    forecast = ensemble.forecast(
        RollingPriceWindow(),
        observation,
        make_book(0.49, 0.51),
        make_book(0.49, 0.51),
    )

    assert forecast is not None
    assert 0.50 < forecast.p_up < 0.88


def test_ensemble_calibrates_large_model_market_dislocation() -> None:
    observation = PriceObservation(
        timestamp=1010,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("a", 0.92),
            StaticModel("b", 0.89),
            StaticModel("c", 0.86),
            StaticModel("d", 0.84),
        ],
        market_prior_weight=0.35,
        disagreement_shrink=0.85,
    )
    forecast = ensemble.forecast(
        RollingPriceWindow(),
        observation,
        make_book(0.39, 0.41),
        make_book(0.59, 0.61),
    )

    assert forecast is not None
    assert forecast.p_up < 0.70
    assert forecast.market_prior_p_up is not None
    assert forecast.raw_p_up > 0.50
    assert forecast.market_prior_p_up < forecast.p_up < forecast.raw_p_up


def test_ensemble_shrinks_early_interval_confidence_by_horizon() -> None:
    early_observation = PriceObservation(
        timestamp=1015,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    late_observation = PriceObservation(
        timestamp=1210,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("a", 0.90),
            StaticModel("b", 0.88),
            StaticModel("c", 0.86),
            StaticModel("d", 0.84),
        ],
        market_prior_weight=0.35,
        disagreement_shrink=0.85,
        horizon_confidence_min_multiplier=0.25,
        horizon_confidence_power=0.65,
    )
    up_book = make_book(0.49, 0.51)
    down_book = make_book(0.49, 0.51)

    early = ensemble.forecast(RollingPriceWindow(), early_observation, up_book, down_book)
    late = ensemble.forecast(RollingPriceWindow(), late_observation, up_book, down_book)

    assert early is not None
    assert late is not None
    assert early.p_up > 0.50
    assert late.p_up > early.p_up
    assert early.horizon_confidence_multiplier < late.horizon_confidence_multiplier
    assert early.raw_p_up == late.raw_p_up
    assert early.market_prior_p_up == late.market_prior_p_up


def test_zero_weight_models_do_not_influence_probability_pool() -> None:
    observation = PriceObservation(
        timestamp=1100,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("disabled_overconfident_model", 0.99),
            StaticModel("short_momentum", 0.22),
            StaticModel("garch_1_1", 0.24),
            StaticModel("student_t_garch", 0.26),
        ],
        model_weights={
            "disabled_overconfident_model": 0.0,
            "short_momentum": 1.0,
            "garch_1_1": 1.0,
            "student_t_garch": 1.0,
        },
        market_prior_weight=0.0,
        disagreement_shrink=0.0,
        horizon_confidence_min_multiplier=1.0,
    )

    forecast = ensemble.forecast(RollingPriceWindow(), observation, None, None)

    assert forecast is not None
    assert forecast.p_up < 0.30
    assert forecast.total_weight == 3.0


def test_horizon_shrink_targets_market_prior_for_large_dislocations() -> None:
    early_observation = PriceObservation(
        timestamp=1015,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    late_observation = PriceObservation(
        timestamp=1210,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("short_momentum", 0.72),
            StaticModel("garch_1_1", 0.69),
            StaticModel("student_t_garch", 0.66),
            StaticModel("mean_reversion", 0.78),
        ],
        horizon_confidence_min_multiplier=0.25,
        horizon_confidence_power=0.65,
    )
    up_book = make_book(0.11, 0.13)
    down_book = make_book(0.87, 0.89)

    early = ensemble.forecast(RollingPriceWindow(), early_observation, up_book, down_book)
    late = ensemble.forecast(RollingPriceWindow(), late_observation, up_book, down_book)

    assert early is not None
    assert late is not None
    assert early.market_prior_p_up is not None
    assert early.p_up < late.p_up
    assert abs(early.p_up - early.market_prior_p_up) < abs(early.raw_p_up - early.market_prior_p_up)


def test_ensemble_calibrates_false_recovery_value_against_extreme_market_prior() -> None:
    observation = PriceObservation(
        timestamp=1120,
        market_start_epoch=1000,
        market_end_epoch=1300,
        spot_price=100.0,
    )
    ensemble = Ensemble(
        models=[
            StaticModel("short_momentum", 0.36),
            StaticModel("garch_1_1", 0.34),
            StaticModel("student_t_garch", 0.35),
            StaticModel("mean_reversion", 0.72),
            StaticModel("merton_jump_diffusion", 0.70),
        ],
        horizon_confidence_min_multiplier=1.0,
    )
    up_book = make_book(0.11, 0.13)
    down_book = make_book(0.87, 0.89)

    forecast = ensemble.forecast(RollingPriceWindow(), observation, up_book, down_book)

    assert forecast is not None
    assert forecast.directional_p_up < forecast.reversion_p_up
    assert forecast.market_dislocation_shrink > 0.0
    assert forecast.p_up < 0.35
