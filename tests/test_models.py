from __future__ import annotations

import math

from poly_5m_bot.models import (
    Ensemble,
    GarchVolatilityModel,
    MertonJumpDiffusionModel,
    PriceObservation,
    RollingPriceWindow,
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


def test_garch_model_emits_bounded_forecast() -> None:
    window, observation = populated_window()
    forecast = GarchVolatilityModel().forecast(window, observation, None, None)
    assert forecast is not None
    assert 0.01 <= forecast.p_up <= 0.99
    assert forecast.expected_end_price > 0


def test_merton_jump_diffusion_model_emits_bounded_forecast() -> None:
    window, observation = populated_window()
    forecast = MertonJumpDiffusionModel().forecast(window, observation, None, None)
    assert forecast is not None
    assert 0.01 <= forecast.p_up <= 0.99
    assert "jumps=" in forecast.reason


def test_default_ensemble_has_multiple_econometric_forecasts() -> None:
    window, observation = populated_window()
    forecast = Ensemble().forecast(window, observation, None, None)
    assert forecast is not None
    names = {item.name for item in forecast.forecasts}
    assert "garch_1_1" in names
    assert "merton_jump_diffusion" in names
    assert "har_realized_volatility" in names
