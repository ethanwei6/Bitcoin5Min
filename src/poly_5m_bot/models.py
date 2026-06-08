from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

from .orderbook import OrderBook


@dataclass(frozen=True)
class PriceObservation:
    timestamp: float
    market_start_epoch: int
    market_end_epoch: int
    spot_price: float


@dataclass(frozen=True)
class ModelForecast:
    name: str
    p_up: float
    expected_end_price: float
    confidence: float
    reason: str

    @property
    def vote(self) -> str:
        return "UP" if self.p_up >= 0.5 else "DOWN"


@dataclass(frozen=True)
class EnsembleForecast:
    p_up: float
    expected_end_price: float
    confidence: float
    forecasts: list[ModelForecast]
    majority_side: str
    majority_count: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def ols_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def distribution_forecast(
    name: str,
    observation: PriceObservation,
    start_price: float,
    mean_log_return: float,
    variance_log_return: float,
    reason: str,
) -> ModelForecast:
    sigma = math.sqrt(max(variance_log_return, 1e-12))
    threshold = math.log(start_price / observation.spot_price)
    z = (threshold - mean_log_return) / sigma
    p_up = clamp(1.0 - normal_cdf(z), 0.01, 0.99)
    expected = observation.spot_price * math.exp(
        mean_log_return + 0.5 * variance_log_return
    )
    return ModelForecast(
        name=name,
        p_up=p_up,
        expected_end_price=expected,
        confidence=abs(p_up - 0.5) * 2,
        reason=reason,
    )


class RollingPriceWindow:
    def __init__(self, maxlen: int = 900, max_start_capture_lag_seconds: int = 10):
        self.observations: deque[PriceObservation] = deque(maxlen=maxlen)
        self.market_start_prices: dict[int, float] = {}
        self.market_start_capture_times: dict[int, float] = {}
        self.max_start_capture_lag_seconds = max_start_capture_lag_seconds

    def append(self, observation: PriceObservation) -> None:
        self.observations.append(observation)
        lag = observation.timestamp - observation.market_start_epoch
        if (
            observation.market_start_epoch not in self.market_start_prices
            and lag <= self.max_start_capture_lag_seconds
        ):
            self.market_start_prices[observation.market_start_epoch] = observation.spot_price
            self.market_start_capture_times[observation.market_start_epoch] = observation.timestamp

    def current_market_start_price(self, market_start_epoch: int) -> float | None:
        return self.market_start_prices.get(market_start_epoch)

    def recent(self, seconds: float) -> list[PriceObservation]:
        if not self.observations:
            return []
        cutoff = self.observations[-1].timestamp - seconds
        return [obs for obs in self.observations if obs.timestamp >= cutoff]

    def log_returns(self, seconds: float) -> list[float]:
        return [item[1] for item in self.log_return_points(seconds)]

    def log_return_points(self, seconds: float) -> list[tuple[float, float]]:
        obs = self.recent(seconds)
        returns: list[tuple[float, float]] = []
        for left, right in zip(obs, obs[1:]):
            if left.spot_price > 0 and right.spot_price > 0:
                dt = max(right.timestamp - left.timestamp, 1e-6)
                returns.append((dt, math.log(right.spot_price / left.spot_price)))
        return returns

    def realized_vol_per_second(self, seconds: float = 180.0) -> float:
        returns = [ret / math.sqrt(dt) for dt, ret in self.log_return_points(seconds)]
        if len(returns) < 3:
            return 0.00035
        return max(statistics.pstdev(returns), 0.00001)

    def mean_return_per_second(self, seconds: float = 180.0) -> float:
        points = self.log_return_points(seconds)
        total_time = sum(dt for dt, _ret in points)
        if total_time <= 0:
            return 0.0
        return sum(ret for _dt, ret in points) / total_time

    def realized_variance(self, seconds: float) -> float:
        return sum(ret * ret for _dt, ret in self.log_return_points(seconds))


class ForecastModel:
    name = "base"

    def forecast(
        self,
        window: RollingPriceWindow,
        observation: PriceObservation,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> ModelForecast | None:
        raise NotImplementedError


class DistanceToStartModel(ForecastModel):
    name = "distance_to_start_random_walk"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        vol = window.realized_vol_per_second()
        sigma_price = observation.spot_price * vol * math.sqrt(seconds_left)
        z = (observation.spot_price - start_price) / max(sigma_price, 1e-9)
        p_up = clamp(logistic(1.7 * z), 0.01, 0.99)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=observation.spot_price,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"spot distance from interval start z={z:.3f}",
        )


class ShortMomentumModel(ForecastModel):
    name = "short_momentum"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        recent = window.recent(45.0)
        if len(recent) < 4:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        log_ret = math.log(observation.spot_price / recent[0].spot_price)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = log_ret / max(observation.timestamp - recent[0].timestamp, 1.0)
        expected = observation.spot_price * math.exp(drift * min(seconds_left, 90.0))
        vol = window.realized_vol_per_second()
        sigma_price = observation.spot_price * vol * math.sqrt(seconds_left)
        z = (expected - start_price) / max(sigma_price, 1e-9)
        p_up = clamp(logistic(z), 0.01, 0.99)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"45s drift={drift:.8f}",
        )


class MeanReversionModel(ForecastModel):
    name = "mean_reversion"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        recent = window.recent(180.0)
        if len(recent) < 12:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        prices = [obs.spot_price for obs in recent]
        mean_price = statistics.fmean(prices)
        vol = max(statistics.pstdev(prices), 1e-9)
        stretch = (observation.spot_price - mean_price) / vol
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        reversion_strength = clamp(seconds_left / 300.0, 0.05, 0.5)
        expected = observation.spot_price - stretch * vol * reversion_strength
        sigma_price = observation.spot_price * window.realized_vol_per_second() * math.sqrt(seconds_left)
        z = (expected - start_price) / max(sigma_price, 1e-9)
        p_up = clamp(logistic(z), 0.01, 0.99)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"rolling stretch={stretch:.3f}",
        )


class VolatilityFadeModel(ForecastModel):
    name = "volatility_fade"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        returns = window.log_returns(120.0)
        if len(returns) < 8:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        latest_move = sum(returns[-4:])
        vol = max(statistics.pstdev(returns), 1e-9)
        expected = observation.spot_price * math.exp(-0.35 * latest_move)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        sigma_price = observation.spot_price * vol * math.sqrt(seconds_left)
        z = (expected - start_price) / max(sigma_price, 1e-9)
        p_up = clamp(logistic(z), 0.01, 0.99)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"fade latest log move={latest_move:.7f}",
        )


class EwmaVolatilityModel(ForecastModel):
    name = "ewma_riskmetrics_volatility"

    def __init__(self, decay: float = 0.94):
        self.decay = decay

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(600.0)
        if len(points) < 12:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        variance = statistics.pvariance(scaled_returns) or 1e-8
        for eps in scaled_returns:
            variance = self.decay * variance + (1.0 - self.decay) * eps * eps
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.35 * window.mean_return_per_second(180.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            variance * seconds_left,
            f"EWMA lambda={self.decay:.2f}, variance_per_second={variance:.10f}",
        )


class GarchVolatilityModel(ForecastModel):
    name = "garch_1_1"

    def __init__(self, alpha: float = 0.08, beta: float = 0.90):
        self.alpha = alpha
        self.beta = beta

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(900.0)
        if len(points) < 20:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        unconditional = max(statistics.pvariance(scaled_returns), 1e-10)
        omega = max(unconditional * (1.0 - self.alpha - self.beta), 1e-12)
        variance = unconditional
        for eps in scaled_returns:
            variance = omega + self.alpha * eps * eps + self.beta * variance
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.25 * window.mean_return_per_second(240.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            variance * seconds_left,
            f"GARCH(1,1) alpha={self.alpha:.2f}, beta={self.beta:.2f}, variance_per_second={variance:.10f}",
        )


class ThresholdGarchModel(ForecastModel):
    name = "gjr_threshold_garch"

    def __init__(self, alpha: float = 0.05, gamma: float = 0.08, beta: float = 0.88):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(900.0)
        if len(points) < 20:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        unconditional = max(statistics.pvariance(scaled_returns), 1e-10)
        persistence = self.alpha + 0.5 * self.gamma + self.beta
        omega = max(unconditional * (1.0 - persistence), 1e-12)
        variance = unconditional
        for eps in scaled_returns:
            asymmetry = self.gamma * eps * eps if eps < 0 else 0.0
            variance = omega + self.alpha * eps * eps + asymmetry + self.beta * variance
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.20 * window.mean_return_per_second(240.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            variance * seconds_left,
            f"GJR-style variance_per_second={variance:.10f}",
        )


class HarRealizedVolatilityModel(ForecastModel):
    name = "har_realized_volatility"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        if len(window.log_return_points(900.0)) < 30:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        rv_short = window.realized_variance(60.0) / 60.0
        rv_medium = window.realized_variance(300.0) / 300.0
        rv_long = window.realized_variance(900.0) / 900.0
        variance_per_second = max(0.50 * rv_short + 0.30 * rv_medium + 0.20 * rv_long, 1e-10)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.30 * window.mean_return_per_second(300.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            variance_per_second * seconds_left,
            f"HAR RV short={rv_short:.10f}, medium={rv_medium:.10f}, long={rv_long:.10f}",
        )


class MertonJumpDiffusionModel(ForecastModel):
    name = "merton_jump_diffusion"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(900.0)
        if len(points) < 30:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        total_time = sum(dt for dt, _ret in points)
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        robust_sigma = max(statistics.pstdev(scaled_returns), 1e-8)
        jump_points = [
            (dt, ret)
            for dt, ret in points
            if abs(ret / math.sqrt(dt)) > 2.75 * robust_sigma
        ]
        continuous_points = [item for item in points if item not in jump_points]
        continuous_time = max(sum(dt for dt, _ret in continuous_points), 1e-6)
        continuous_mean = sum(ret for _dt, ret in continuous_points) / continuous_time
        continuous_var = max(
            statistics.pvariance([ret / math.sqrt(dt) for dt, ret in continuous_points])
            if len(continuous_points) >= 3
            else robust_sigma * robust_sigma,
            1e-10,
        )
        jump_lambda = len(jump_points) / max(total_time, 1e-6)
        jump_returns = [ret for _dt, ret in jump_points]
        jump_mean = statistics.fmean(jump_returns) if jump_returns else 0.0
        jump_var = statistics.pvariance(jump_returns) if len(jump_returns) >= 2 else 0.0
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        mean = (continuous_mean + jump_lambda * jump_mean) * seconds_left
        variance = (continuous_var + jump_lambda * (jump_var + jump_mean * jump_mean)) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            mean,
            variance,
            f"lambda={jump_lambda:.5f}/s, jump_mean={jump_mean:.7f}, jumps={len(jump_points)}",
        )


class KalmanLocalTrendModel(ForecastModel):
    name = "kalman_local_trend"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        recent = window.recent(300.0)
        if len(recent) < 20:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        times = [obs.timestamp - recent[0].timestamp for obs in recent]
        log_prices = [math.log(obs.spot_price) for obs in recent]
        trend = ols_slope(times, log_prices)
        residuals = [
            log_price - (log_prices[0] + trend * t)
            for t, log_price in zip(times, log_prices)
        ]
        residual_var = max(statistics.pvariance(residuals), 1e-10)
        trend = clamp(trend, -0.001, 0.001)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        mean = trend * min(seconds_left, 120.0)
        variance = residual_var + window.realized_vol_per_second(300.0) ** 2 * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            mean,
            variance,
            f"local log-price trend={trend:.8f}/s",
        )


class OrderBookImbalanceModel(ForecastModel):
    name = "polymarket_orderbook_imbalance"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        if up_book is None or down_book is None or up_book.best_bid is None or down_book.best_bid is None:
            return None
        up_bid_depth = sum(level.size for level in up_book.bids[:3])
        down_bid_depth = sum(level.size for level in down_book.bids[:3])
        total = up_bid_depth + down_bid_depth
        if total <= 0:
            return None
        imbalance = (up_bid_depth - down_bid_depth) / total
        p_up = clamp(0.5 + 0.18 * imbalance, 0.05, 0.95)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=observation.spot_price,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"top-three bid depth imbalance={imbalance:.3f}",
        )


class Ensemble:
    def __init__(self, models: list[ForecastModel] | None = None):
        self.models = models or [
            DistanceToStartModel(),
            ShortMomentumModel(),
            MeanReversionModel(),
            VolatilityFadeModel(),
            EwmaVolatilityModel(),
            GarchVolatilityModel(),
            ThresholdGarchModel(),
            HarRealizedVolatilityModel(),
            MertonJumpDiffusionModel(),
            KalmanLocalTrendModel(),
            OrderBookImbalanceModel(),
        ]

    def forecast(
        self,
        window: RollingPriceWindow,
        observation: PriceObservation,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> EnsembleForecast | None:
        forecasts = [
            forecast
            for model in self.models
            if (forecast := model.forecast(window, observation, up_book, down_book)) is not None
        ]
        if not forecasts:
            return None
        p_up = statistics.fmean(item.p_up for item in forecasts)
        expected = statistics.fmean(item.expected_end_price for item in forecasts)
        votes = [item.vote for item in forecasts]
        up_votes = votes.count("UP")
        down_votes = votes.count("DOWN")
        majority_side = "UP" if up_votes >= down_votes else "DOWN"
        majority_count = max(up_votes, down_votes)
        return EnsembleForecast(
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            forecasts=forecasts,
            majority_side=majority_side,
            majority_count=majority_count,
        )
