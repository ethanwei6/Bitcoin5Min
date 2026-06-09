from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
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
    majority_weight: float = 0.0
    total_weight: float = 0.0


@dataclass(frozen=True)
class IntervalSample:
    seconds_from_start: float
    partial_log_return: float
    final_log_return: float
    realized_vol_per_second: float


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


def student_t_cdf_approx(value: float, df: float) -> float:
    # Hill's normal approximation to Student-t keeps this dependency-free while
    # preserving the fatter tails needed for five-minute crypto jumps.
    if df <= 2:
        df = 2.1
    adjusted = value * math.sqrt(df / max(df + value * value, 1e-12))
    return normal_cdf(adjusted)


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


def book_mid(book: OrderBook | None) -> float | None:
    if book is None:
        return None
    bid = book.best_bid.price if book.best_bid is not None else None
    ask = book.best_ask.price if book.best_ask is not None else None
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return bid if bid is not None else ask


def market_implied_up_probability(
    up_book: OrderBook | None,
    down_book: OrderBook | None,
) -> float | None:
    up_mid = book_mid(up_book)
    down_mid = book_mid(down_book)
    if up_mid is None or down_mid is None or up_mid + down_mid <= 0:
        return None
    return clamp(up_mid / (up_mid + down_mid), 0.01, 0.99)


def trimmed_mean(values: list[float]) -> float:
    if len(values) <= 2:
        return statistics.fmean(values)
    ordered = sorted(values)
    return statistics.fmean(ordered[1:-1])


def completed_interval_samples(
    window: RollingPriceWindow,
    observation: PriceObservation,
) -> list[IntervalSample]:
    interval_seconds = observation.market_end_epoch - observation.market_start_epoch
    current_offset = observation.timestamp - observation.market_start_epoch
    all_rows = sorted(window.observations, key=lambda item: item.timestamp)
    timestamps = [item.timestamp for item in all_rows]
    samples = []
    candidate_starts = [
        start for start in sorted(window.market_start_prices) if start < observation.market_start_epoch
    ][-120:]
    for start_epoch in candidate_starts:
        if start_epoch >= observation.market_start_epoch:
            continue
        start_price = window.current_market_start_price(start_epoch)
        if start_price is None:
            continue
        final_index = bisect_left(timestamps, start_epoch + interval_seconds)
        if final_index >= len(all_rows):
            continue
        offset_target = start_epoch + current_offset
        partial_index = bisect_right(timestamps, offset_target) - 1
        if partial_index < 0 or all_rows[partial_index].timestamp < start_epoch:
            continue
        partial_row = all_rows[partial_index]
        final_price = all_rows[final_index].spot_price
        returns = []
        left_index = bisect_left(timestamps, start_epoch)
        right_index = bisect_right(timestamps, start_epoch + interval_seconds)
        interval_rows = all_rows[left_index:right_index]
        for left, right in zip(interval_rows, interval_rows[1:]):
            if right.timestamp > start_epoch + interval_seconds:
                break
            if left.spot_price > 0 and right.spot_price > 0:
                dt = max(right.timestamp - left.timestamp, 1e-6)
                returns.append(math.log(right.spot_price / left.spot_price) / math.sqrt(dt))
        vol = statistics.pstdev(returns) if len(returns) >= 3 else window.realized_vol_per_second()
        samples.append(
            IntervalSample(
                seconds_from_start=current_offset,
                partial_log_return=math.log(partial_row.spot_price / start_price),
                final_log_return=math.log(final_price / start_price),
                realized_vol_per_second=max(vol, 1e-8),
            )
        )
    return samples


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
        recent = window.recent(180.0)
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
            reason=f"180s drift={drift:.8f}",
        )


class MeanReversionModel(ForecastModel):
    name = "mean_reversion"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        recent = window.recent(900.0)
        if len(recent) < 8:
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
        returns = window.log_returns(300.0)
        if len(returns) < 4:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        latest_move = sum(returns[-3:])
        vol = max(statistics.pstdev(returns), 1e-9)
        expected = observation.spot_price * math.exp(-0.25 * latest_move)
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
        points = window.log_return_points(3600.0)
        if len(points) < 20:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        variance = statistics.pvariance(scaled_returns) or 1e-8
        for eps in scaled_returns:
            variance = self.decay * variance + (1.0 - self.decay) * eps * eps
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.25 * window.mean_return_per_second(900.0) * seconds_left
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
        points = window.log_return_points(7200.0)
        if len(points) < 30:
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
        drift = 0.20 * window.mean_return_per_second(900.0) * seconds_left
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
        points = window.log_return_points(7200.0)
        if len(points) < 30:
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
        drift = 0.15 * window.mean_return_per_second(900.0) * seconds_left
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
        if len(window.log_return_points(7200.0)) < 30:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        rv_short = window.realized_variance(300.0) / 300.0
        rv_medium = window.realized_variance(1800.0) / 1800.0
        rv_long = window.realized_variance(7200.0) / 7200.0
        variance_per_second = max(0.50 * rv_short + 0.30 * rv_medium + 0.20 * rv_long, 1e-10)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.20 * window.mean_return_per_second(900.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            variance_per_second * seconds_left,
            f"HAR RV short={rv_short:.10f}, medium={rv_medium:.10f}, long={rv_long:.10f}",
        )


class StudentTGarchModel(ForecastModel):
    name = "student_t_garch"

    def __init__(self, alpha: float = 0.08, beta: float = 0.90):
        self.alpha = alpha
        self.beta = beta

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(7200.0)
        if len(points) < 40:
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
        fourth = statistics.fmean((ret - statistics.fmean(scaled_returns)) ** 4 for ret in scaled_returns)
        kurtosis = fourth / max(unconditional * unconditional, 1e-12)
        df = clamp(6.0 / max(kurtosis - 3.0, 0.25) + 4.0, 4.0, 30.0)
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.12 * window.mean_return_per_second(900.0) * seconds_left
        sigma = math.sqrt(max(variance * seconds_left, 1e-12))
        threshold = math.log(start_price / observation.spot_price)
        z = (threshold - drift) / sigma
        p_up = clamp(1.0 - student_t_cdf_approx(z, df), 0.01, 0.99)
        expected = observation.spot_price * math.exp(drift + 0.5 * variance * seconds_left)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"Student-t GARCH df={df:.1f}, variance_per_second={variance:.10f}",
        )


class RegimeSwitchingVolatilityModel(ForecastModel):
    name = "regime_switching_volatility"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(10800.0)
        if len(points) < 50:
            return None
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        scaled_returns = [ret / math.sqrt(dt) for dt, ret in points]
        abs_returns = [abs(ret) for ret in scaled_returns]
        median_abs = statistics.median(abs_returns)
        high = [ret for ret in scaled_returns if abs(ret) >= median_abs]
        low = [ret for ret in scaled_returns if abs(ret) < median_abs]
        current_vol = window.realized_vol_per_second(900.0)
        low_var = statistics.pvariance(low) if len(low) >= 5 else statistics.pvariance(scaled_returns)
        high_var = statistics.pvariance(high) if len(high) >= 5 else statistics.pvariance(scaled_returns)
        high_prob = clamp((current_vol - math.sqrt(max(low_var, 1e-12))) / max(math.sqrt(max(high_var, 1e-12)) - math.sqrt(max(low_var, 1e-12)), 1e-8), 0.0, 1.0)
        variance = (1.0 - high_prob) * low_var + high_prob * high_var
        seconds_left = max(observation.market_end_epoch - observation.timestamp, 1.0)
        drift = 0.10 * window.mean_return_per_second(900.0) * seconds_left
        return distribution_forecast(
            self.name,
            observation,
            start_price,
            drift,
            max(variance, 1e-10) * seconds_left,
            f"two-regime vol high_prob={high_prob:.3f}",
        )


class EmpiricalIntervalKnnModel(ForecastModel):
    name = "empirical_interval_knn"

    def __init__(self, neighbors: int = 25):
        self.neighbors = neighbors

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        start_price = window.current_market_start_price(observation.market_start_epoch)
        if start_price is None:
            return None
        samples = completed_interval_samples(window, observation)
        if len(samples) < 12:
            return None
        current_partial = math.log(observation.spot_price / start_price)
        current_vol = window.realized_vol_per_second(900.0)
        ranked = sorted(
            samples,
            key=lambda sample: (
                abs(sample.partial_log_return - current_partial) / max(current_vol, 1e-8)
                + 0.25 * abs(sample.realized_vol_per_second - current_vol) / max(current_vol, 1e-8)
            ),
        )
        neighbors = ranked[: min(self.neighbors, len(ranked))]
        wins = sum(1 for sample in neighbors if sample.final_log_return >= 0.0)
        # Bayesian shrinkage prevents a tiny local neighborhood from producing
        # false 0/1 probabilities.
        p_up = clamp((wins + 3.0) / (len(neighbors) + 6.0), 0.01, 0.99)
        avg_final = statistics.fmean(sample.final_log_return for sample in neighbors)
        expected = start_price * math.exp(avg_final)
        return ModelForecast(
            name=self.name,
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            reason=f"KNN completed intervals n={len(neighbors)}",
        )


class MertonJumpDiffusionModel(ForecastModel):
    name = "merton_jump_diffusion"

    def forecast(self, window: RollingPriceWindow, observation: PriceObservation, up_book: OrderBook | None, down_book: OrderBook | None) -> ModelForecast | None:
        points = window.log_return_points(7200.0)
        if len(points) < 40:
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
        recent = window.recent(3600.0)
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
    default_model_weights = {
        "distance_to_start_random_walk": 0.75,
        "short_momentum": 0.35,
        "mean_reversion": 0.30,
        "volatility_fade": 0.20,
        "ewma_riskmetrics_volatility": 1.45,
        "garch_1_1": 1.55,
        "gjr_threshold_garch": 1.60,
        "har_realized_volatility": 1.55,
        "student_t_garch": 1.50,
        "regime_switching_volatility": 1.35,
        "empirical_interval_knn": 0.35,
        "merton_jump_diffusion": 1.10,
        "kalman_local_trend": 0.55,
        "polymarket_orderbook_imbalance": 0.25,
    }

    def __init__(
        self,
        models: list[ForecastModel] | None = None,
        model_weights: dict[str, float] | None = None,
        market_prior_weight: float = 0.25,
        disagreement_shrink: float = 0.25,
    ):
        self.models = models or [
            DistanceToStartModel(),
            ShortMomentumModel(),
            MeanReversionModel(),
            VolatilityFadeModel(),
            EwmaVolatilityModel(),
            GarchVolatilityModel(),
            ThresholdGarchModel(),
            HarRealizedVolatilityModel(),
            StudentTGarchModel(),
            RegimeSwitchingVolatilityModel(),
            EmpiricalIntervalKnnModel(),
            MertonJumpDiffusionModel(),
            KalmanLocalTrendModel(),
            OrderBookImbalanceModel(),
        ]
        self.model_weights = model_weights or self.default_model_weights
        self.market_prior_weight = clamp(market_prior_weight, 0.0, 1.0)
        self.disagreement_shrink = max(0.0, disagreement_shrink)

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
        weights = [max(self.model_weights.get(item.name, 1.0), 0.01) for item in forecasts]
        weight_total = sum(weights)
        model_p_up = sum(weight * item.p_up for weight, item in zip(weights, forecasts))
        model_p_up /= weight_total
        model_probabilities = [item.p_up for item in forecasts]
        robust_p_up = (
            0.90 * model_p_up
            + 0.05 * statistics.median(model_probabilities)
            + 0.05 * trimmed_mean(model_probabilities)
        )
        dispersion = statistics.pstdev(model_probabilities) if len(model_probabilities) > 1 else 0.0
        market_prior = market_implied_up_probability(up_book, down_book)
        if market_prior is not None:
            prior_weight = clamp(self.market_prior_weight + 0.50 * dispersion, 0.0, 0.55)
            anchored_p_up = (1.0 - prior_weight) * robust_p_up + prior_weight * market_prior
        else:
            anchored_p_up = robust_p_up
        shrink = clamp(dispersion * self.disagreement_shrink, 0.0, 0.10)
        p_up = clamp(0.5 + (anchored_p_up - 0.5) * (1.0 - shrink), 0.01, 0.99)
        expected = sum(
            weight * item.expected_end_price for weight, item in zip(weights, forecasts)
        ) / weight_total
        up_weight = sum(weight for weight, item in zip(weights, forecasts) if item.vote == "UP")
        down_weight = sum(weight for weight, item in zip(weights, forecasts) if item.vote == "DOWN")
        majority_side = "UP" if up_weight >= down_weight else "DOWN"
        majority_count = sum(
            1
            for item in forecasts
            if item.vote == majority_side
        )
        majority_weight = max(up_weight, down_weight)
        return EnsembleForecast(
            p_up=p_up,
            expected_end_price=expected,
            confidence=abs(p_up - 0.5) * 2,
            forecasts=forecasts,
            majority_side=majority_side,
            majority_count=majority_count,
            majority_weight=majority_weight,
            total_weight=weight_total,
        )
