from __future__ import annotations

from dataclasses import dataclass

from .config import BotConfig
from .models import EnsembleForecast
from .orderbook import OrderBook


@dataclass(frozen=True)
class TradeDecision:
    should_trade: bool
    side: str
    probability: float
    executable_price: float
    effective_cost: float
    edge: float
    shares: float
    spend_usd: float
    kelly_fraction_full: float
    reason: str


def taker_fee_per_share(price: float, fee_rate: float) -> float:
    return fee_rate * price * (1.0 - price)


def full_kelly_fraction(win_probability: float, effective_cost: float) -> float:
    if effective_cost <= 0.0 or effective_cost >= 1.0:
        return 0.0
    return max(0.0, (win_probability - effective_cost) / (1.0 - effective_cost))


class RiskEngine:
    def __init__(self, config: BotConfig):
        self.config = config

    def decide(
        self,
        forecast: EnsembleForecast,
        up_book: OrderBook,
        down_book: OrderBook,
        cash_usd: float,
        market_exposure_usd: float,
        market_entries: int,
        daily_pnl_usd: float,
        daily_drawdown_usd: float,
        consecutive_losses: int,
        seconds_from_start: float,
        seconds_to_end: float,
    ) -> TradeDecision:
        if len(forecast.forecasts) < self.config.min_models_required:
            return self._reject("NONE", "not enough model forecasts", forecast.p_up)
        if self.config.trade_only_if_majority:
            strict_majority = len(forecast.forecasts) // 2 + 1
            if forecast.majority_count < strict_majority:
                return self._reject("NONE", "no strict model majority", forecast.p_up)
        if seconds_from_start < self.config.min_seconds_after_market_start:
            return self._reject("NONE", "too soon after market start", forecast.p_up)
        if (
            self.config.max_seconds_after_market_start > 0
            and seconds_from_start > self.config.max_seconds_after_market_start
        ):
            return self._reject("NONE", "too late after market start", forecast.p_up)
        if seconds_to_end < self.config.decision_cutoff_seconds_before_end:
            return self._reject("NONE", "too close to resolution", forecast.p_up)
        if daily_pnl_usd <= -abs(self.config.max_daily_loss_usd):
            return self._reject("NONE", "daily loss limit reached", forecast.p_up)
        if daily_drawdown_usd >= abs(self.config.max_daily_drawdown_usd):
            return self._reject("NONE", "daily drawdown limit reached", forecast.p_up)
        if consecutive_losses >= self.config.max_consecutive_losses:
            return self._reject("NONE", "consecutive loss limit reached", forecast.p_up)
        if market_entries >= self.config.max_entries_per_market:
            return self._reject("NONE", "market entry limit reached", forecast.p_up)

        side = forecast.majority_side
        win_probability = forecast.p_up if side == "UP" else 1.0 - forecast.p_up
        if win_probability < self.config.min_confidence:
            return self._reject(side, "ensemble confidence below threshold", win_probability)

        book = up_book if side == "UP" else down_book
        if book.best_ask is None:
            return self._reject(side, "no executable ask", win_probability)

        executable_price = book.best_ask.price
        if executable_price < self.config.min_contract_entry_price:
            return self._reject(side, "contract price below risk floor", win_probability)
        if executable_price > self.config.max_contract_entry_price:
            return self._reject(side, "contract price above risk cap", win_probability)
        effective_cost = executable_price + taker_fee_per_share(executable_price, self.config.fee_rate)
        edge = win_probability - effective_cost
        if edge < self.config.min_edge:
            return TradeDecision(
                should_trade=False,
                side=side,
                probability=win_probability,
                executable_price=executable_price,
                effective_cost=effective_cost,
                edge=edge,
                shares=0.0,
                spend_usd=0.0,
                kelly_fraction_full=0.0,
                reason="edge below threshold",
            )
        if edge > self.config.max_edge:
            return TradeDecision(
                should_trade=False,
                side=side,
                probability=win_probability,
                executable_price=executable_price,
                effective_cost=effective_cost,
                edge=edge,
                shares=0.0,
                spend_usd=0.0,
                kelly_fraction_full=0.0,
                reason="edge above dislocation cap",
            )

        kelly_full = full_kelly_fraction(win_probability, effective_cost)
        desired_spend = cash_usd * kelly_full * self.config.kelly_fraction
        allowed_exposure = max(0.0, self.config.max_position_usd_per_market - market_exposure_usd)
        top_level_capacity = book.best_ask.size * effective_cost
        spend = min(
            desired_spend,
            self.config.max_trade_usd,
            allowed_exposure,
            top_level_capacity,
            cash_usd,
        )
        if spend <= 0.0:
            return self._reject(side, "risk caps leave no spend capacity", win_probability)
        shares = spend / effective_cost
        return TradeDecision(
            should_trade=True,
            side=side,
            probability=win_probability,
            executable_price=executable_price,
            effective_cost=effective_cost,
            edge=edge,
            shares=shares,
            spend_usd=spend,
            kelly_fraction_full=kelly_full,
            reason="trade eligible",
        )

    def check_execution_fill(
        self,
        decision: TradeDecision,
        *,
        fill_price: float,
        fill_cost_usd: float,
        fill_shares: float,
    ) -> dict[str, float | bool | str]:
        if fill_shares <= 0.0:
            return self._execution_reject("no executed shares", 0.0, 0.0)
        effective_cost = fill_cost_usd / fill_shares
        edge = decision.probability - effective_cost
        if fill_price < self.config.min_contract_entry_price:
            return self._execution_reject("fill price below risk floor", effective_cost, edge)
        if fill_price > self.config.max_contract_entry_price:
            return self._execution_reject("fill price above risk cap", effective_cost, edge)
        if edge < self.config.min_edge:
            return self._execution_reject("executed edge below threshold", effective_cost, edge)
        if edge > self.config.max_edge:
            return self._execution_reject("executed edge above dislocation cap", effective_cost, edge)
        return {
            "accepted": True,
            "reason": "execution fill still clears risk",
            "effective_cost": effective_cost,
            "edge": edge,
        }

    @staticmethod
    def _reject(side: str, reason: str, probability: float) -> TradeDecision:
        return TradeDecision(
            should_trade=False,
            side=side,
            probability=probability,
            executable_price=0.0,
            effective_cost=0.0,
            edge=0.0,
            shares=0.0,
            spend_usd=0.0,
            kelly_fraction_full=0.0,
            reason=reason,
        )

    @staticmethod
    def _execution_reject(reason: str, effective_cost: float, edge: float) -> dict[str, float | bool | str]:
        return {
            "accepted": False,
            "reason": reason,
            "effective_cost": effective_cost,
            "edge": edge,
        }
