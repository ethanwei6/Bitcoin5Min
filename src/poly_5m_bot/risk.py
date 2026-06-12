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
    raw_probability: float = 0.0
    probability_haircut: float = 0.0
    kelly_scale: float = 1.0
    trade_cohort: str = "none"


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
            if forecast.total_weight > 0.0:
                if forecast.majority_weight <= forecast.total_weight / 2.0:
                    return self._reject("NONE", "no strict weighted model majority", forecast.p_up)
            else:
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
        if (
            self.config.max_daily_loss_usd > 0.0
            and daily_pnl_usd <= -abs(self.config.max_daily_loss_usd)
        ):
            return self._reject("NONE", "daily loss limit reached", forecast.p_up)
        if (
            self.config.max_daily_drawdown_usd > 0.0
            and daily_drawdown_usd >= abs(self.config.max_daily_drawdown_usd)
        ):
            return self._reject("NONE", "daily drawdown limit reached", forecast.p_up)
        if (
            self.config.max_consecutive_losses > 0
            and consecutive_losses >= self.config.max_consecutive_losses
        ):
            return self._reject("NONE", "consecutive loss limit reached", forecast.p_up)
        if (
            self.config.max_entries_per_market > 0
            and market_entries >= self.config.max_entries_per_market
        ):
            return self._reject("NONE", "market entry limit reached", forecast.p_up)

        side = forecast.majority_side
        win_probability = forecast.p_up if side == "UP" else 1.0 - forecast.p_up
        if win_probability < self.config.min_confidence:
            return self._reject(side, "ensemble confidence below threshold", win_probability)

        book = up_book if side == "UP" else down_book
        if book.best_ask is None:
            return self._reject(side, "no executable ask", win_probability)

        executable_price = book.best_ask.price
        if (
            self.config.min_contract_entry_price > 0.0
            and executable_price < self.config.min_contract_entry_price
        ):
            return self._reject(side, "contract price below risk floor", win_probability)
        if (
            self.config.max_contract_entry_price < 1.0
            and executable_price > self.config.max_contract_entry_price
        ):
            return self._reject(side, "contract price above risk cap", win_probability)
        trade_cohort = self._trade_cohort(
            side=side,
            win_probability=win_probability,
            executable_price=executable_price,
            forecast=forecast,
            seconds_to_end=seconds_to_end,
        )
        probability_haircut = self._probability_haircut(trade_cohort)
        probability_haircut += self._same_market_reentry_haircut(market_entries)
        adjusted_probability = max(0.01, win_probability - probability_haircut)
        effective_cost = executable_price + taker_fee_per_share(executable_price, self.config.fee_rate)
        edge = adjusted_probability - effective_cost
        kelly_scale = self._kelly_scale(trade_cohort) * self._same_market_reentry_kelly_scale(
            market_entries
        )
        if edge < self.config.min_edge:
            return TradeDecision(
                should_trade=False,
                side=side,
                probability=adjusted_probability,
                executable_price=executable_price,
                effective_cost=effective_cost,
                edge=edge,
                shares=0.0,
                spend_usd=0.0,
                kelly_fraction_full=0.0,
                reason="edge below threshold",
                raw_probability=win_probability,
                probability_haircut=probability_haircut,
                kelly_scale=kelly_scale,
                trade_cohort=trade_cohort,
            )
        if self.config.max_edge < 1.0 and edge > self.config.max_edge:
            return TradeDecision(
                should_trade=False,
                side=side,
                probability=adjusted_probability,
                executable_price=executable_price,
                effective_cost=effective_cost,
                edge=edge,
                shares=0.0,
                spend_usd=0.0,
                kelly_fraction_full=0.0,
                reason="edge above dislocation cap",
                raw_probability=win_probability,
                probability_haircut=probability_haircut,
                kelly_scale=kelly_scale,
                trade_cohort=trade_cohort,
            )

        kelly_full = full_kelly_fraction(adjusted_probability, effective_cost)
        account_base_usd = cash_usd + market_exposure_usd
        target_market_exposure = (
            account_base_usd
            * kelly_full
            * self.config.kelly_fraction
            * kelly_scale
        )
        desired_spend = max(0.0, target_market_exposure - market_exposure_usd)
        top_level_capacity = book.best_ask.size * effective_cost
        spend_limits = [desired_spend, top_level_capacity, cash_usd]
        if self.config.max_trade_usd > 0.0:
            spend_limits.append(self.config.max_trade_usd)
        if self.config.max_position_usd_per_market > 0.0:
            spend_limits.append(
                max(0.0, self.config.max_position_usd_per_market - market_exposure_usd)
            )
        spend = min(spend_limits)
        if spend <= 0.0:
            return TradeDecision(
                should_trade=False,
                side=side,
                probability=adjusted_probability,
                executable_price=executable_price,
                effective_cost=effective_cost,
                edge=edge,
                shares=0.0,
                spend_usd=0.0,
                kelly_fraction_full=kelly_full,
                reason="Kelly target exposure already reached",
                raw_probability=win_probability,
                probability_haircut=probability_haircut,
                kelly_scale=kelly_scale,
                trade_cohort=trade_cohort,
            )
        shares = spend / effective_cost
        return TradeDecision(
            should_trade=True,
            side=side,
            probability=adjusted_probability,
            executable_price=executable_price,
            effective_cost=effective_cost,
            edge=edge,
            shares=shares,
            spend_usd=spend,
            kelly_fraction_full=kelly_full,
            reason="trade eligible",
            raw_probability=win_probability,
            probability_haircut=probability_haircut,
            kelly_scale=kelly_scale,
            trade_cohort=trade_cohort,
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
        if self.config.min_contract_entry_price > 0.0 and fill_price < self.config.min_contract_entry_price:
            return self._execution_reject("fill price below risk floor", effective_cost, edge)
        if self.config.max_contract_entry_price < 1.0 and fill_price > self.config.max_contract_entry_price:
            return self._execution_reject("fill price above risk cap", effective_cost, edge)
        if edge < self.config.min_edge:
            return self._execution_reject("executed edge below threshold", effective_cost, edge)
        if self.config.max_edge < 1.0 and edge > self.config.max_edge:
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
            raw_probability=probability,
        )

    @staticmethod
    def _execution_reject(reason: str, effective_cost: float, edge: float) -> dict[str, float | bool | str]:
        return {
            "accepted": False,
            "reason": reason,
            "effective_cost": effective_cost,
            "edge": edge,
        }

    def _trade_cohort(
        self,
        *,
        side: str,
        win_probability: float,
        executable_price: float,
        forecast: EnsembleForecast,
        seconds_to_end: float,
    ) -> str:
        if side not in {"UP", "DOWN"}:
            return "none"
        if win_probability >= 0.5:
            return "directional_confidence"
        if executable_price >= 0.5:
            return "low_confidence_full_price"

        distance = forecast.spot_distance_from_start
        rebound = (side == "UP" and distance < 0.0) or (side == "DOWN" and distance > 0.0)
        base = "underdog_rebound" if rebound else "underdog_continuation"
        if seconds_to_end <= self.config.late_underdog_seconds:
            return f"late_{base}"
        return base

    def _probability_haircut(self, trade_cohort: str) -> float:
        haircut = 0.0
        if "underdog" in trade_cohort:
            haircut += max(0.0, self.config.underdog_probability_haircut)
        if "rebound" in trade_cohort:
            haircut += max(0.0, self.config.rebound_probability_haircut)
        if trade_cohort.startswith("late_") and "underdog" in trade_cohort:
            haircut += max(0.0, self.config.late_underdog_probability_haircut)
        return haircut

    def _kelly_scale(self, trade_cohort: str) -> float:
        scale = 1.0
        if "underdog" in trade_cohort:
            scale *= max(0.0, self.config.underdog_kelly_scale)
        if "rebound" in trade_cohort:
            scale *= max(0.0, self.config.rebound_kelly_scale)
        if trade_cohort.startswith("late_") and "underdog" in trade_cohort:
            scale *= max(0.0, self.config.late_underdog_kelly_scale)
        return scale

    def _same_market_reentry_haircut(self, market_entries: int) -> float:
        return max(0.0, self.config.same_market_reentry_probability_haircut) * max(
            0, market_entries
        )

    def _same_market_reentry_kelly_scale(self, market_entries: int) -> float:
        decay = min(1.0, max(0.0, self.config.same_market_reentry_kelly_decay))
        return decay ** max(0, market_entries)
