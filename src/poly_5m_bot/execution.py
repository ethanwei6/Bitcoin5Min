from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

from .orderbook import BookLevel, OrderBook
from .risk import TradeDecision, taker_fee_per_share


@dataclass(frozen=True)
class ExecutionIntent:
    side: str
    token_id: str
    order_side: str
    order_type: str
    requested_shares: float
    requested_spend_usd: float
    reference_price: float
    max_price: float
    book_hash_at_signal: str
    created_at: float


@dataclass(frozen=True)
class ExecutionSimulation:
    accepted: bool
    reason: str
    intent: ExecutionIntent
    submitted_at: float
    refreshed_at: float
    latency_ms: float
    book_hash_at_submit: str
    fill_price: float
    fill_shares: float
    fill_cost_usd: float
    fill_fee_usd: float
    top_ask_at_signal: float | None
    top_ask_at_submit: float | None
    top_ask_size_at_submit: float | None
    consumed_levels: list[dict[str, float]]
    market_taker_delay_ms: float
    market_taker_delay_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _tick_size(book: OrderBook) -> float:
    try:
        return float(book.tick_size)
    except ValueError:
        return 0.01


def _buy_capacity(
    asks: list[BookLevel],
    max_price: float,
    target_shares: float,
    fee_rate: float,
) -> tuple[float, float, float, list[dict[str, float]]]:
    remaining = target_shares
    fill_shares = 0.0
    fill_trade_value = 0.0
    fill_fee = 0.0
    consumed = []
    for level in asks:
        if remaining <= 1e-12:
            break
        if level.price > max_price + 1e-12:
            break
        shares = min(remaining, level.size)
        fee = shares * taker_fee_per_share(level.price, fee_rate)
        fill_shares += shares
        fill_trade_value += shares * level.price
        fill_fee += fee
        remaining -= shares
        consumed.append(
            {
                "price": level.price,
                "shares": shares,
                "trade_value_usd": shares * level.price,
                "fee_usd": fee,
            }
        )
    avg_price = fill_trade_value / fill_shares if fill_shares > 0 else 0.0
    return avg_price, fill_shares, fill_fee, consumed


class PaperExecutionSimulator:
    """Paper-only CLOB execution model.

    This intentionally does not sign or post an order. It mirrors the real
    order path by creating a marketable BUY intent, making a fresh public CLOB
    book request, and filling against the refreshed asks within the configured
    worst-price limit.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        order_type: str,
        max_slippage_ticks: int,
        min_fill_ratio: float,
        fee_rate: float,
    ):
        self.enabled = enabled
        self.order_type = order_type.upper()
        self.max_slippage_ticks = max(0, max_slippage_ticks)
        self.min_fill_ratio = max(0.0, min(1.0, min_fill_ratio))
        self.fee_rate = fee_rate

    def simulate_buy(
        self,
        *,
        decision: TradeDecision,
        token_id: str,
        book_at_signal: OrderBook,
        fetch_current_book: Callable[[str], OrderBook],
        market_taker_delay_ms: float = 0.0,
        market_taker_delay_source: str = "not_checked",
    ) -> ExecutionSimulation:
        created_at = time.time()
        max_price = min(
            0.99,
            decision.executable_price + self.max_slippage_ticks * _tick_size(book_at_signal),
        )
        intent = ExecutionIntent(
            side=decision.side,
            token_id=token_id,
            order_side="BUY",
            order_type=self.order_type,
            requested_shares=decision.shares,
            requested_spend_usd=decision.spend_usd,
            reference_price=decision.executable_price,
            max_price=max_price,
            book_hash_at_signal=book_at_signal.hash,
            created_at=created_at,
        )
        if not self.enabled:
            fee = decision.shares * taker_fee_per_share(decision.executable_price, self.fee_rate)
            return ExecutionSimulation(
                accepted=True,
                reason="legacy_immediate_fill",
                intent=intent,
                submitted_at=created_at,
                refreshed_at=created_at,
                latency_ms=0.0,
                book_hash_at_submit=book_at_signal.hash,
                fill_price=decision.executable_price,
                fill_shares=decision.shares,
                fill_cost_usd=decision.spend_usd,
                fill_fee_usd=fee,
                top_ask_at_signal=book_at_signal.best_ask.price if book_at_signal.best_ask else None,
                top_ask_at_submit=book_at_signal.best_ask.price if book_at_signal.best_ask else None,
                top_ask_size_at_submit=book_at_signal.best_ask.size if book_at_signal.best_ask else None,
                consumed_levels=[
                    {
                        "price": decision.executable_price,
                        "shares": decision.shares,
                        "trade_value_usd": decision.shares * decision.executable_price,
                        "fee_usd": fee,
                    }
                ],
                market_taker_delay_ms=0.0,
                market_taker_delay_source="simulator_disabled",
            )

        submitted_at = time.time()
        if market_taker_delay_ms > 0:
            time.sleep(market_taker_delay_ms / 1000.0)
        current_book = fetch_current_book(token_id)
        refreshed_at = time.time()
        avg_price, shares, fee, consumed = _buy_capacity(
            current_book.asks,
            max_price,
            decision.shares,
            self.fee_rate,
        )
        fill_ratio = shares / decision.shares if decision.shares > 0 else 0.0
        accepted = shares > 0.0
        reason = "filled"
        if self.order_type == "FOK" and fill_ratio < self.min_fill_ratio:
            accepted = False
            reason = "fok_would_not_fill_after_book_recheck"
        elif self.order_type == "FAK" and fill_ratio <= 0.0:
            accepted = False
            reason = "fak_no_liquidity_after_book_recheck"
        elif current_book.best_ask is None:
            accepted = False
            reason = "no_ask_after_book_recheck"
        elif current_book.best_ask.price > max_price + 1e-12:
            accepted = False
            reason = "top_ask_exceeded_worst_price_after_book_recheck"

        if not accepted:
            avg_price = 0.0
            shares = 0.0
            fee = 0.0
            consumed = []

        trade_value = sum(item["trade_value_usd"] for item in consumed)
        return ExecutionSimulation(
            accepted=accepted,
            reason=reason,
            intent=intent,
            submitted_at=submitted_at,
            refreshed_at=refreshed_at,
            latency_ms=(refreshed_at - created_at) * 1000.0,
            book_hash_at_submit=current_book.hash,
            fill_price=avg_price,
            fill_shares=shares,
            fill_cost_usd=trade_value + fee,
            fill_fee_usd=fee,
            top_ask_at_signal=book_at_signal.best_ask.price if book_at_signal.best_ask else None,
            top_ask_at_submit=current_book.best_ask.price if current_book.best_ask else None,
            top_ask_size_at_submit=current_book.best_ask.size if current_book.best_ask else None,
            consumed_levels=consumed,
            market_taker_delay_ms=max(0.0, market_taker_delay_ms),
            market_taker_delay_source=market_taker_delay_source,
        )
