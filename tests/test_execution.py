from __future__ import annotations

from poly_5m_bot.execution import PaperExecutionSimulator
from poly_5m_bot.orderbook import BookLevel, OrderBook
from poly_5m_bot.risk import TradeDecision


def make_book(asks: list[tuple[float, float]], hash_value: str = "hash") -> OrderBook:
    return OrderBook(
        token_id="token",
        bids=[BookLevel(price=0.49, size=100.0)],
        asks=[BookLevel(price=price, size=size) for price, size in asks],
        tick_size="0.01",
        min_order_size=5.0,
        hash=hash_value,
    )


def make_decision() -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        side="UP",
        probability=0.65,
        executable_price=0.50,
        effective_cost=0.5175,
        edge=0.1325,
        shares=10.0,
        spend_usd=5.175,
        kelly_fraction_full=0.25,
        reason="trade eligible",
    )


def test_execution_uses_refreshed_book_and_records_latency() -> None:
    simulator = PaperExecutionSimulator(
        enabled=True,
        order_type="FOK",
        max_slippage_ticks=1,
        min_fill_ratio=0.999,
        fee_rate=0.07,
    )
    result = simulator.simulate_buy(
        decision=make_decision(),
        token_id="token",
        book_at_signal=make_book([(0.50, 10.0)], "signal"),
        fetch_current_book=lambda _token: make_book([(0.51, 20.0)], "submit"),
    )

    assert result.accepted
    assert result.fill_price == 0.51
    assert result.fill_shares == 10.0
    assert result.book_hash_at_submit == "submit"
    assert result.latency_ms >= 0.0


def test_fok_rejects_when_refreshed_book_cannot_fill_inside_worst_price() -> None:
    simulator = PaperExecutionSimulator(
        enabled=True,
        order_type="FOK",
        max_slippage_ticks=1,
        min_fill_ratio=0.999,
        fee_rate=0.07,
    )
    result = simulator.simulate_buy(
        decision=make_decision(),
        token_id="token",
        book_at_signal=make_book([(0.50, 10.0)], "signal"),
        fetch_current_book=lambda _token: make_book([(0.52, 20.0)], "submit"),
    )

    assert not result.accepted
    assert result.reason == "fok_would_not_fill_after_book_recheck"
    assert result.fill_shares == 0.0
