from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .journal import JsonlJournal
from .market import FiveMinuteMarket
from .risk import TradeDecision, taker_fee_per_share


@dataclass
class PaperPosition:
    market_slug: str
    market_start_epoch: int
    market_end_epoch: int
    side: str
    shares: float
    price: float
    fee_usd: float
    cost_usd: float
    opened_at: float
    start_price_proxy: float | None = None
    settled: bool = False


@dataclass
class PaperState:
    cash_usd: float
    realized_pnl_usd: float
    positions: list[PaperPosition]


class PaperBroker:
    def __init__(self, state_path: Path, starting_cash_usd: float, journal: JsonlJournal, fee_rate: float):
        self.state_path = state_path
        self.journal = journal
        self.fee_rate = fee_rate
        self.state = self._load(starting_cash_usd)

    def _load(self, starting_cash_usd: float) -> PaperState:
        if not self.state_path.exists():
            return PaperState(cash_usd=starting_cash_usd, realized_pnl_usd=0.0, positions=[])
        raw = json.loads(self.state_path.read_text())
        return PaperState(
            cash_usd=float(raw["cash_usd"]),
            realized_pnl_usd=float(raw.get("realized_pnl_usd", 0.0)),
            positions=[
                PaperPosition(
                    market_slug=item["market_slug"],
                    market_start_epoch=int(item["market_start_epoch"]),
                    market_end_epoch=int(item["market_end_epoch"]),
                    side=item["side"],
                    shares=float(item["shares"]),
                    price=float(item["price"]),
                    fee_usd=float(item["fee_usd"]),
                    cost_usd=float(item["cost_usd"]),
                    opened_at=float(item["opened_at"]),
                    start_price_proxy=(
                        float(item["start_price_proxy"])
                        if item.get("start_price_proxy") is not None
                        else None
                    ),
                    settled=bool(item.get("settled", False)),
                )
                for item in raw.get("positions", [])
            ],
        )

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash_usd": self.state.cash_usd,
            "realized_pnl_usd": self.state.realized_pnl_usd,
            "positions": [asdict(position) for position in self.state.positions],
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def market_exposure(self, market_slug: str) -> float:
        return sum(
            position.cost_usd
            for position in self.state.positions
            if position.market_slug == market_slug and not position.settled
        )

    def has_open_positions(self) -> bool:
        return any(not position.settled for position in self.state.positions)

    def open_position(
        self,
        market: FiveMinuteMarket,
        decision: TradeDecision,
        start_price_proxy: float | None,
        execution: dict | None = None,
    ) -> PaperPosition | None:
        if not decision.should_trade or decision.shares <= 0:
            return None
        fill_price = decision.executable_price
        fill_shares = decision.shares
        fee = decision.shares * taker_fee_per_share(decision.executable_price, self.fee_rate)
        cost = decision.spend_usd
        if execution is not None:
            fill_price = float(execution["fill_price"])
            fill_shares = float(execution["fill_shares"])
            fee = float(execution["fill_fee_usd"])
            cost = float(execution["fill_cost_usd"])
        if fill_shares <= 0.0:
            return None
        if cost > self.state.cash_usd:
            return None
        position = PaperPosition(
            market_slug=market.slug,
            market_start_epoch=market.start_epoch,
            market_end_epoch=market.end_epoch,
            side=decision.side,
            shares=fill_shares,
            price=fill_price,
            fee_usd=fee,
            cost_usd=cost,
            opened_at=time.time(),
            start_price_proxy=start_price_proxy,
        )
        self.state.cash_usd -= cost
        self.state.positions.append(position)
        self.save()
        payload = {"type": "open", "position": asdict(position), "decision": decision}
        if execution is not None:
            payload["execution"] = execution
        self.journal.append("trades.jsonl", payload)
        return position

    def settle_due_positions(self, spot_by_market_start: dict[int, tuple[float, float]], now: float | None = None) -> None:
        timestamp = time.time() if now is None else now
        for position in self.state.positions:
            if position.settled or timestamp < position.market_end_epoch:
                continue
            price_pair = spot_by_market_start.get(position.market_start_epoch)
            if price_pair is None:
                continue
            start_price, end_price = price_pair
            if position.start_price_proxy is not None:
                start_price = position.start_price_proxy
            winning_side = "UP" if end_price >= start_price else "DOWN"
            payout = position.shares if position.side == winning_side else 0.0
            pnl = payout - position.cost_usd
            self.state.cash_usd += payout
            self.state.realized_pnl_usd += pnl
            position.settled = True
            self.journal.append(
                "proxy_settlements.jsonl",
                {
                    "market_slug": position.market_slug,
                    "side": position.side,
                    "winning_side": winning_side,
                    "shares": position.shares,
                    "cost_usd": position.cost_usd,
                    "payout_usd": payout,
                    "pnl_usd": pnl,
                    "start_price_proxy": start_price,
                    "end_price_proxy": end_price,
                    "settlement_source": "spot_proxy_until_gamma_resolution_adapter",
                },
            )
        self.save()

    def settle_due_positions_official(
        self,
        official_outcomes: dict[str, tuple[str, str]],
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        for position in self.state.positions:
            if position.settled or timestamp < position.market_end_epoch:
                continue
            resolution = official_outcomes.get(position.market_slug)
            if resolution is None:
                continue
            winning_side, source = resolution
            payout = position.shares if position.side == winning_side else 0.0
            pnl = payout - position.cost_usd
            self.state.cash_usd += payout
            self.state.realized_pnl_usd += pnl
            position.settled = True
            self.journal.append(
                "official_settlements.jsonl",
                {
                    "market_slug": position.market_slug,
                    "side": position.side,
                    "winning_side": winning_side,
                    "shares": position.shares,
                    "cost_usd": position.cost_usd,
                    "payout_usd": payout,
                    "pnl_usd": pnl,
                    "settlement_source": "gamma_official_outcome",
                    "resolution_source": source,
                },
            )
        self.save()
