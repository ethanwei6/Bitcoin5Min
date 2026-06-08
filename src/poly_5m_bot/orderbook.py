from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PolymarketConfig
from .http import get_json


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    tick_size: str
    min_order_size: float
    hash: str

    @property
    def best_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None


def _levels(raw: Any, reverse: bool) -> list[BookLevel]:
    if not isinstance(raw, list):
        return []
    levels = []
    for item in raw:
        try:
            levels.append(BookLevel(price=float(item["price"]), size=float(item["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(levels, key=lambda level: level.price, reverse=reverse)


class ClobOrderBookClient:
    def __init__(self, config: PolymarketConfig):
        self.config = config

    def get_book(self, token_id: str) -> OrderBook:
        raw = get_json(f"{self.config.clob_base_url}/book?token_id={token_id}", timeout=6.0)
        return OrderBook(
            token_id=token_id,
            bids=_levels(raw.get("bids"), reverse=True),
            asks=_levels(raw.get("asks"), reverse=False),
            tick_size=str(raw.get("tick_size") or "0.01"),
            min_order_size=float(raw.get("min_order_size") or 0.0),
            hash=str(raw.get("hash") or ""),
        )

    def get_market_info(self, condition_id: str) -> dict[str, Any]:
        raw = get_json(f"{self.config.clob_base_url}/clob-markets/{condition_id}", timeout=6.0)
        return raw if isinstance(raw, dict) else {}
