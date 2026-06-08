from __future__ import annotations

import json
from dataclasses import dataclass

from .config import PolymarketConfig
from .http import HttpError, get_json


@dataclass(frozen=True)
class OfficialResolution:
    market_slug: str
    closed: bool
    winning_side: str | None
    outcome_prices: list[float]
    source: str


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


class GammaResolutionClient:
    def __init__(self, config: PolymarketConfig):
        self.config = config

    def get_resolution(self, market_slug: str) -> OfficialResolution | None:
        try:
            event = get_json(f"{self.config.gamma_base_url}/events/slug/{market_slug}", timeout=6.0)
        except HttpError:
            return None
        markets = event.get("markets") or []
        if not markets:
            return None
        market = markets[0]
        outcomes = [str(item).upper() for item in _json_list(market.get("outcomes"))]
        prices = []
        for item in _json_list(market.get("outcomePrices")):
            try:
                prices.append(float(item))
            except (TypeError, ValueError):
                continue
        closed = bool(market.get("closed"))
        winning_side = None
        if closed and outcomes and prices and len(outcomes) == len(prices):
            max_price = max(prices)
            if max_price >= 0.99:
                winning_side = outcomes[prices.index(max_price)]
        return OfficialResolution(
            market_slug=market_slug,
            closed=closed,
            winning_side=winning_side,
            outcome_prices=prices,
            source=str(market.get("resolutionSource") or event.get("resolutionSource") or "gamma"),
        )
