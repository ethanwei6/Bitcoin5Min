from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import PolymarketConfig
from .http import HttpError, get_json


@dataclass(frozen=True)
class TokenPair:
    up: str
    down: str


@dataclass(frozen=True)
class FiveMinuteMarket:
    event_id: str
    market_id: str
    condition_id: str
    slug: str
    question: str
    start_epoch: int
    end_epoch: int
    tokens: TokenPair
    minimum_tick_size: str
    neg_risk: bool

    @property
    def seconds_to_end(self) -> float:
        return self.end_epoch - time.time()

    @property
    def seconds_from_start(self) -> float:
        return time.time() - self.start_epoch


def interval_start(now: float, interval_seconds: int) -> int:
    return int(now // interval_seconds * interval_seconds)


def parse_epoch(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    normalized = text.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(normalized).timestamp())
    except ValueError:
        return None


def parse_jsonish_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def select_binary_market(event: dict[str, Any]) -> dict[str, Any] | None:
    markets = event.get("markets")
    if isinstance(markets, list) and markets:
        for market in markets:
            outcomes = [str(item).lower() for item in parse_jsonish_list(market.get("outcomes"))]
            if "up" in outcomes and "down" in outcomes:
                return market
        return markets[0]
    if "clobTokenIds" in event or "clob_token_ids" in event:
        return event
    return None


def market_from_event(event: dict[str, Any], start_epoch: int, interval_seconds: int) -> FiveMinuteMarket | None:
    market = select_binary_market(event)
    if market is None:
        return None
    outcomes = [str(item).lower() for item in parse_jsonish_list(market.get("outcomes"))]
    token_ids = parse_jsonish_list(
        market.get("clobTokenIds")
        or market.get("clob_token_ids")
        or market.get("clobTokenIDs")
    )
    if len(token_ids) < 2 or len(outcomes) < 2:
        return None
    try:
        up_index = outcomes.index("up")
        down_index = outcomes.index("down")
    except ValueError:
        return None

    # For rolling 5m crypto events, Gamma's startDate can represent when the
    # event opened for trading, not the interval boundary. The slug epoch is the
    # interval start and is the reliable timing anchor.
    start = start_epoch
    parsed_end = parse_epoch(market.get("endDate")) or parse_epoch(event.get("endDate"))
    if parsed_end is not None and abs(parsed_end - (start + interval_seconds)) <= 2:
        end = parsed_end
    else:
        end = start + interval_seconds
    return FiveMinuteMarket(
        event_id=str(event.get("id") or event.get("eventId") or ""),
        market_id=str(market.get("id") or ""),
        condition_id=str(market.get("conditionId") or market.get("condition_id") or ""),
        slug=str(event.get("slug") or market.get("slug") or ""),
        question=str(market.get("question") or event.get("title") or event.get("question") or ""),
        start_epoch=start,
        end_epoch=end,
        tokens=TokenPair(up=str(token_ids[up_index]), down=str(token_ids[down_index])),
        minimum_tick_size=str(market.get("minimum_tick_size") or market.get("minimumTickSize") or "0.01"),
        neg_risk=bool(market.get("neg_risk") or market.get("negRisk") or False),
    )


class PolymarketDiscovery:
    def __init__(self, config: PolymarketConfig, interval_seconds: int):
        self.config = config
        self.interval_seconds = interval_seconds
        self._cached_market: FiveMinuteMarket | None = None

    def current_market(self, now: float | None = None) -> FiveMinuteMarket | None:
        timestamp = time.time() if now is None else now
        if (
            self._cached_market is not None
            and self._cached_market.start_epoch <= timestamp < self._cached_market.end_epoch
        ):
            return self._cached_market
        base_start = interval_start(timestamp, self.interval_seconds)
        starts = [
            base_start + offset * self.interval_seconds
            for offset in range(-self.config.slug_search_radius, self.config.slug_search_radius + 1)
        ]
        candidates: list[FiveMinuteMarket] = []
        for start in starts:
            slug = self.config.event_slug_template.format(start_epoch=start)
            url = f"{self.config.gamma_base_url}/events/slug/{slug}"
            try:
                event = get_json(url, timeout=6.0)
            except HttpError:
                continue
            market = market_from_event(event, start, self.interval_seconds)
            if market and market.start_epoch <= timestamp < market.end_epoch:
                candidates.append(market)
        if not candidates:
            return None
        candidates.sort(key=lambda item: abs(item.start_epoch - base_start))
        self._cached_market = candidates[0]
        return self._cached_market
