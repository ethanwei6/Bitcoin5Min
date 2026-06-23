from __future__ import annotations

import json

from poly_5m_bot.config import PolymarketConfig
from poly_5m_bot.market import PolymarketDiscovery, market_from_event


def test_market_from_event_uses_slug_epoch_as_interval_start() -> None:
    event = {
        "id": "event",
        "slug": "btc-updown-5m-1780585500",
        "startDate": "2026-06-03T15:12:21Z",
        "endDate": "2026-06-04T15:10:00Z",
        "markets": [
            {
                "id": "market",
                "conditionId": "condition",
                "question": "Bitcoin Up or Down - June 4, 11:05AM-11:10AM ET",
                "outcomes": json.dumps(["Up", "Down"]),
                "clobTokenIds": json.dumps(["up-token", "down-token"]),
                "minimum_tick_size": "0.01",
            }
        ],
    }
    market = market_from_event(event, start_epoch=1780585500, interval_seconds=300)
    assert market is not None
    assert market.start_epoch == 1780585500
    assert market.end_epoch == 1780585800
    assert market.tokens.up == "up-token"
    assert market.tokens.down == "down-token"


def test_discovery_reuses_cached_market_inside_active_interval() -> None:
    import poly_5m_bot.market as market_module

    calls = []

    def fake_get_json(url: str, timeout: float = 6.0) -> dict[str, object]:
        calls.append(url)
        return {
            "id": "event",
            "slug": "btc-updown-5m-1000",
            "markets": [
                {
                    "id": "market",
                    "conditionId": "condition",
                    "question": "Bitcoin Up or Down",
                    "outcomes": json.dumps(["Up", "Down"]),
                    "clobTokenIds": json.dumps(["up-token", "down-token"]),
                }
            ],
        }

    original = market_module.get_json
    market_module.get_json = fake_get_json
    try:
        discovery = PolymarketDiscovery(
            PolymarketConfig(
                gamma_base_url="https://gamma.example",
                clob_base_url="https://clob.example",
                event_slug_template="btc-updown-5m-{start_epoch}",
                slug_search_radius=0,
            ),
            interval_seconds=300,
        )
        first = discovery.current_market(now=1010)
        second = discovery.current_market(now=1090)
    finally:
        market_module.get_json = original

    assert first is not None
    assert second is first
    assert len(calls) == 1
