from __future__ import annotations

import json

from poly_5m_bot.market import market_from_event


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
