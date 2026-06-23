from __future__ import annotations

import time

from poly_5m_bot.config import SpotSourceConfig
from poly_5m_bot.spot import SpotPriceClient


def test_spot_snapshot_fetches_sources_concurrently() -> None:
    import poly_5m_bot.spot as spot_module

    payloads = {
        "coinbase": {"price": "100.0"},
        "binance_us": {"price": "101.0"},
        "kraken": {"result": {"XXBTZUSD": {"c": ["99.0"]}}},
        "gemini": {"last": "100.5"},
    }

    def fake_get_json(url: str, timeout: float = 5.0):
        time.sleep(0.05)
        return payloads[url]

    original = spot_module.get_json
    spot_module.get_json = fake_get_json
    try:
        client = SpotPriceClient(
            [
                SpotSourceConfig(name="coinbase", url="coinbase"),
                SpotSourceConfig(name="binance_us", url="binance_us"),
                SpotSourceConfig(name="kraken", url="kraken"),
                SpotSourceConfig(name="gemini", url="gemini"),
            ],
            max_source_spread_usd=10.0,
        )
        started = time.perf_counter()
        snapshot = client.snapshot()
        elapsed = time.perf_counter() - started
    finally:
        spot_module.get_json = original

    assert len(snapshot.quotes) == 4
    assert snapshot.median_price == 100.25
    assert elapsed < 0.16
