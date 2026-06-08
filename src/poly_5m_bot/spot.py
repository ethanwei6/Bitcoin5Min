from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from .config import SpotSourceConfig
from .http import HttpError, get_json


@dataclass(frozen=True)
class SpotQuote:
    source: str
    price: float


@dataclass(frozen=True)
class SpotSnapshot:
    median_price: float
    quotes: list[SpotQuote]
    failed_sources: list[str]
    source_spread_usd: float
    warnings: list[str]


def _extract_price(source: str, payload: Any) -> float:
    if source == "coinbase":
        price = payload.get("price") or payload.get("best_bid")
        return float(price)
    if source == "binance_us":
        return float(payload["price"])
    if source == "kraken":
        pair_payload = next(iter(payload["result"].values()))
        return float(pair_payload["c"][0])
    if source == "gemini":
        return float(payload["last"])
    raise ValueError(f"Unknown spot source: {source}")


class SpotPriceClient:
    def __init__(self, sources: list[SpotSourceConfig], max_source_spread_usd: float):
        self.sources = sources
        self.max_source_spread_usd = max_source_spread_usd

    def snapshot(self) -> SpotSnapshot:
        quotes: list[SpotQuote] = []
        failed: list[str] = []
        for source in self.sources:
            try:
                payload = get_json(source.url, timeout=5.0)
                quotes.append(SpotQuote(source=source.name, price=_extract_price(source.name, payload)))
            except (HttpError, KeyError, TypeError, ValueError) as exc:
                failed.append(f"{source.name}: {exc}")
        if not quotes:
            raise RuntimeError("No spot sources returned a usable BTC price")
        prices = [quote.price for quote in quotes]
        source_spread = max(prices) - min(prices)
        warnings = []
        if source_spread > self.max_source_spread_usd:
            warnings.append(
                f"spot source spread ${source_spread:.2f} exceeds ${self.max_source_spread_usd:.2f}"
            )
        return SpotSnapshot(
            median_price=statistics.median(prices),
            quotes=quotes,
            failed_sources=failed,
            source_spread_usd=source_spread,
            warnings=warnings,
        )
