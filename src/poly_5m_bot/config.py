from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpotSourceConfig:
    name: str
    url: str


@dataclass(frozen=True)
class PolymarketConfig:
    gamma_base_url: str
    clob_base_url: str
    event_slug_template: str
    slug_search_radius: int


@dataclass(frozen=True)
class BotConfig:
    asset: str
    market_interval_seconds: int
    sample_interval_seconds: int
    decision_cutoff_seconds_before_end: int
    paper_starting_cash_usd: float
    trade_only_if_majority: bool
    min_models_required: int
    min_edge: float
    min_confidence: float
    kelly_fraction: float
    max_trade_usd: float
    max_position_usd_per_market: float
    max_daily_loss_usd: float
    max_daily_drawdown_usd: float
    max_consecutive_losses: int
    max_entries_per_market: int
    max_contract_entry_price: float
    fee_rate: float
    min_seconds_after_market_start: int
    max_start_capture_lag_seconds: int
    drain_before_stop_seconds: int
    max_spot_source_spread_usd: float
    simulate_execution_latency: bool
    execution_order_type: str
    execution_max_slippage_ticks: int
    execution_min_fill_ratio: float
    output_dir: Path
    polymarket: PolymarketConfig
    spot_sources: list[SpotSourceConfig]


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing config key: {key}")
    return data[key]


def _optional(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def load_config(path: str | Path) -> BotConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    raw = json.loads(config_path.read_text())
    pm = _require(raw, "polymarket")
    spot_sources = [
        SpotSourceConfig(name=item["name"], url=item["url"])
        for item in _require(raw, "spot_sources")
    ]
    output_dir = Path(_require(raw, "output_dir")).expanduser()
    if not output_dir.is_absolute():
        root = os.environ.get("POLY5M_ROOT")
        output_dir = (Path(root).expanduser() if root else Path.cwd()) / output_dir

    return BotConfig(
        asset=str(_require(raw, "asset")),
        market_interval_seconds=int(_require(raw, "market_interval_seconds")),
        sample_interval_seconds=int(_require(raw, "sample_interval_seconds")),
        decision_cutoff_seconds_before_end=int(
            _require(raw, "decision_cutoff_seconds_before_end")
        ),
        paper_starting_cash_usd=float(_require(raw, "paper_starting_cash_usd")),
        trade_only_if_majority=bool(_require(raw, "trade_only_if_majority")),
        min_models_required=int(_require(raw, "min_models_required")),
        min_edge=float(_require(raw, "min_edge")),
        min_confidence=float(_require(raw, "min_confidence")),
        kelly_fraction=float(_require(raw, "kelly_fraction")),
        max_trade_usd=float(_require(raw, "max_trade_usd")),
        max_position_usd_per_market=float(
            _require(raw, "max_position_usd_per_market")
        ),
        max_daily_loss_usd=float(_require(raw, "max_daily_loss_usd")),
        max_daily_drawdown_usd=float(_optional(raw, "max_daily_drawdown_usd", 100.0)),
        max_consecutive_losses=int(_optional(raw, "max_consecutive_losses", 6)),
        max_entries_per_market=int(_optional(raw, "max_entries_per_market", 2)),
        max_contract_entry_price=float(_optional(raw, "max_contract_entry_price", 0.60)),
        fee_rate=float(_require(raw, "fee_rate")),
        min_seconds_after_market_start=int(_require(raw, "min_seconds_after_market_start")),
        max_start_capture_lag_seconds=int(_require(raw, "max_start_capture_lag_seconds")),
        drain_before_stop_seconds=int(_require(raw, "drain_before_stop_seconds")),
        max_spot_source_spread_usd=float(_require(raw, "max_spot_source_spread_usd")),
        simulate_execution_latency=bool(_optional(raw, "simulate_execution_latency", True)),
        execution_order_type=str(_optional(raw, "execution_order_type", "FOK")).upper(),
        execution_max_slippage_ticks=int(_optional(raw, "execution_max_slippage_ticks", 1)),
        execution_min_fill_ratio=float(_optional(raw, "execution_min_fill_ratio", 0.999)),
        output_dir=output_dir,
        polymarket=PolymarketConfig(
            gamma_base_url=str(_require(pm, "gamma_base_url")).rstrip("/"),
            clob_base_url=str(_require(pm, "clob_base_url")).rstrip("/"),
            event_slug_template=str(_require(pm, "event_slug_template")),
            slug_search_radius=int(_require(pm, "slug_search_radius")),
        ),
        spot_sources=spot_sources,
    )
