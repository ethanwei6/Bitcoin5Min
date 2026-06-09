#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poly_5m_bot.models import Ensemble, PriceObservation, RollingPriceWindow
from poly_5m_bot.risk import taker_fee_per_share


@dataclass(frozen=True)
class AssetSpec:
    asset: str
    polymarket_slug_prefix: str
    source: str
    symbol: str


DEFAULT_ASSETS = [
    AssetSpec("BTC", "btc-updown-5m", "binance", "BTCUSDT"),
    AssetSpec("ETH", "eth-updown-5m", "binance", "ETHUSDT"),
    AssetSpec("SOL", "sol-updown-5m", "binance", "SOLUSDT"),
    AssetSpec("BNB", "bnb-updown-5m", "binance", "BNBUSDT"),
    AssetSpec("XRP", "xrp-updown-5m", "binance", "XRPUSDT"),
    AssetSpec("HYPE", "hyperliquid-updown-5m", "hyperliquid", "HYPE"),
    AssetSpec("DOGE", "dogecoin-updown-5m", "binance", "DOGEUSDT"),
]


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def get_json(url: str, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "poly-5m-model-backtest/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_binance_klines(
    symbol: str,
    *,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_seconds: float = 0.08,
) -> list[Candle]:
    candles: list[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        url = f"https://api.binance.com/api/v3/klines?{params}"
        raw = get_json(url)
        if not raw:
            break
        batch = [
            Candle(
                open_time_ms=int(item[0]),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
            )
            for item in raw
        ]
        candles.extend(batch)
        next_cursor = batch[-1].open_time_ms + interval_ms(interval)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(pause_seconds)
    deduped = {candle.open_time_ms: candle for candle in candles}
    return [deduped[key] for key in sorted(deduped)]


def fetch_hyperliquid_candles(
    coin: str,
    *,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[Candle]:
    payload = json.dumps(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "poly-5m-model-backtest/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        raw = json.loads(response.read().decode("utf-8"))
    return [
        Candle(
            open_time_ms=int(item["t"]),
            open=float(item["o"]),
            high=float(item["h"]),
            low=float(item["l"]),
            close=float(item["c"]),
            volume=float(item["v"]),
        )
        for item in raw
    ]


def fetch_candles(
    spec: AssetSpec,
    *,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[Candle]:
    if spec.source == "binance":
        return fetch_binance_klines(
            spec.symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    if spec.source == "hyperliquid":
        return fetch_hyperliquid_candles(
            spec.symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    raise ValueError(f"Unsupported source {spec.source} for {spec.asset}")


def interval_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    raise ValueError(f"Unsupported interval: {interval}")


def price_by_second(candles: list[Candle]) -> dict[int, float]:
    return {candle.open_time_ms // 1000: candle.open for candle in candles}


def log_loss(probability: float, outcome_up: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, probability))
    return -(outcome_up * math.log(p) + (1.0 - outcome_up) * math.log(1.0 - p))


def expected_calibration_error(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[min(9, max(0, int(row["p_up"] * 10)))].append(row)
    ece = 0.0
    for bucket_rows in buckets.values():
        avg_p = statistics.fmean(row["p_up"] for row in bucket_rows)
        avg_y = statistics.fmean(row["outcome_up"] for row in bucket_rows)
        ece += len(bucket_rows) / len(rows) * abs(avg_p - avg_y)
    return ece


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "brier": None,
            "log_loss": None,
            "direction_accuracy": None,
            "ece": None,
            "avg_abs_confidence": None,
            "even_odds_trades": 0,
            "even_odds_pnl_units": 0.0,
            "even_odds_win_rate": None,
        }
    briers = [(row["p_up"] - row["outcome_up"]) ** 2 for row in rows]
    losses = [log_loss(row["p_up"], row["outcome_up"]) for row in rows]
    directions = [
        (row["p_up"] >= 0.5 and row["outcome_up"] == 1.0)
        or (row["p_up"] < 0.5 and row["outcome_up"] == 0.0)
        for row in rows
    ]
    trades = [row for row in rows if abs(row["p_up"] - 0.5) >= row["confidence_threshold"]]
    trade_wins = [
        (row["p_up"] >= 0.5 and row["outcome_up"] == 1.0)
        or (row["p_up"] < 0.5 and row["outcome_up"] == 0.0)
        for row in trades
    ]
    return {
        "count": len(rows),
        "brier": statistics.fmean(briers),
        "log_loss": statistics.fmean(losses),
        "direction_accuracy": statistics.fmean(1.0 if item else 0.0 for item in directions),
        "ece": expected_calibration_error(rows),
        "avg_abs_confidence": statistics.fmean(abs(row["p_up"] - 0.5) for row in rows),
        "even_odds_trades": len(trades),
        "even_odds_pnl_units": sum(1.0 if item else -1.0 for item in trade_wins),
        "even_odds_win_rate": (
            statistics.fmean(1.0 if item else 0.0 for item in trade_wins)
            if trade_wins
            else None
        ),
    }


def proxy_effective_cost(entry_price: float, fee_rate: float) -> float:
    return entry_price + taker_fee_per_share(entry_price, fee_rate)


def combine_probability(example: dict[str, Any], weights: dict[str, float]) -> float | None:
    total = 0.0
    weighted = 0.0
    for name, weight in weights.items():
        if name not in example["models"]:
            continue
        w = max(float(weight), 0.0)
        total += w
        weighted += w * float(example["models"][name])
    if total <= 0:
        return None
    return max(0.01, min(0.99, weighted / total))


def build_examples(model_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    examples_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for name, rows in model_rows.items():
        if name.startswith("ensemble"):
            continue
        for row in rows:
            key = (str(row["asset"]), int(row["timestamp"]))
            if key not in examples_by_key:
                examples_by_key[key] = {
                    "asset": row["asset"],
                    "timestamp": row["timestamp"],
                    "outcome_up": row["outcome_up"],
                    "models": {},
                }
            examples_by_key[key]["models"][name] = row["p_up"]
    return list(examples_by_key.values())


def profile_rows(
    examples: list[dict[str, Any]],
    weights: dict[str, float],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        p_up = combine_probability(example, weights)
        if p_up is None:
            continue
        rows.append(
            {
                "p_up": p_up,
                "outcome_up": example["outcome_up"],
                "confidence_threshold": confidence_threshold,
            }
        )
    return rows


def evaluate_weight_profile(
    examples: list[dict[str, Any]],
    weights: dict[str, float],
    *,
    confidence_threshold: float,
    entry_price: float,
    fee_rate: float,
    min_edge: float,
) -> dict[str, Any]:
    rows = profile_rows(examples, weights, confidence_threshold)
    summary = summarize(rows)
    cost = proxy_effective_cost(entry_price, fee_rate)
    trades = []
    for row in rows:
        p_up = float(row["p_up"])
        p_win = max(p_up, 1.0 - p_up)
        edge = p_win - cost
        if edge < min_edge:
            continue
        side_up = p_up >= 0.5
        won = (side_up and row["outcome_up"] == 1.0) or (
            not side_up and row["outcome_up"] == 0.0
        )
        trades.append({"won": won, "pnl": (1.0 - cost) if won else -cost})
    summary.update(
        {
            "proxy_entry_price": entry_price,
            "proxy_effective_cost": cost,
            "proxy_min_edge": min_edge,
            "proxy_trades": len(trades),
            "proxy_net_pnl_units": sum(item["pnl"] for item in trades),
            "proxy_win_rate": (
                statistics.fmean(1.0 if item["won"] else 0.0 for item in trades)
                if trades
                else None
            ),
            "weights": weights,
        }
    )
    return summary


def candidate_weight_profiles(recommended: dict[str, float]) -> dict[str, dict[str, float]]:
    volatility_core = {
        "ewma_riskmetrics_volatility": 1.35,
        "garch_1_1": 1.50,
        "gjr_threshold_garch": 1.55,
        "har_realized_volatility": 1.45,
        "student_t_garch": 1.20,
        "regime_switching_volatility": 1.10,
        "merton_jump_diffusion": 0.85,
    }
    return {
        "recommended_calibration": recommended,
        "equal_all_models": {
            name: 1.0
            for name in [
                "distance_to_start_random_walk",
                "short_momentum",
                "mean_reversion",
                "volatility_fade",
                "ewma_riskmetrics_volatility",
                "garch_1_1",
                "gjr_threshold_garch",
                "har_realized_volatility",
                "student_t_garch",
                "regime_switching_volatility",
                "empirical_interval_knn",
                "merton_jump_diffusion",
                "kalman_local_trend",
            ]
        },
        "volatility_core": volatility_core,
        "volatility_core_plus_distance": {
            **volatility_core,
            "distance_to_start_random_walk": 0.40,
            "kalman_local_trend": 0.25,
        },
        "garch_family_only": {
            "garch_1_1": 1.0,
            "gjr_threshold_garch": 1.0,
            "student_t_garch": 0.8,
        },
        "gjr_only": {"gjr_threshold_garch": 1.0},
    }


def asset_specs(selected: list[str] | None) -> list[AssetSpec]:
    if not selected:
        return DEFAULT_ASSETS
    wanted = {item.upper() for item in selected}
    return [item for item in DEFAULT_ASSETS if item.asset in wanted]


def replay_asset(
    spec: AssetSpec,
    candles: list[Candle],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    prices = price_by_second(candles)
    window = RollingPriceWindow(max_start_capture_lag_seconds=65)
    ensemble = Ensemble()
    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intervals = 0
    evaluated_points = 0
    for candle in candles:
        timestamp = candle.open_time_ms // 1000
        start_epoch = timestamp - (timestamp % 300)
        end_epoch = start_epoch + 300
        start_price = prices.get(start_epoch)
        end_price = prices.get(end_epoch)
        observation = PriceObservation(
            timestamp=timestamp,
            market_start_epoch=start_epoch,
            market_end_epoch=end_epoch,
            spot_price=candle.open,
        )
        window.append(observation)
        if start_price is None or end_price is None or timestamp >= end_epoch:
            continue
        forecast = ensemble.forecast(window, observation, None, None)
        if forecast is None:
            continue
        intervals += 1 if timestamp == start_epoch else 0
        evaluated_points += 1
        outcome_up = 1.0 if end_price >= start_price else 0.0
        models = {item.name: item.p_up for item in forecast.forecasts}
        models["ensemble_calibrated"] = forecast.p_up
        models["ensemble_raw_mean"] = statistics.fmean(item.p_up for item in forecast.forecasts)
        for name, p_up in models.items():
            model_rows[name].append(
                {
                    "asset": spec.asset,
                    "timestamp": timestamp,
                    "seconds_from_start": timestamp - start_epoch,
                    "p_up": p_up,
                    "outcome_up": outcome_up,
                    "confidence_threshold": confidence_threshold,
                }
            )
    return {
        "asset": spec.asset,
        "source": spec.source,
        "symbol": spec.symbol,
        "polymarket_slug_prefix": spec.polymarket_slug_prefix,
        "candles": len(candles),
        "evaluated_points": evaluated_points,
        "models": {name: summarize(rows) for name, rows in sorted(model_rows.items())},
        "rows": model_rows,
    }


def recommended_weights(model_rows: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, rows in model_rows.items():
        if name.startswith("ensemble") or len(rows) < 500:
            continue
        summary = summarize(rows)
        brier = summary["brier"]
        ece = summary["ece"]
        if brier is None or ece is None:
            continue
        # The bot uses model probabilities for Kelly sizing, so calibration
        # quality matters more than directional hit rate at an arbitrary 50c
        # benchmark. Directional/PnL diagnostics stay in the report, but the
        # recommendation is intentionally Brier/ECE first.
        scores[name] = max(0.001, 0.25 - float(brier) - 0.75 * float(ece))
    if not scores:
        return {}
    best_score = max(scores.values())
    weights = {}
    for name, score in scores.items():
        relative = score / best_score if best_score > 0 else 1.0
        weights[name] = round(max(0.35, min(1.60, 0.35 + 1.25 * relative)), 2)
    return weights


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Underlying Crypto Model Backtest",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Source: `Binance spot klines`, interval `{report['interval']}`, lookback `{report['days']}d`",
        "",
        "## Polymarket 5M Crypto Universe",
        "",
        "| Asset | Slug Prefix | Exchange Symbol | Candles | Evaluated Points |",
        "|---|---|---|---:|---:|",
    ]
    for asset in report["assets"]:
        lines.append(
            f"| {asset['asset']} | `{asset['polymarket_slug_prefix']}` | `{asset['source']}:{asset['symbol']}` | {asset['candles']} | {asset['evaluated_points']} |"
        )
    lines.extend(
        [
            "",
            "## Combined Model Ranking",
            "",
            "| Model | Count | Brier | Log Loss | ECE | Direction | Even-Odds Trades | Even-Odds PnL Units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in sorted(
        report["combined_models"].items(),
        key=lambda item: item[1]["brier"] if item[1]["brier"] is not None else 999,
    ):
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                name,
                row["count"],
                f"{row['brier']:.4f}" if row["brier"] is not None else "",
                f"{row['log_loss']:.4f}" if row["log_loss"] is not None else "",
                f"{row['ece']:.4f}" if row["ece"] is not None else "",
                f"{row['direction_accuracy']:.3f}" if row["direction_accuracy"] is not None else "",
                row["even_odds_trades"],
                row["even_odds_pnl_units"],
            )
        )
    lines.extend(["", "## Recommended Research Weights", ""])
    for name, weight in report["recommended_weights"].items():
        lines.append(f"- `{name}`: `{weight}`")
    lines.extend(
        [
            "",
            "## Weight Profile Trading Proxy",
            "",
            f"Assumes a `{report['entry_price']:.2f}` binary entry, fee rate `{report['fee_rate']:.3f}`, and minimum edge `{report['min_edge']:.3f}`.",
            "",
            "| Profile | Brier | Log Loss | Proxy Trades | Proxy Win Rate | Proxy Net PnL Units |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in sorted(
        report["weight_profiles"].items(),
        key=lambda item: item[1]["proxy_net_pnl_units"],
        reverse=True,
    ):
        lines.append(
            "| `{}` | {} | {} | {} | {} | {:.2f} |".format(
                name,
                f"{row['brier']:.4f}" if row["brier"] is not None else "",
                f"{row['log_loss']:.4f}" if row["log_loss"] is not None else "",
                row["proxy_trades"],
                f"{row['proxy_win_rate']:.3f}" if row["proxy_win_rate"] is not None else "",
                row["proxy_net_pnl_units"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest model forecasts on underlying crypto price candles")
    parser.add_argument("--asset", action="append", help="Asset to include, e.g. BTC. Defaults to current Polymarket 5M crypto universe.")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--confidence-threshold", type=float, default=0.04)
    parser.add_argument("--entry-price", type=float, default=0.50)
    parser.add_argument("--min-edge", type=float, default=0.04)
    parser.add_argument("--fee-rate", type=float, default=0.07)
    parser.add_argument("--reports-dir", default="reports/model_backtests")
    args = parser.parse_args()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(args.days * 24 * 60 * 60 * 1000)
    assets = []
    combined_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in asset_specs(args.asset):
        candles = fetch_candles(
            spec,
            interval=args.interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        asset_report = replay_asset(
            spec,
            candles,
            confidence_threshold=args.confidence_threshold,
        )
        for name, rows in asset_report.pop("rows").items():
            combined_rows[name].extend(rows)
        assets.append(asset_report)
    recommended = recommended_weights(combined_rows)
    examples = build_examples(combined_rows)
    profiles = {
        name: evaluate_weight_profile(
            examples,
            weights,
            confidence_threshold=args.confidence_threshold,
            entry_price=args.entry_price,
            fee_rate=args.fee_rate,
            min_edge=args.min_edge,
        )
        for name, weights in candidate_weight_profiles(recommended).items()
        if weights
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "days": args.days,
        "interval": args.interval,
        "confidence_threshold": args.confidence_threshold,
        "entry_price": args.entry_price,
        "min_edge": args.min_edge,
        "fee_rate": args.fee_rate,
        "assets": assets,
        "combined_models": {name: summarize(rows) for name, rows in sorted(combined_rows.items())},
        "recommended_weights": recommended,
        "weight_profiles": profiles,
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = reports_dir / f"underlying_model_backtest_{stamp}.json"
    md_path = reports_dir / f"underlying_model_backtest_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
