from __future__ import annotations

import json
import calendar
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import BotConfig
from .execution import PaperExecutionSimulator
from .journal import JsonlJournal
from .market import PolymarketDiscovery
from .models import Ensemble, PriceObservation, RollingPriceWindow
from .orderbook import ClobOrderBookClient, OrderBook
from .paper import PaperBroker
from .resolution import GammaResolutionClient
from .risk import RiskEngine
from .spot import SpotPriceClient


@dataclass(frozen=True)
class DailyRiskStats:
    realized_pnl_usd: float
    drawdown_usd: float
    consecutive_losses: int


class PaperTradingBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.journal = JsonlJournal(self.output_dir)
        self.discovery = PolymarketDiscovery(config.polymarket, config.market_interval_seconds)
        self.books = ClobOrderBookClient(config.polymarket)
        self.resolution = GammaResolutionClient(config.polymarket)
        self.spot = SpotPriceClient(
            config.spot_sources,
            max_source_spread_usd=config.max_spot_source_spread_usd,
        )
        self.window = RollingPriceWindow(
            max_start_capture_lag_seconds=config.max_start_capture_lag_seconds
        )
        self.ensemble = Ensemble(
            model_weights=config.model_weights or None,
            market_prior_weight=config.market_prior_weight,
            disagreement_shrink=config.disagreement_shrink,
            horizon_confidence_min_multiplier=config.horizon_confidence_min_multiplier,
            horizon_confidence_power=config.horizon_confidence_power,
        )
        self.risk = RiskEngine(config)
        self.execution = PaperExecutionSimulator(
            enabled=config.simulate_execution_latency,
            order_type=config.execution_order_type,
            max_slippage_ticks=config.execution_max_slippage_ticks,
            min_fill_ratio=config.execution_min_fill_ratio,
            fee_rate=config.fee_rate,
        )
        self.broker = PaperBroker(
            state_path=self.output_dir / "state.json",
            starting_cash_usd=config.paper_starting_cash_usd,
            journal=self.journal,
            fee_rate=config.fee_rate,
        )
        self.market_end_prices: dict[int, tuple[float, float]] = {}
        self._running = True
        self._draining = False
        self._stop_requested = False

    def stop(self, *_args: object) -> None:
        self._draining = True
        self._stop_requested = True
        self.journal.append(
            "lifecycle.jsonl",
            {"event": "stop_requested", "mode": "drain_until_positions_resolve"},
        )

    def run_forever(
        self,
        duration_seconds: float | None = None,
        settle_only: bool = False,
        drain_before_stop_seconds: float | None = None,
    ) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        stop_at = time.time() + duration_seconds if duration_seconds is not None else None
        drain_before = (
            self.config.drain_before_stop_seconds
            if drain_before_stop_seconds is None
            else drain_before_stop_seconds
        )
        if settle_only:
            self._draining = True
        while self._running:
            now = time.time()
            if stop_at is not None and now >= stop_at - drain_before:
                if not self._draining:
                    self.journal.append(
                        "lifecycle.jsonl",
                        {
                            "event": "drain_started",
                            "reason": "bounded_run_approaching_stop",
                            "stop_at": stop_at,
                            "drain_before_stop_seconds": drain_before,
                        },
                    )
                self._draining = True
            if (
                self._draining
                and not self.broker.has_open_positions()
                and (self._stop_requested or stop_at is None or now >= stop_at)
            ):
                self.journal.append(
                    "lifecycle.jsonl",
                    {"event": "drain_complete", "open_positions": 0},
                )
                break
            started = time.time()
            try:
                self.tick(settle_only=settle_only or self._draining)
            except Exception as exc:
                self.journal.append("errors.jsonl", {"error": repr(exc)})
                print(f"tick failed: {exc}", flush=True)
            elapsed = time.time() - started
            time.sleep(max(0.0, self.config.sample_interval_seconds - elapsed))

    def tick(self, settle_only: bool = False) -> None:
        timestamp = time.time()
        market = self.discovery.current_market(timestamp)
        spot = self.spot.snapshot()

        official_outcomes = self._official_outcomes_for_due_positions(timestamp)
        self.broker.settle_due_positions_official(official_outcomes, now=timestamp)

        if market is None:
            self.journal.append(
                "snapshots.jsonl",
                {"timestamp": timestamp, "market": None, "spot": spot},
            )
            return

        if settle_only:
            self.journal.append(
                "snapshots.jsonl",
                {"timestamp": timestamp, "market": market, "spot": spot, "settle_only": True},
            )
            return

        up_book = self._safe_book(market.tokens.up)
        down_book = self._safe_book(market.tokens.down)
        observation = PriceObservation(
            timestamp=timestamp,
            market_start_epoch=market.start_epoch,
            market_end_epoch=market.end_epoch,
            spot_price=spot.median_price,
        )
        self.window.append(observation)
        self.journal.append(
            "snapshots.jsonl",
            {
                "timestamp": timestamp,
                "market": market,
                "spot": spot,
                "up_book": up_book,
                "down_book": down_book,
            },
        )

        if up_book is None or down_book is None:
            return
        forecast = self.ensemble.forecast(self.window, observation, up_book, down_book)
        if forecast is None:
            return

        daily_risk = self._daily_risk_stats()
        market_entries = self._market_entry_count(market.slug)
        decision = self.risk.decide(
            forecast=forecast,
            up_book=up_book,
            down_book=down_book,
            cash_usd=self.broker.state.cash_usd,
            market_exposure_usd=self.broker.market_exposure(market.slug),
            market_entries=market_entries,
            daily_pnl_usd=daily_risk.realized_pnl_usd,
            daily_drawdown_usd=daily_risk.drawdown_usd,
            consecutive_losses=daily_risk.consecutive_losses,
            seconds_from_start=market.seconds_from_start,
            seconds_to_end=market.seconds_to_end,
        )
        self.journal.append(
            "signals.jsonl",
            {
                "timestamp": timestamp,
                "market_slug": market.slug,
                "forecast": forecast,
                "decision": decision,
                "cash_usd": self.broker.state.cash_usd,
                "market_exposure_usd": self.broker.market_exposure(market.slug),
                "market_entries": market_entries,
                "daily_risk": daily_risk,
            },
        )
        if decision.should_trade:
            book = up_book if decision.side == "UP" else down_book
            token_id = market.tokens.up if decision.side == "UP" else market.tokens.down
            execution = self.execution.simulate_buy(
                decision=decision,
                token_id=token_id,
                book_at_signal=book,
                fetch_current_book=self.books.get_book,
                market_taker_delay_ms=self._market_taker_delay_ms(market.condition_id),
                market_taker_delay_source="clob_market_info_itode",
            )
            execution_record = {
                "timestamp": time.time(),
                "market_slug": market.slug,
                "decision": decision,
                "execution": execution,
            }
            if not execution.accepted:
                self.journal.append("execution_simulations.jsonl", execution_record)
                return
            post_execution_risk = self.risk.check_execution_fill(
                decision,
                fill_price=execution.fill_price,
                fill_cost_usd=execution.fill_cost_usd,
                fill_shares=execution.fill_shares,
            )
            execution_record["post_execution_risk"] = post_execution_risk
            self.journal.append("execution_simulations.jsonl", execution_record)
            if not post_execution_risk["accepted"]:
                return
            self.broker.open_position(
                market,
                decision,
                self.window.current_market_start_price(market.start_epoch),
                execution=execution.to_dict(),
            )

    def _safe_book(self, token_id: str) -> OrderBook | None:
        try:
            return self.books.get_book(token_id)
        except Exception as exc:
            self.journal.append("errors.jsonl", {"error": repr(exc), "token_id": token_id})
            return None

    def _market_taker_delay_ms(self, condition_id: str) -> float:
        try:
            market_info = self.books.get_market_info(condition_id)
        except Exception as exc:
            self.journal.append(
                "errors.jsonl",
                {"error": repr(exc), "condition_id": condition_id, "context": "clob_market_info"},
            )
            return 0.0
        return 250.0 if bool(market_info.get("itode")) else 0.0

    def _capture_proxy_settlement_prices(self, timestamp: float, spot_price: float) -> None:
        for start_epoch, start_price in list(self.window.market_start_prices.items()):
            end_epoch = start_epoch + self.config.market_interval_seconds
            if timestamp >= end_epoch and start_epoch not in self.market_end_prices:
                self.market_end_prices[start_epoch] = (start_price, spot_price)
        for position in self.broker.state.positions:
            if (
                position.settled
                or position.start_price_proxy is None
                or timestamp < position.market_end_epoch
                or position.market_start_epoch in self.market_end_prices
            ):
                continue
            self.market_end_prices[position.market_start_epoch] = (
                position.start_price_proxy,
                spot_price,
            )

    def _official_outcomes_for_due_positions(self, timestamp: float) -> dict[str, tuple[str, str]]:
        outcomes = {}
        due_slugs = {
            position.market_slug
            for position in self.broker.state.positions
            if not position.settled and timestamp >= position.market_end_epoch
        }
        for slug in due_slugs:
            resolution = self.resolution.get_resolution(slug)
            if resolution and resolution.closed and resolution.winning_side:
                outcomes[slug] = (resolution.winning_side, resolution.source)
        return outcomes

    def _market_entry_count(self, market_slug: str) -> int:
        path = self.output_dir / "trades.jsonl"
        if not path.exists():
            return 0
        entries = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    position = record.get("position") or {}
                    if position.get("market_slug") == market_slug:
                        entries += 1
        except (OSError, ValueError):
            return 0
        return entries

    def _daily_risk_stats(self) -> DailyRiskStats:
        trades_path = self.output_dir / "trades.jsonl"
        official_path = self.output_dir / "official_settlements.jsonl"
        settlement_path = official_path if official_path.exists() else self.output_dir / "settlements.jsonl"
        if not trades_path.exists() or not settlement_path.exists():
            return DailyRiskStats(0.0, 0.0, 0)
        current_day = time.gmtime()
        day_start = calendar.timegm(
            (current_day.tm_year, current_day.tm_mon, current_day.tm_mday, 0, 0, 0, 0, 0, 0)
        )
        try:
            with trades_path.open("r", encoding="utf-8") as handle:
                trades = [json.loads(line) for line in handle if line.strip()]
            with settlement_path.open("r", encoding="utf-8") as handle:
                settlements = [json.loads(line) for line in handle if line.strip()]
        except (OSError, ValueError):
            return DailyRiskStats(0.0, 0.0, 0)

        pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        consecutive_losses = 0
        for index, trade in enumerate(trades):
            position = trade.get("position") or {}
            if float(position.get("opened_at", 0.0)) < day_start:
                continue
            if index >= len(settlements):
                continue
            trade_pnl = float(settlements[index].get("pnl_usd", 0.0))
            pnl += trade_pnl
            peak = max(peak, pnl)
            max_drawdown = max(max_drawdown, peak - pnl)
            if trade_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
        return DailyRiskStats(pnl, max_drawdown, consecutive_losses)


def run_once(config: BotConfig, settle_only: bool = False) -> None:
    bot = PaperTradingBot(config)
    bot.tick(settle_only=settle_only)
