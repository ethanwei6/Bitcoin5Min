from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_evidence_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_evidence_report.py"
    spec = importlib.util.spec_from_file_location("generate_evidence_report", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proxy_winners_use_all_observed_completed_markets() -> None:
    evidence = load_evidence_module()
    snapshots = [
        {
            "timestamp": 1002.0,
            "market": {"slug": "btc-a", "start_epoch": 1000, "end_epoch": 1300},
            "spot": {"median_price": 100.0},
        },
        {
            "timestamp": 1303.0,
            "market": {"slug": "btc-b", "start_epoch": 1300, "end_epoch": 1600},
            "spot": {"median_price": 101.0},
        },
        {
            "timestamp": 1602.0,
            "market": {"slug": "btc-c", "start_epoch": 1600, "end_epoch": 1900},
            "spot": {"median_price": 99.0},
        },
    ]

    winners = evidence.proxy_winners_from_snapshots(snapshots)

    assert winners["btc-a"] == "UP"
    assert winners["btc-b"] == "DOWN"


def test_market_level_calibration_uses_one_latest_signal_per_market() -> None:
    evidence = load_evidence_module()
    signals = [
        {"timestamp": 1, "market_slug": "a", "forecast": {"p_up": 0.20, "forecasts": []}},
        {"timestamp": 2, "market_slug": "a", "forecast": {"p_up": 0.80, "forecasts": []}},
        {"timestamp": 1, "market_slug": "b", "forecast": {"p_up": 0.30, "forecasts": []}},
    ]

    report = evidence.market_level_calibration(signals, {"a": "UP", "b": "DOWN"})

    assert report["settled_market_count"] == 2
    assert report["ensemble_brier"] < 0.10
