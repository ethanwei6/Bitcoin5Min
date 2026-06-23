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


def test_suppressed_edge_diagnostics_show_raw_positive_edges_cut_by_calibration() -> None:
    evidence = load_evidence_module()
    signals = [
        {
            "decision": {
                "trade_cohort": "underdog_continuation",
                "effective_cost": 0.30,
                "raw_probability": 0.32,
                "probability": 0.29,
                "probability_haircut": 0.03,
                "executable_price": 0.28,
                "reason": "edge below threshold",
            }
        },
        {
            "decision": {
                "trade_cohort": "directional_confidence",
                "effective_cost": 0.60,
                "raw_probability": 0.64,
                "probability": 0.64,
                "probability_haircut": 0.0,
                "executable_price": 0.59,
                "reason": "trade eligible",
            }
        },
    ]

    report = evidence.suppressed_edge_diagnostics(signals)

    assert report["overall"]["raw_positive_edges"] == 2
    assert report["overall"]["adjusted_positive_edges"] == 1
    assert report["overall"]["suppressed_raw_positive_edges"] == 1
    assert report["by_cohort"]["underdog_continuation"]["suppressed_raw_positive_edges"] == 1


def test_data_freshness_reports_sampling_and_book_churn() -> None:
    evidence = load_evidence_module()
    snapshots = [
        {
            "timestamp": 100.0,
            "market": {"slug": "btc-a"},
            "up_book": {"hash": "u1", "best_ask": {"price": 0.51}},
            "down_book": {"hash": "d1", "best_ask": {"price": 0.50}},
        },
        {
            "timestamp": 105.0,
            "market": {"slug": "btc-a"},
            "up_book": {"hash": "u2", "best_ask": {"price": 0.52}},
            "down_book": {"hash": "d1", "best_ask": {"price": 0.50}},
        },
        {
            "timestamp": 113.0,
            "market": {"slug": "btc-a"},
            "up_book": None,
            "down_book": {"hash": "d2", "best_ask": {"price": 0.49}},
        },
        {
            "timestamp": 120.0,
            "market": {"slug": "btc-a"},
            "settle_only": True,
        },
    ]
    signals = [{"timestamp": 106.0, "market_slug": "btc-a"}]

    report = evidence.data_freshness(snapshots, signals)

    assert report["market_snapshots"] == 4
    assert report["decision_snapshots"] == 3
    assert report["settle_only_snapshots"] == 1
    assert report["median_snapshot_interval_seconds"] == 7.0
    assert report["p95_snapshot_interval_seconds"] == 8.0
    assert report["missing_up_book_snapshots"] == 1
    assert report["missing_down_book_snapshots"] == 0
    assert report["up_book_hash_changes"] == 1
    assert report["down_book_hash_changes"] == 1
    assert report["up_top_ask_changes"] == 1
    assert report["down_top_ask_changes"] == 1
    assert report["median_signal_snapshot_lag_seconds"] == 1.0
