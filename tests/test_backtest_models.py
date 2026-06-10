from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_backtest_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "backtest_models.py"
    spec = importlib.util.spec_from_file_location("backtest_models", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interval_ms_supports_second_and_minute_candles() -> None:
    backtest = load_backtest_module()

    assert backtest.interval_ms("1s") == 1_000
    assert backtest.interval_ms("5s") == 5_000
    assert backtest.interval_ms("1m") == 60_000


def test_train_test_split_is_chronological_before_asset_name() -> None:
    backtest = load_backtest_module()
    rows = [
        {"asset": "ZZZ", "timestamp": 2},
        {"asset": "AAA", "timestamp": 1},
        {"asset": "AAA", "timestamp": 3},
        {"asset": "ZZZ", "timestamp": 1},
    ]

    train, test = backtest.train_test_split_rows(rows, 0.5)

    assert [(row["timestamp"], row["asset"]) for row in train] == [(1, "AAA"), (1, "ZZZ")]
    assert [(row["timestamp"], row["asset"]) for row in test] == [(2, "ZZZ"), (3, "AAA")]


def test_confidence_calibration_preserves_probability_side() -> None:
    backtest = load_backtest_module()
    train_rows = [
        {"seconds_from_start": 90, "p_up": 0.7, "outcome_up": 1.0},
        {"seconds_from_start": 90, "p_up": 0.7, "outcome_up": 1.0},
        {"seconds_from_start": 90, "p_up": 0.7, "outcome_up": 0.0},
        {"seconds_from_start": 90, "p_up": 0.3, "outcome_up": 0.0},
        {"seconds_from_start": 90, "p_up": 0.3, "outcome_up": 1.0},
        {"seconds_from_start": 90, "p_up": 0.3, "outcome_up": 0.0},
    ]
    test_rows = [
        {"seconds_from_start": 90, "p_up": 0.7, "outcome_up": 1.0},
        {"seconds_from_start": 90, "p_up": 0.3, "outcome_up": 0.0},
    ]

    calibration, fallback = backtest.bucket_win_rate(train_rows, min_bucket_count=2)
    calibrated = backtest.apply_confidence_calibration(test_rows, calibration, fallback)

    assert calibrated[0]["p_up"] > 0.5
    assert calibrated[1]["p_up"] < 0.5
    assert abs(calibrated[0]["p_up"] - 2 / 3) < 1e-12
    assert abs(calibrated[1]["p_up"] - 1 / 3) < 1e-12
