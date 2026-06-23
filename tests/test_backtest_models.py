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


def test_model_rows_for_examples_uses_asset_timestamp_keys() -> None:
    backtest = load_backtest_module()
    rows = {
        "a": [
            {"asset": "BTC", "timestamp": 1, "p_up": 0.6},
            {"asset": "ETH", "timestamp": 1, "p_up": 0.4},
            {"asset": "BTC", "timestamp": 2, "p_up": 0.7},
        ],
        "b": [
            {"asset": "BTC", "timestamp": 1, "p_up": 0.55},
            {"asset": "ETH", "timestamp": 2, "p_up": 0.45},
        ],
    }
    examples = [
        {"asset": "BTC", "timestamp": 1},
        {"asset": "ETH", "timestamp": 2},
    ]

    filtered = backtest.model_rows_for_examples(rows, examples)

    assert filtered["a"] == [{"asset": "BTC", "timestamp": 1, "p_up": 0.6}]
    assert filtered["b"] == [
        {"asset": "BTC", "timestamp": 1, "p_up": 0.55},
        {"asset": "ETH", "timestamp": 2, "p_up": 0.45},
    ]


def test_model_rows_for_asset_keeps_only_requested_asset() -> None:
    backtest = load_backtest_module()
    rows = {
        "a": [
            {"asset": "BTC", "timestamp": 1, "p_up": 0.6},
            {"asset": "ETH", "timestamp": 1, "p_up": 0.4},
        ],
        "b": [
            {"asset": "BTC", "timestamp": 2, "p_up": 0.55},
            {"asset": "SOL", "timestamp": 2, "p_up": 0.45},
        ],
    }

    filtered = backtest.model_rows_for_asset(rows, "BTC")

    assert filtered == {
        "a": [{"asset": "BTC", "timestamp": 1, "p_up": 0.6}],
        "b": [{"asset": "BTC", "timestamp": 2, "p_up": 0.55}],
    }


def test_candidate_weight_profiles_include_current_config_when_provided() -> None:
    backtest = load_backtest_module()
    configured = {"garch_1_1": 1.25, "merton_jump_diffusion": 0.75}

    profiles = backtest.candidate_weight_profiles(
        {"garch_1_1": 1.0},
        configured=configured,
    )

    assert profiles["configured_model_weights"] == configured


def test_weight_profile_reports_proxy_trade_buckets() -> None:
    backtest = load_backtest_module()
    examples = [
        {
            "asset": "BTC",
            "timestamp": 1,
            "seconds_from_start": 20,
            "outcome_up": 1.0,
            "models": {"a": 0.62},
        },
        {
            "asset": "BTC",
            "timestamp": 2,
            "seconds_from_start": 80,
            "outcome_up": 0.0,
            "models": {"a": 0.38},
        },
        {
            "asset": "ETH",
            "timestamp": 3,
            "seconds_from_start": 250,
            "outcome_up": 1.0,
            "models": {"a": 0.40},
        },
    ]

    profile = backtest.evaluate_weight_profile(
        examples,
        {"a": 1.0},
        confidence_threshold=0.04,
        entry_price=0.50,
        fee_rate=0.0,
        min_edge=0.0,
    )

    assert profile["proxy_trades"] == 3
    assert profile["proxy_by_time_bucket"]["000-030s"]["trades"] == 1
    assert profile["proxy_by_time_bucket"]["060-120s"]["trades"] == 1
    assert profile["proxy_by_time_bucket"]["240-300s"]["net_pnl_units"] == -0.5
    assert profile["proxy_by_confidence_bucket"]["60-65%"]["trades"] == 3
    assert profile["proxy_by_asset"]["BTC"]["trades"] == 2
    assert profile["proxy_by_asset"]["ETH"]["win_rate"] == 0.0


def test_data_quality_report_flags_minute_candles_as_research_only() -> None:
    backtest = load_backtest_module()

    quality = backtest.data_quality_report(
        [{"asset": "BTC", "sample_seconds": 60.0}],
        requested_interval="1m",
        bot_sample_interval_seconds=5.0,
        market_interval_seconds=300.0,
    )

    assert quality["grade"] == "coarse_research_only"
    assert quality["requested_sample_seconds"] == 60.0
    assert quality["historical_samples_per_market"] == 5.0
    assert quality["bot_samples_per_market"] == 60.0
    assert quality["cadence_ratio_to_bot"] == 12.0
    assert quality["cadence_ratio_to_requested"] == 1.0
    assert quality["requested_interval_match"] is True
    assert any("second-by-second" in item for item in quality["limitations"])


def test_data_quality_report_flags_requested_interval_mismatch() -> None:
    backtest = load_backtest_module()

    quality = backtest.data_quality_report(
        [{"asset": "BTC", "sample_seconds": 60.0}],
        requested_interval="1s",
        bot_sample_interval_seconds=5.0,
        market_interval_seconds=300.0,
    )

    assert quality["requested_sample_seconds"] == 1.0
    assert quality["cadence_ratio_to_requested"] == 60.0
    assert quality["requested_interval_match"] is False
    assert any("coarser than the requested interval" in item for item in quality["limitations"])


def test_markdown_discloses_exchange_data_not_polymarket_replay() -> None:
    backtest = load_backtest_module()
    report = {
        "generated_at": "2026-06-19T00:00:00+00:00",
        "interval": "1m",
        "days": 1.0,
        "config_path": "config/paper_btc_5m.json",
        "configured_model_weights": {"garch_1_1": 1.25},
        "data_quality": {
            "requested_interval": "1m",
            "requested_sample_seconds": 60.0,
            "historical_sample_seconds_median": 60.0,
            "historical_sample_seconds_max": 60.0,
            "bot_sample_interval_seconds": 5.0,
            "market_interval_seconds": 300.0,
            "historical_samples_per_market": 5.0,
            "bot_samples_per_market": 60.0,
            "cadence_ratio_to_bot": 12.0,
            "cadence_ratio_to_requested": 1.0,
            "requested_interval_match": True,
            "asset_sample_seconds": {"BTC": 60.0},
            "grade": "coarse_research_only",
            "limitations": [
                "Minute candles only observe one price per minute; they miss intraminute path.",
            ],
        },
        "assets": [
            {
                "asset": "BTC",
                "polymarket_slug_prefix": "btc-updown-5m",
                "source": "binance",
                "symbol": "BTCUSDT",
                "sample_seconds": 60.0,
                "candles": 10,
                "evaluated_points": 5,
            }
        ],
        "combined_models": {},
        "recommended_weights": {},
        "entry_price": 0.5,
        "fee_rate": 0.07,
        "min_edge": 0.04,
        "weight_profiles": {},
        "holdout_train_examples": 0,
        "holdout_test_examples": 0,
        "holdout_recommended_weights": {},
        "holdout_weight_profiles": {},
        "holdout_best_profile": None,
        "asset_holdout_weight_profiles": {},
        "train_fraction": 0.7,
        "min_calibration_bucket_count": 30,
        "train_test_calibration": {},
    }

    text = backtest.markdown(report)

    assert "real exchange candles" in text
    assert "does not replay historical Polymarket CLOB" in text
    assert "Data Adequacy" in text
    assert "coarse_research_only" in text
    assert "Actual / Requested" in text
    assert "Cadence Ratio To Bot" in text
    assert "Sample Seconds" in text
    assert "Chronological Holdout Weight Profiles" in text
    assert "Per-Asset Holdout Weight Profiles" in text
    assert "Use the chronological holdout section" in text
