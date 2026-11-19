# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt: off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init
# fmt: on

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
import tempfile
import os
from unittest.mock import Mock

from load_regression import LoadRegressionPredictor, MODEL_VERSION, MIN_HISTORY_DAYS, STEP_MINUTES, TERMS
from load_predictor import LoadPredictor


def test_load_regression(my_predbat=None):
    """
    Test suite for LoadReg, the linear regression that load_ml_model: regression selects.

    Tests:
    - The regression recovering known weights, and its per-term breakdown
    - The trimmed time-of-day profile
    - Forecast format, with and without temperature data
    - Model save/load with version check
    - Cold start and gaps in the data
    - LoadMLComponent picking the model, keeping a model file per type, and training, predicting and publishing with it
    """
    sub_tests = [
        ("training_recovers_weights", _test_training_recovers_weights, "Fit recovers known weights and explain() adds up"),
        ("profile_ignores_spike", _test_profile_ignores_one_off_spike, "Profile leaves out a one-off spike but keeps the day's energy"),
        ("prediction", _test_prediction, "End-to-end prediction"),
        ("prediction_without_temperature", _test_prediction_without_temperature, "Prediction with no temperature data, and with no temperature forecast"),
        ("model_persistence", _test_model_persistence, "Model save/load with version check"),
        ("cold_start", _test_cold_start, "Cold start with insufficient data"),
        ("nan_inf_robustness", _test_nan_inf_robustness, "Handles NaN, Inf, and None in its inputs"),
        ("component_selects_model", _test_component_selects_model, "load_ml_model picks the predictor, its model file and its minimum history"),
        ("component_end_to_end", _test_component_end_to_end, "LoadMLComponent trains, predicts and publishes with the regression"),
    ]

    failed_tests = []
    passed_count = 0

    for name, test_func, description in sub_tests:
        try:
            print(f"  Running {name}: {description}...", end=" ")
            test_func()
            print("PASS")
            passed_count += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback

            traceback.print_exc()
            failed_tests.append((name, str(e)))

    print(f"\nLoadReg Tests: {passed_count}/{len(sub_tests)} passed")
    if failed_tests:
        print("Failed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
        assert False, f"LoadReg: {len(failed_tests)} tests failed"


def _create_synthetic_temp_data(n_days=7, now_utc=None, forecast_hours=48):
    """Synthetic temperature, history (positive minutes) and forecast (negative minutes): 2 to 10 C over the day plus a 3-day swing."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    def temp_at(minute):
        dt = now_utc - timedelta(minutes=minute)
        hours_since_peak = (dt.hour + dt.minute / 60.0 - 13.0) % 24.0
        return 6.0 + 4.0 * np.cos(2 * np.pi * hours_since_peak / 24.0) + 0.5 * np.sin(2 * np.pi * abs(minute) / (24 * 60) / 3.0)

    temp_minutes = {minute: temp_at(minute) for minute in range((n_days * 24 * 60 // STEP_MINUTES) * STEP_MINUTES, -STEP_MINUTES, -STEP_MINUTES)}
    for step in range(1, (forecast_hours * 60 // STEP_MINUTES) + 1):
        temp_minutes[-step * STEP_MINUTES] = temp_at(-step * STEP_MINUTES)
    return temp_minutes


def _create_synthetic_load_data(n_days=7, now_utc=None):
    """Synthetic per-5-minute load (not cumulative), minute 0 the most recent: ~0.2 kWh by day, ~0.05 kWh at night."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    load_minutes = {}
    for minute in range((n_days * 24 * 60 // STEP_MINUTES) * STEP_MINUTES, -STEP_MINUTES, -STEP_MINUTES):
        hour = (now_utc - timedelta(minutes=minute)).hour
        energy = 0.2 + 0.1 * np.random.randn() if 6 <= hour < 22 else 0.05 + 0.02 * np.random.randn()
        load_minutes[minute] = max(0.0, energy)
    return load_minutes


def _test_training_recovers_weights():
    """Load built as a daily shape plus 0.03 kWh per heating degree per 30 minutes is close to profile x 1 + temperature x 0.03."""
    now_utc = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    temp_data = _create_synthetic_temp_data(n_days=21, now_utc=now_utc, forecast_hours=48)

    def load_at(minute):
        """Energy in the 5 minutes that ended this many minutes ago (negative = future)."""
        return (0.3 + 0.2 * np.sin(2 * np.pi * minute / 1440.0)) / 6 + 0.005 * (15.5 - temp_data[minute])

    load_data = {minute: load_at(minute) for minute in temp_data if minute >= 0}

    predictor = LoadRegressionPredictor()
    val_mae = predictor.train(load_data, temp_minutes=temp_data)
    # Not exact: the profile leaves out the coldest day at each time of day and is then rescaled
    assert val_mae is not None and val_mae < 1e-3, f"Holdout error should be close to zero on data the model fits, got {val_mae}"
    assert np.allclose(predictor.weights, [0.0, 1.0, 0.03, 0.0, 0.0], atol=0.05), f"Weights {dict(zip(TERMS, predictor.weights))}"

    predictions = predictor.predict(load_data, temp_minutes=temp_data)
    breakdown = predictor.explain(load_data, temp_minutes=temp_data)
    assert sorted(breakdown) == list(range(0, 48 * 60, 30)), "One breakdown per 30-minute slot"
    cumulative = 0
    for minute, terms in breakdown.items():
        expected = sum(load_at(-(minute + step)) for step in range(5, 35, 5))
        assert abs(terms["total"] - expected) < 2e-3, f"Slot at +{minute} min: forecast {terms['total']}, truth {expected}"
        assert abs(sum(terms[term] for term in TERMS) - terms["total"]) < 1e-3, f"Terms should add up to the total: {terms}"
        cumulative += terms["total"]
        assert abs(predictions[minute + 25] - cumulative) < 1e-2, "Forecast should be the running sum of the slot totals"
    assert any(abs(terms["temperature"]) > 0.005 for terms in breakdown.values()), "Temperature forecast should move the forecast"


def _test_profile_ignores_one_off_spike():
    """A spike on one day does not shape the profile at that time of day, but its energy stays in the day's total."""
    load_data = {minute: 0.1 for minute in range(0, 14 * 24 * 60, STEP_MINUTES)}
    spike_minute = 2 * 24 * 60 + 12 * 60  # 2.5 days ago, so the forecast slot 12 hours ahead is at the same time of day
    load_data[spike_minute] += 3.0

    inputs, _ = LoadRegressionPredictor()._inputs(load_data, None, np.array([0]))
    profile = inputs[0, :48, TERMS.index("profile")]
    assert np.allclose(profile, profile[0]), f"Profile should be flat, the spike's slot is {profile[23]} against {profile[0]}"
    # A plain 7-day mean would put 0.6 + 3 / 7 = 1.03 kWh in that slot; the day's total is kept either way
    assert abs(profile.sum() - (48 * 0.6 + 3.0 / 7)) < 1e-9, f"Profile should keep the week's mean energy per day, got {profile.sum()}"

    # With fewer than 3 days at a time of day there is nothing to trim against, so the plain mean is used
    for minute in list(load_data):
        if (minute // 30) % 48 == (spike_minute // 30) % 48 and not 24 * 60 <= minute < 3 * 24 * 60:
            del load_data[minute]
    inputs, _ = LoadRegressionPredictor()._inputs(load_data, None, np.array([0]))
    assert inputs[0, 23, TERMS.index("profile")] > 1.5 * inputs[0, 0, TERMS.index("profile")], "With only two days known the spike should stay in the profile"


def _test_prediction():
    """End-to-end prediction: 48 hours at 5-minute steps, cumulative, with a daily shape."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)
    load_data = _create_synthetic_load_data(n_days=14, now_utc=now_utc)
    temp_data = _create_synthetic_temp_data(n_days=14, now_utc=now_utc, forecast_hours=48)

    predictor = LoadRegressionPredictor()
    val_mae = predictor.train(load_data, temp_minutes=temp_data)
    assert val_mae is not None and 0 < val_mae < 0.5, f"Holdout MAE should be small against ~0.9 kWh per 30 min, got {val_mae}"
    assert predictor.is_valid() == (True, None)

    predictions = predictor.predict(load_data, temp_minutes=temp_data)
    assert sorted(predictions) == list(range(0, 48 * 60, STEP_MINUTES)), "Should have a prediction every 5 minutes for 48 hours"
    values = [predictions[minute] for minute in sorted(predictions)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:])), "Cumulative forecast should never fall"
    assert all(isinstance(value, float) for value in values), "Forecast should be plain floats"
    # About 43 kWh a day: 16 hours at 0.2 kWh and 8 hours at 0.05 kWh per 5 minutes
    assert 70 < values[-1] < 100, f"48 hour total should be close to 86 kWh, got {values[-1]}"
    energy = np.diff([0] + values)
    assert energy.max() > 2 * energy.min(), "Forecast should have a daily shape"


def _test_prediction_without_temperature():
    """With no temperature at all the temperature weight is zero; with no forecast the temperature term is."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)
    load_data = _create_synthetic_load_data(n_days=14, now_utc=now_utc)

    predictor = LoadRegressionPredictor()
    assert predictor.train(load_data) is not None
    assert abs(predictor.weights[TERMS.index("temperature")]) < 1e-9, "No temperature data should give no temperature weight"
    assert len(predictor.predict(load_data)) == 48 * 60 // STEP_MINUTES

    predictor = LoadRegressionPredictor()
    history_only = _create_synthetic_temp_data(n_days=14, now_utc=now_utc, forecast_hours=0)
    assert predictor.train(load_data, temp_minutes=history_only) is not None
    breakdown = predictor.explain(load_data, temp_minutes=history_only)
    assert len(breakdown) == 96 and all(terms["temperature"] == 0 for terms in breakdown.values()), "A missing temperature forecast should leave the forecast unadjusted"


def _test_model_persistence():
    """Model save/load round trip, and a model of another version is refused."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)
    load_data = _create_synthetic_load_data(n_days=14, now_utc=now_utc)
    temp_data = _create_synthetic_temp_data(n_days=14, now_utc=now_utc, forecast_hours=48)
    predictor = LoadRegressionPredictor()
    predictor.train(load_data, temp_minutes=temp_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "predbat_ml_regression.npz")
        assert LoadRegressionPredictor().save(model_path) is False, "An untrained model should not be saved"
        assert predictor.save(model_path) is True

        predictor2 = LoadRegressionPredictor()
        assert predictor2.load(model_path) is True, "Model load should succeed"
        assert np.allclose(predictor.weights, predictor2.weights), "Weights should match after load"
        assert predictor2.training_timestamp == predictor.training_timestamp
        assert predictor2.validation_mae == predictor.validation_mae and predictor2.validation_bias == predictor.validation_bias
        assert predictor2.predict(load_data, temp_minutes=temp_data) == predictor.predict(load_data, temp_minutes=temp_data), "Predictions should match after load"

        np.savez(model_path, metadata_json='{"model_version": %d}' % (MODEL_VERSION + 1))
        predictor3 = LoadRegressionPredictor()
        assert predictor3.load(model_path) is False and not predictor3.model_initialized, "Version mismatch should be refused"
        assert LoadRegressionPredictor().load(os.path.join(tmpdir, "missing.npz")) is False


def _test_cold_start():
    """Too little history neither trains nor predicts."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)

    predictor = LoadRegressionPredictor()
    assert predictor.train({}) is None, "No data should not train"
    short = _create_synthetic_load_data(n_days=MIN_HISTORY_DAYS - 2, now_utc=now_utc)
    assert predictor.train(short) is None, "Too little history should not train"
    assert not predictor.model_initialized and predictor.is_valid() == (False, "not_initialized")
    assert predictor.predict(short) == {}, "An untrained model should not predict"

    # Trained, but asked to predict from less than the 7 days the profile needs
    predictor.train(_create_synthetic_load_data(n_days=14, now_utc=now_utc))
    assert predictor.predict(_create_synthetic_load_data(n_days=3, now_utc=now_utc)) == {}


def _test_nan_inf_robustness():
    """NaN, Inf and None in the load and temperature data are treated as missing."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)
    load_data = _create_synthetic_load_data(n_days=14, now_utc=now_utc)
    temp_data = _create_synthetic_temp_data(n_days=14, now_utc=now_utc, forecast_hours=48)
    for index, bad in enumerate([float("nan"), float("inf"), float("-inf"), None] * 10):
        load_data[1000 + index * 300] = bad
        temp_data[2000 + index * 300] = bad
        temp_data[-(100 + index * 30)] = bad
    del load_data[5000]

    predictor = LoadRegressionPredictor()
    val_mae = predictor.train(load_data, temp_minutes=temp_data)
    assert val_mae is not None and np.isfinite(val_mae), f"Validation MAE must be finite, got {val_mae}"
    assert np.isfinite(predictor.weights).all(), "Weights must be finite"
    predictions = predictor.predict(load_data, temp_minutes=temp_data)
    assert len(predictions) == 48 * 60 // STEP_MINUTES and np.isfinite(list(predictions.values())).all(), "Predictions must be finite"


class _MockBase:
    """Minimal base required by ComponentBase and LoadMLComponent.initialize."""

    def __init__(self, config_root):
        self.prefix = "predbat"
        self.config_root = config_root
        self.now_utc = datetime.now(timezone.utc)
        self.midnight_utc = self.now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        self.minutes_now = (self.now_utc - self.midnight_utc).seconds // 60
        self.local_tz = timezone.utc
        self.args = {}
        self.log_messages = []
        self.prediction_started = False

    def log(self, msg):
        self.log_messages.append(msg)

    def get_arg(self, key, default=None, indirect=True, combine=False, attribute=None, index=None, domain=None, can_override=True, required_unit=None):
        return {
            "load_today": ["sensor.load_today"],
            "load_power": None,
            "car_charging_energy": None,
            "load_scaling": 1.0,
            "car_charging_energy_scale": 1.0,
        }.get(key, default)


def _test_component_selects_model():
    """load_ml_model picks the predictor class, a model file per type and the minimum history; an unknown value falls back to the MLP."""
    from load_ml_component import LoadMLComponent

    with tempfile.TemporaryDirectory() as tmpdir:
        mlp = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True)
        assert mlp.ml_model == "mlp" and isinstance(mlp.predictor, LoadPredictor), "The MLP should be the default"
        assert os.path.basename(mlp.model_filepath) == "predbat_ml_model.npz" and mlp.ml_min_days == 1

        regression = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True, load_ml_model="Regression")
        assert regression.ml_model == "regression" and isinstance(regression.predictor, LoadRegressionPredictor)
        assert os.path.basename(regression.model_filepath) == "predbat_ml_regression.npz" and regression.ml_min_days == MIN_HISTORY_DAYS
        assert regression.database_filepath == mlp.database_filepath, "Both models should share the load history database"

        base = _MockBase(config_root=tmpdir)
        unknown = LoadMLComponent(base, load_ml_enable=True, load_ml_model="transformer")
        assert unknown.ml_model == "mlp" and isinstance(unknown.predictor, LoadPredictor)
        assert any("load_ml_model 'transformer'" in msg for msg in base.log_messages), "An unknown model should be logged"

        # A saved regression is picked up by the regression and left alone by the MLP, so switching back and forth keeps both
        np.random.seed(42)
        predictor = LoadRegressionPredictor()
        predictor.train(_create_synthetic_load_data(n_days=14))
        predictor.save(os.path.join(tmpdir, "predbat_ml_regression.npz"))
        regression = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True, load_ml_model="regression")
        assert regression.model_valid and regression.last_train_time == predictor.training_timestamp, "The saved regression should be loaded"
        mlp = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True)
        assert not mlp.model_valid and not mlp.initial_training_done, "The MLP should not load the regression's model"
        assert os.path.exists(os.path.join(tmpdir, "predbat_ml_regression.npz")), "Starting the MLP should not touch the regression's model file"


def _test_component_end_to_end():
    """The component fits the regression, predicts from it and publishes its weights and breakdown, with none of the MLP's attributes."""
    from load_ml_component import LoadMLComponent

    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _MockBase(config_root=tmpdir)
        component = LoadMLComponent(base, load_ml_enable=True, load_ml_model="regression", load_ml_database_days=0)
        component.load_data = _create_synthetic_load_data(n_days=14, now_utc=base.now_utc)
        component.temperature_data = _create_synthetic_temp_data(n_days=14, now_utc=base.now_utc, forecast_hours=48)
        component.load_data_age_days = 14
        component.data_ready = True

        asyncio.run(component._do_training(is_initial=True))
        assert component.model_status == "active" and component.model_valid, component.model_status
        assert os.path.exists(os.path.join(tmpdir, "predbat_ml_regression.npz")), "The fitted model should be saved"
        assert not os.path.exists(os.path.join(tmpdir, "predbat_ml_model.npz")), "The MLP's model file should not be written"

        predictions = component._get_predictions(base.now_utc, base.midnight_utc)
        assert len(predictions) == 48 * 60 // STEP_MINUTES, len(predictions)
        assert len(component.current_breakdown) == 96, "There should be a breakdown for every 30-minute slot"

        component.dashboard_item = Mock()
        component._publish_entity()
        published = {call.args[0]: call.kwargs["attributes"] for call in component.dashboard_item.call_args_list}
        forecast = published["sensor.predbat_load_ml_forecast"]
        stats = published["sensor.predbat_load_ml_stats"]
        assert len(forecast["results"]) == len(predictions) and len(forecast["breakdown"]) == 96
        assert stats["model"] == "regression" and stats["model_version"] == MODEL_VERSION
        assert sorted(stats["weights"]) == sorted(TERMS)
        for mlp_only in ("rollout_mae_kwh", "pattern_mae_kwh", "epochs_trained"):
            assert mlp_only not in stats, f"{mlp_only} belongs to the MLP"

        # With less history than the regression needs, run() stops at insufficient_data rather than training
        short = LoadMLComponent(_MockBase(config_root=None), load_ml_enable=True, load_ml_model="regression", load_ml_database_days=0)
        short.data_ready = True
        short.load_data_age_days = MIN_HISTORY_DAYS - 1

        async def no_fetch():
            return None

        short._do_fetch = no_fetch
        assert asyncio.run(short.run(seconds=0, first=True))
        assert short.model_status == "insufficient_data", short.model_status
