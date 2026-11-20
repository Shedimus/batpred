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

from load_regression import LoadRegressionPredictor, MODEL_VERSION, MIN_HISTORY_DAYS, STEP_MINUTES, TERMS, HEAT_TERMS, HEATING_BASE_TEMP, COP_REFERENCE_TEMP, relative_cop, heat_input
from load_predictor import LoadPredictor


def test_load_regression(my_predbat=None):
    """
    Test suite for LoadReg, the linear regression that load_ml_model: regression selects.

    Tests:
    - The regression recovering known weights, and its per-term breakdown
    - The heat input curve, the heat blocks by time of day, and dropping a negative heat weight
    - The trimmed time-of-day profile
    - Forecast format, with and without temperature data
    - Model save/load with version check
    - Cold start and gaps in the data
    - LoadMLComponent picking the model, keeping a model file per type, and training, predicting and publishing with it
    - load_ml_shadow running the other model without the plan using it
    """
    sub_tests = [
        ("training_recovers_weights", _test_training_recovers_weights, "Fit recovers known weights and explain() adds up"),
        ("heat_input_curve", _test_heat_input_curve, "Heat input is zero above the base and rises faster than degrees as it gets colder"),
        ("heat_blocks", _test_heat_blocks, "Heating in one part of the day is learned by that block's weight, placed by local time"),
        ("negative_heat_dropped", _test_negative_heat_dropped, "A heat weight that comes out negative is dropped and the rest refitted"),
        ("profile_ignores_spike", _test_profile_ignores_one_off_spike, "Profile leaves out a one-off spike but keeps the day's energy"),
        ("prediction", _test_prediction, "End-to-end prediction"),
        ("prediction_without_temperature", _test_prediction_without_temperature, "Prediction with no temperature data, and with no temperature forecast"),
        ("model_persistence", _test_model_persistence, "Model save/load with version check"),
        ("cold_start", _test_cold_start, "Cold start with insufficient data"),
        ("nan_inf_robustness", _test_nan_inf_robustness, "Handles NaN, Inf, and None in its inputs"),
        ("component_selects_model", _test_component_selects_model, "load_ml_model picks the predictor, its model file and its minimum history"),
        ("component_end_to_end", _test_component_end_to_end, "LoadMLComponent trains, predicts and publishes with the regression"),
        ("shadow_picks_other_model", _test_shadow_picks_other_model, "load_ml_shadow runs the model that is not selected, with its own model file"),
        ("shadow_never_reaches_plan", _test_shadow_never_reaches_plan, "The shadow trains, forecasts and publishes on its own sensors without touching the main forecast"),
        ("shadow_in_run_cycle", _test_shadow_in_run_cycle, "run() handles the shadow only after the main forecast is published"),
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


def _create_synthetic_temp_data(n_days=7, now_utc=None, forecast_hours=48, offset=0.0):
    """Synthetic temperature, history (positive minutes) and forecast (negative minutes): 2 to 10 C over the day plus a 3-day swing, plus offset."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    def temp_at(minute):
        dt = now_utc - timedelta(minutes=minute)
        hours_since_peak = (dt.hour + dt.minute / 60.0 - 13.0) % 24.0
        return offset + 6.0 + 4.0 * np.cos(2 * np.pi * hours_since_peak / 24.0) + 0.5 * np.sin(2 * np.pi * abs(minute) / (24 * 60) / 3.0)

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


def _minutes_now(now_utc):
    """Local minutes since midnight, the tests' clock being UTC."""
    return now_utc.hour * 60 + now_utc.minute


def _test_training_recovers_weights():
    """Load built as a daily shape plus 0.03 kWh per unit of heat input per 30 minutes is close to profile x 1 + 0.03 x each heat block."""
    now_utc = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    temp_data = _create_synthetic_temp_data(n_days=21, now_utc=now_utc, forecast_hours=48)

    def load_at(minute):
        """Energy in the 5 minutes that ended this many minutes ago (negative = future)."""
        return (0.3 + 0.2 * np.sin(2 * np.pi * minute / 1440.0)) / 6 + 0.005 * heat_input(temp_data[minute])

    load_data = {minute: load_at(minute) for minute in temp_data if minute >= 0}

    predictor = LoadRegressionPredictor()
    val_mae = predictor.train(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc))
    # Not exact: the profile leaves out the coldest day at each time of day and is then rescaled
    assert val_mae is not None and val_mae < 2e-3, f"Holdout error should be close to zero on data the model fits, got {val_mae}"
    expected = {"base": 0.0, "profile": 1.0, "heat_night": 0.03, "heat_day": 0.03, "heat_evening": 0.03, "heat_daily": 0.0, "last_2h": 0.0, "last_24h": 0.0}
    tolerance = [0.01 if term.startswith("heat_") else 0.05 for term in TERMS]
    assert np.all(np.abs(predictor.weights - [expected[term] for term in TERMS]) < tolerance), f"Weights {dict(zip(TERMS, predictor.weights))}"
    assert predictor.weights[TERMS.index("profile")] == 1.0, "The profile weight is fixed at 1"

    predictions = predictor.predict(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc))
    breakdown = predictor.explain(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc))
    assert sorted(breakdown) == list(range(0, 48 * 60, 30)), "One breakdown per 30-minute slot"
    cumulative = 0
    for minute, terms in breakdown.items():
        expected = sum(load_at(-(minute + step)) for step in range(5, 35, 5))
        assert abs(terms["total"] - expected) < 3e-3, f"Slot at +{minute} min: forecast {terms['total']}, truth {expected}"
        assert abs(sum(terms[term] for term in TERMS) - terms["total"]) < 1e-3, f"Terms should add up to the total: {terms}"
        cumulative += terms["total"]
        assert abs(predictions[minute + 25] - cumulative) < 1e-2, "Forecast should be the running sum of the slot totals"
    for term in ["heat_night", "heat_day", "heat_evening"]:
        assert any(abs(terms[term]) > 0.005 for terms in breakdown.values()), f"Temperature forecast should move the forecast through {term}"


def _test_heat_input_curve():
    """Heat input is heating degrees divided by a COP that falls as it gets colder: none above the base, more than degrees below 7 C."""
    assert abs(relative_cop(COP_REFERENCE_TEMP) - 1.0) < 1e-12
    assert heat_input(HEATING_BASE_TEMP) == 0 and heat_input(20.0) == 0, "No heat input above the heating base temperature"
    temps = np.arange(HEATING_BASE_TEMP, -15.0, -0.5)
    heat = heat_input(temps)
    assert np.all(np.diff(heat) > 0), "Heat input should rise all the way down"
    assert np.all(np.diff(heat) >= np.diff(heat)[0] - 1e-9), "Each degree colder should add at least as much as the last"
    # Close to rated heat pumps: the COP at 2 C and -7 C is about 0.8 and 0.55 of that at 7 C
    assert 0.75 < relative_cop(2.0) < 0.85 and 0.5 < relative_cop(-7.0) < 0.65, f"COP ratios {relative_cop(2.0)}, {relative_cop(-7.0)}"


def _test_heat_blocks():
    """Heating that only runs in the evening is learned by heat_evening; the slots are placed by the local time passed in."""
    now_utc = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    temp_data = _create_synthetic_temp_data(n_days=21, now_utc=now_utc, forecast_hours=48)

    def load_at(minute):
        hour = (now_utc - timedelta(minutes=minute + STEP_MINUTES)).hour  # Hour the 5 minutes started in
        evening = 0.01 * heat_input(temp_data[minute]) if 16 <= hour < 22 else 0.0
        return (0.3 + 0.2 * np.sin(2 * np.pi * minute / 1440.0)) / 6 + evening

    load_data = {minute: load_at(minute) for minute in temp_data if minute >= 0}
    predictor = LoadRegressionPredictor()
    assert predictor.train(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc)) is not None
    weights = dict(zip(TERMS, predictor.weights))
    assert abs(weights["heat_evening"] - 0.06) < 0.01, f"Weights {weights}"
    assert abs(weights["heat_night"]) < 0.01 and abs(weights["heat_day"]) < 0.01, f"Weights {weights}"

    # Told it is six hours later than it is, the evening heating lands in the wrong block and fits worse
    shifted = LoadRegressionPredictor()
    assert shifted.train(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc) + 360) is not None
    assert shifted.validation_mae > 2 * predictor.validation_mae, f"Wrong clock MAE {shifted.validation_mae} vs {predictor.validation_mae}"


def _test_negative_heat_dropped():
    """Using less energy the colder it gets is not heating: the heat terms are dropped rather than fitted negative."""
    now_utc = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    temp_data = _create_synthetic_temp_data(n_days=21, now_utc=now_utc, forecast_hours=48)
    load_data = {minute: max((0.3 + 0.2 * np.sin(2 * np.pi * minute / 1440.0)) / 6 - 0.002 * heat_input(temp), 0.0) for minute, temp in temp_data.items() if minute >= 0}

    predictor = LoadRegressionPredictor()
    assert predictor.train(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc)) is not None
    weights = dict(zip(TERMS, predictor.weights))
    assert all(weights[term] >= 0 for term in HEAT_TERMS), f"No heat weight should be negative: {weights}"
    assert predictor.heat_dropped, "At least one heat term should have been dropped"
    assert all(weights[term] == 0.0 for term in predictor.heat_dropped), f"Dropped terms should have zero weight: {weights}"
    breakdown = predictor.explain(load_data, temp_minutes=temp_data, minutes_now=_minutes_now(now_utc))
    assert all(abs(sum(terms[term] for term in TERMS) - terms["total"]) < 1e-3 for terms in breakdown.values() if terms["total"] > 0), "Terms should add up to the total"


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
    """With no temperature at all the heat weights are zero; with no forecast the heat terms are."""
    np.random.seed(42)
    now_utc = datetime.now(timezone.utc)
    load_data = _create_synthetic_load_data(n_days=14, now_utc=now_utc)

    predictor = LoadRegressionPredictor()
    assert predictor.train(load_data) is not None
    assert all(abs(predictor.weights[TERMS.index(term)]) < 1e-9 for term in HEAT_TERMS), "No temperature data should give no heat weights"
    assert len(predictor.predict(load_data)) == 48 * 60 // STEP_MINUTES

    predictor = LoadRegressionPredictor()
    history_only = _create_synthetic_temp_data(n_days=14, now_utc=now_utc, forecast_hours=0)
    assert predictor.train(load_data, temp_minutes=history_only) is not None
    breakdown = predictor.explain(load_data, temp_minutes=history_only)
    # heat_daily still sees the known half of the 24 hours around the first slots, so only the slot terms are checked
    assert len(breakdown) == 96 and all(terms[term] == 0 for terms in breakdown.values() for term in ["heat_night", "heat_day", "heat_evening"]), "A missing temperature forecast should leave the slot heat terms out"


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


def _test_shadow_picks_other_model():
    """The shadow is whichever model is not selected, saves to that model's own file and is off by default."""
    from load_ml_component import LoadMLComponent

    with tempfile.TemporaryDirectory() as tmpdir:
        assert LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True).shadow is None, "Shadow should be off by default"

        component = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True, load_ml_shadow=True)
        assert component.shadow.model == "regression" and isinstance(component.shadow.predictor, LoadRegressionPredictor)
        assert os.path.basename(component.shadow.filepath) == "predbat_ml_regression.npz" and isinstance(component.predictor, LoadPredictor)

        component = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True, load_ml_model="regression", load_ml_shadow=True)
        assert component.shadow.model == "mlp" and isinstance(component.shadow.predictor, LoadPredictor)
        assert os.path.basename(component.shadow.filepath) == "predbat_ml_model.npz" and component.shadow.status == "not_initialized"

        # A saved model is picked up by the shadow on restart, just as by the main model
        np.random.seed(42)
        predictor = LoadRegressionPredictor()
        predictor.train(_create_synthetic_load_data(n_days=14))
        predictor.save(os.path.join(tmpdir, "predbat_ml_regression.npz"))
        component = LoadMLComponent(_MockBase(config_root=tmpdir), load_ml_enable=True, load_ml_shadow=True)
        assert component.shadow.status == "active" and component.shadow.last_train_time == predictor.training_timestamp


def _test_shadow_never_reaches_plan():
    """The shadow regression trains, forecasts and publishes on shadow sensors only; the MLP's state, file and sensors are untouched."""
    from load_ml_component import LoadMLComponent

    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _MockBase(config_root=tmpdir)
        component = LoadMLComponent(base, load_ml_enable=True, load_ml_shadow=True, load_ml_database_days=0)
        component.load_data = _create_synthetic_load_data(n_days=14, now_utc=base.now_utc)
        component.temperature_data = _create_synthetic_temp_data(n_days=14, now_utc=base.now_utc, forecast_hours=48)
        component.load_data_age_days = 14
        component.data_ready = True
        component.last_data_fetch = base.now_utc
        component.dashboard_item = Mock()

        # Held back on the first run, as the main model's training is
        asyncio.run(component._run_shadow(first=True))
        assert component.shadow.last_train_time is None and component.shadow.status == "not_initialized"

        component.dashboard_item.reset_mock()
        asyncio.run(component._run_shadow(first=False))
        shadow = component.shadow
        assert shadow.status == "active" and shadow.last_train_time is not None, shadow.status
        assert len(shadow.predictions) == 48 * 60 // STEP_MINUTES and len(shadow.breakdown) == 96
        assert os.path.exists(os.path.join(tmpdir, "predbat_ml_regression.npz")), "The shadow's model should be saved"
        assert not os.path.exists(os.path.join(tmpdir, "predbat_ml_model.npz")), "The main model's file should not be written"

        # Nothing of the main model changed: fetch.py reads the plan's forecast from sensor.predbat_load_ml_forecast only
        assert component.current_predictions == {} and component.current_breakdown == {}
        assert component.model_status == "not_initialized" and not component.model_valid and component.last_train_time is None
        assert not component.predictor.model_initialized, "The main MLP should not have been trained"
        published = {call.args[0]: call.kwargs for call in component.dashboard_item.call_args_list}
        assert sorted(published) == ["sensor.predbat_load_ml_shadow_forecast", "sensor.predbat_load_ml_shadow_stats"], sorted(published)
        forecast = published["sensor.predbat_load_ml_shadow_forecast"]
        stats = published["sensor.predbat_load_ml_shadow_stats"]["attributes"]
        assert forecast["state"] == "active" and len(forecast["attributes"]["results"]) == len(shadow.predictions) and len(forecast["attributes"]["breakdown"]) == 96
        assert stats["model"] == "regression" and sorted(stats["weights"]) == sorted(TERMS) and stats["mae_kwh"] is not None

        # Within the retrain interval it forecasts again without refitting
        trained_at = shadow.last_train_time
        asyncio.run(component._run_shadow(first=False))
        assert shadow.last_train_time == trained_at and shadow.status == "active"

        # Too little history for the regression: no training, an empty forecast and a status that says why
        short = LoadMLComponent(_MockBase(config_root=None), load_ml_enable=True, load_ml_shadow=True, load_ml_database_days=0)
        short.load_data = _create_synthetic_load_data(n_days=5)
        short.load_data_age_days = 5
        short.data_ready = True
        short.dashboard_item = Mock()
        asyncio.run(short._run_shadow(first=False))
        assert short.shadow.status == "insufficient_data" and short.shadow.predictions == {}
        states = {call.args[0]: call.kwargs["state"] for call in short.dashboard_item.call_args_list}
        assert states["sensor.predbat_load_ml_shadow_forecast"] == "insufficient_data", states


def _test_shadow_in_run_cycle():
    """run() handles the shadow after publishing the main forecast, and not at all when the shadow is off."""
    from unittest.mock import AsyncMock, MagicMock
    from load_ml_component import LoadMLComponent

    for shadow_on in (True, False):
        base = _MockBase(config_root=None)
        component = LoadMLComponent(base, load_ml_enable=True, load_ml_model="regression", load_ml_shadow=shadow_on, load_ml_database_days=0)
        component.data_ready = True
        component.load_data_age_days = 14
        component.initial_training_done = True
        component.last_train_time = base.now_utc
        order = MagicMock()
        component._do_fetch = AsyncMock()
        component._get_predictions = order.predict
        component._publish_entity = order.publish
        component._run_shadow = AsyncMock(side_effect=lambda first: order.shadow(first))

        assert asyncio.run(component.run(seconds=1800, first=False))
        calls = [call[0] for call in order.mock_calls]
        expected = ["predict", "publish", "shadow"] if shadow_on else ["predict", "publish"]
        assert calls == expected, (shadow_on, calls)
