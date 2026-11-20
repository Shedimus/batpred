# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# LoadReg - linear regression load predictor
# -----------------------------------------------------------------------------
# fmt off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init


"""LoadReg: linear regression for household load forecasting, an alternative to the LoadML MLP in load_predictor.py.

One ordinary least-squares model forecasts the next 48 hours in 30-minute slots.
The load in a future slot is the profile plus a base amount and six weighted inputs:

    profile       typical load at the same time of day over the 7 days before the forecast: the mean without the
                  highest day, scaled so that a day of it still adds up to the plain 7-day mean. Its weight is
                  fixed at 1, so the forecast follows the level of the last week rather than being pulled towards
                  the average of the whole history (which under-forecasts as the weather turns colder)
    heat_night    heat input in the slot (see below), minus its 7-day mean for that time of day, for slots
    heat_day      starting 22:00-07:00, 07:00-16:00 and 16:00-22:00 local time. A weight per block lets
    heat_evening  heating schedules (a night setback, an evening boost) show through
    heat_daily    heating degrees of the mean temperature over the 24 hours around the slot, minus its 7-day
                  mean: cold that lasts all day costs more than a cold hour, as the house and any buffer cool
    last_2h       how far the last 2 hours ran above the profile, fading out over the next few hours
    last_24h      mean load over the last 24 hours minus the mean over the last 7 days

Heat input is heating degrees (below 12 C) divided by how efficient a heat pump is at that temperature, relative
to 7 C. The efficiency is generic, not any one model's: Carnot's limit along a typical weather-compensation curve,
which matches manufacturers' rated COPs from 7 C down to about -2 C and is a little gentler below (see
relative_cop()). It rises faster than degrees as it gets colder, as a heat pump's power does; for a house heated
some other way the fitted weights simply come out smaller. A heat weight that comes out negative is dropped, so a
house with no temperature-driven load is left with the profile.

Every forecast value can be read back as weight x input for each term (see explain()).
"""

import numpy as np
import json
import os
from datetime import datetime, timezone

MODEL_VERSION = 3  # Saved models of another version are refitted
STEP_MINUTES = 5  # Resolution of the input data and of the forecast returned
SLOT_MINUTES = 30  # Resolution the model works at
SLOT_STEPS = SLOT_MINUTES // STEP_MINUTES
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES
PROFILE_DAYS = 7  # Days the time-of-day profile is built from
PREDICT_HORIZON = 2 * SLOTS_PER_DAY  # 48 hours of 30-minute slots
RECENT_SLOTS = 4  # The "last 2 hours"
RECENT_FADE_SLOTS = 4  # The last 2 hours term decays by 1/e every 2 hours ahead
HEATING_BASE_TEMP = 12.0  # No heating load is assumed above this outside temperature (C)
COP_REFERENCE_TEMP = 7.0  # Heat input is relative to a heat pump's efficiency at this outside temperature (C)
FLOW_TEMP_MIN = 35.0  # Weather-compensated flow temperature at HEATING_BASE_TEMP and above (C)
FLOW_TEMP_SLOPE = 0.75  # Rise in flow temperature per degree colder outside
FLOW_TEMP_MAX = 55.0  # Highest flow temperature (C)
DAILY_SLOTS = SLOTS_PER_DAY + 1  # The heat_daily mean covers the slot and 12 hours either side
# Local (start, end) minute of day of each heat block; night wraps past midnight
HEAT_BLOCKS = {"heat_night": (22 * 60, 7 * 60), "heat_day": (7 * 60, 16 * 60), "heat_evening": (16 * 60, 22 * 60)}
MIN_HISTORY_DAYS = PROFILE_DAYS + 3  # 7 days of profile, 2 days held out for validation and 1 day to fit on
TERMS = ["base", "profile", "heat_night", "heat_day", "heat_evening", "heat_daily", "last_2h", "last_24h"]
FIXED_WEIGHTS = {"profile": 1.0}  # Terms whose weight is set rather than fitted
HEAT_TERMS = [term for term in TERMS if term.startswith("heat_")]  # Dropped from the fit if their weight comes out negative


def nanmean(values, axis):
    """Mean ignoring NaN, giving NaN (without numpy's warning) where there is nothing to average."""
    count = np.sum(~np.isnan(values), axis=axis)
    return np.where(count > 0, np.nansum(values, axis=axis) / np.maximum(count, 1), np.nan)


def rolling_nanmean(values, window):
    """Mean of values[i : i + window] for every i that has a full window."""
    return nanmean(np.lib.stride_tricks.sliding_window_view(values, window), axis=1)


def relative_cop(temperature):
    """
    How efficient a heat pump is at an outside temperature, relative to COP_REFERENCE_TEMP.

    Carnot's limit (flow temperature in kelvin over the lift) along a typical weather-compensation curve. Real heat
    pumps reach a roughly constant share of it, which cancels in the ratio. Against EN 14511 ratings (Vaillant
    aroTHERM pro, for example) it is within a few percent from 7 C down to -2 C, and gentler below -5 C where
    defrosting and capacity limits make each model different.
    """

    def carnot(t):
        """Carnot COP at outside temperature t with a weather-compensated flow temperature."""
        flow = np.clip(FLOW_TEMP_MIN + FLOW_TEMP_SLOPE * (HEATING_BASE_TEMP - t), FLOW_TEMP_MIN, FLOW_TEMP_MAX)
        return (flow + 273.15) / np.maximum(flow - t, 5.0)

    return carnot(temperature) / carnot(COP_REFERENCE_TEMP)


def heat_input(temperature):
    """Heating degrees below HEATING_BASE_TEMP divided by relative_cop(): roughly what a heat pump draws, in degrees at 7 C."""
    return np.maximum(HEATING_BASE_TEMP - temperature, 0) / relative_cop(temperature)


class LoadRegressionPredictor:
    """
    Linear regression load predictor.

    All data is keyed by minutes relative to now at 5-minute steps: key m is the 5 minutes that
    ended m minutes ago, negative keys are in the future (-5 is the next 5 minutes).
    """

    def __init__(self, log_func=None, max_load_kw=23.0):
        """
        Initialise the load predictor.

        Args:
            log_func: Logging function (defaults to print)
            max_load_kw: Maximum load in kW for clipping predictions
        """
        self.log = log_func if log_func else print
        self.max_load_kw = max_load_kw

        self.weights = None  # One per entry of TERMS, set by train()
        self.heat_dropped = []  # Heat terms the last train() dropped because their weight came out negative
        self.training_timestamp = None
        self.validation_mae = None  # kWh per 30 minutes over the 48 hours held out
        self.validation_bias = None  # Mean (predicted - actual) over the same 48 hours; + = over-predicting
        self.model_initialized = False

    def _slots(self, data, first_slot, last_slot, total):
        """
        Turn {minute: value} at 5-minute steps into an array of 30-minute slots.

        Slot s covers the 30 minutes that ended 30 * s minutes ago, so slot 0 is the newest
        history and slot -1 the first future slot.

        Args:
            data: Dict of {minute: value}
            first_slot: First slot to return
            last_slot: One past the last slot to return
            total: True to sum the steps (energy, NaN unless all six are present), False to average those present

        Returns:
            Array of last_slot - first_slot values, NaN where there is no data
        """
        steps = np.full((last_slot - first_slot) * SLOT_STEPS, np.nan)
        for index in range(len(steps)):
            value = data.get(first_slot * SLOT_MINUTES + index * STEP_MINUTES)
            try:
                steps[index] = float(value)
            except (TypeError, ValueError):
                pass
        steps[~np.isfinite(steps)] = np.nan
        steps = steps.reshape(-1, SLOT_STEPS)
        return np.where(np.isnan(steps).any(axis=1), np.nan, np.nansum(steps, axis=1)) if total else nanmean(steps, axis=1)

    def _inputs(self, load_minutes, temp_minutes, origins, minutes_now=0):
        """
        Build the model inputs for forecasts made at the given origins.

        An origin is how many slots ago the forecast is made, 0 being now. Each origin gives
        PREDICT_HORIZON rows, one per 30-minute slot ahead of it.

        Args:
            load_minutes: Dict of {minute: kwh_per_5min}, history only
            temp_minutes: Dict of {minute: temperature_celsius}, history and forecast
            origins: Array of origins, each needing PROFILE_DAYS of load history behind it
            minutes_now: Local minutes since midnight at minute 0 of the data, to place each slot in its heat block

        Returns:
            Tuple of (inputs, actual): inputs has shape (origins, PREDICT_HORIZON, terms) with NaN where history is
            missing, and actual is the load that happened in each slot (NaN when not known yet)
        """
        slots = max(load_minutes) // SLOT_MINUTES + 1
        load = self._slots(load_minutes, 0, slots, total=True)
        temperature = self._slots(temp_minutes or {}, -PREDICT_HORIZON, slots, total=False)

        def same_time(values):
            """Values at each slot stacked with those at the same slot on each of the PROFILE_DAYS - 1 days before it."""
            span = len(values) - (PROFILE_DAYS - 1) * SLOTS_PER_DAY
            return np.stack([values[day * SLOTS_PER_DAY : day * SLOTS_PER_DAY + span] for day in range(PROFILE_DAYS)])

        # shortcut: "same time of day" is whole days back on the clock that does not change, so the profile sits an hour
        # off local habits for the week after a clock change; key the slots by local time if that week matters
        days = same_time(load)
        count = np.sum(~np.isnan(days), axis=0)
        mean = nanmean(days, axis=0)
        # The highest day at each time of day is left out so a one-off spike does not become part of the shape, then each
        # origin's profile is scaled so that its day still adds up to the plain mean and the spikes' energy is not lost
        trimmed = np.where(count >= 3, (np.nansum(days, axis=0) - np.fmax.reduce(days, axis=0)) / np.maximum(count - 1, 1), mean)
        trimmed_day = rolling_nanmean(trimmed, SLOTS_PER_DAY)
        scale = np.where(trimmed_day > 0, rolling_nanmean(mean, SLOTS_PER_DAY) / np.where(trimmed_day > 0, trimmed_day, 1), 1.0)[origins]
        week = PROFILE_DAYS * SLOTS_PER_DAY
        last_24h = (rolling_nanmean(load, SLOTS_PER_DAY)[: slots - week + 1] - rolling_nanmean(load, week))[origins]
        last_2h = rolling_nanmean(load, RECENT_SLOTS)[origins] - scale * rolling_nanmean(trimmed, RECENT_SLOTS)[origins]

        ahead = np.arange(PREDICT_HORIZON)
        target = origins[:, None] - 1 - ahead[None, :]  # Slot being forecast, negative when it has not happened yet
        newest = origins[:, None] + SLOTS_PER_DAY - 1 - ahead[None, :] % SLOTS_PER_DAY  # Newest history slot at the target's time of day

        def above_last_week(values):
            """Values in the target slot minus their mean at that time of day over the last week.
            No temperature known for the slot or for its time of day in the last week means no adjustment."""
            return np.nan_to_num(values[target + PREDICT_HORIZON] - nanmean(same_time(values[PREDICT_HORIZON:]), axis=0)[newest])

        # Mean temperature over the 24 hours centred on each slot, from whatever of it is known
        half = DAILY_SLOTS // 2
        daily_temperature = rolling_nanmean(np.concatenate([np.full(half, np.nan), temperature, np.full(half, np.nan)]), DAILY_SLOTS)

        heat = above_last_week(heat_input(temperature))
        # shortcut: like the profile, slots are placed on the clock as it is now, so across a clock change they
        # sit in their heat block an hour off
        start = (minutes_now - SLOT_MINUTES * (target + 1)) % (24 * 60)  # Local minute of day each target slot starts

        inputs = np.empty(target.shape + (len(TERMS),))
        inputs[..., TERMS.index("base")] = 1.0
        inputs[..., TERMS.index("profile")] = trimmed[newest] * scale[:, None]
        for term, (block_start, block_end) in HEAT_BLOCKS.items():
            in_block = ((start >= block_start) | (start < block_end)) if block_start > block_end else ((start >= block_start) & (start < block_end))
            inputs[..., TERMS.index(term)] = heat * in_block
        inputs[..., TERMS.index("heat_daily")] = above_last_week(np.maximum(HEATING_BASE_TEMP - daily_temperature, 0))
        inputs[..., TERMS.index("last_2h")] = last_2h[:, None] * np.exp(-ahead / RECENT_FADE_SLOTS)[None, :]
        inputs[..., TERMS.index("last_24h")] = last_24h[:, None]
        actual = np.where(target >= 0, load[np.maximum(target, 0)], np.nan)
        return inputs, actual

    def train(self, load_minutes, temp_minutes=None, minutes_now=0):
        """
        Fit the weights by least squares on every forecast the history allows.

        The newest 48 hours are first held out: the model is fitted without them, forecasts them
        from 48 hours back and is scored on the result, then refitted on everything.

        Args:
            load_minutes: Dict of {minute: kwh_per_5min}, history only
            temp_minutes: Dict of {minute: temperature_celsius}, history (forecast keys are ignored)
            minutes_now: Local minutes since midnight at minute 0 of the data

        Returns:
            Validation MAE in kWh per 30 minutes, or None if there is not enough history
        """
        slots = max(load_minutes) // SLOT_MINUTES + 1 if load_minutes else 0
        if slots < MIN_HISTORY_DAYS * SLOTS_PER_DAY:
            self.log("Warn: LoadReg Predictor: {:.1f} days of load history, need {} to train".format(slots / SLOTS_PER_DAY, MIN_HISTORY_DAYS))
            return None

        origins = np.arange(1, slots - PROFILE_DAYS * SLOTS_PER_DAY + 1)
        inputs, actual = self._inputs(load_minutes, temp_minutes, origins, minutes_now)
        usable = ~np.isnan(actual) & ~np.isnan(inputs).any(axis=-1)
        target = origins[:, None] - 1 - np.arange(PREDICT_HORIZON)[None, :]

        seen = usable & (target >= PREDICT_HORIZON)
        held_out = usable[PREDICT_HORIZON - 1]  # The forecast made 48 hours ago, origins start at 1
        if seen.sum() < 10 * len(TERMS) or not held_out.any():
            self.log("Warn: LoadReg Predictor: Not enough complete load history to train")
            return None

        fixed = [TERMS.index(term) for term in FIXED_WEIGHTS]
        fixed_weights = np.array([FIXED_WEIGHTS[term] for term in FIXED_WEIGHTS])

        def fit(rows):
            """
            Weights by least squares on the given rows, with the fixed weights set and any heat term whose
            weight comes out negative dropped (refitting without it), as a negative slope is noise.

            Returns:
                Tuple of (weights, heat terms dropped)
            """
            target_rows = actual[rows] - inputs[rows][:, fixed] @ fixed_weights
            free = [index for index in range(len(TERMS)) if index not in fixed]
            dropped = []
            while True:
                fitted = np.linalg.lstsq(inputs[rows][:, free], target_rows, rcond=None)[0]
                negative = [index for index, weight in zip(free, fitted) if TERMS[index] in HEAT_TERMS and weight < 0]
                if not negative:
                    break
                worst = min(negative, key=lambda index: fitted[free.index(index)])
                free.remove(worst)
                dropped.append(TERMS[worst])
            weights = np.zeros(len(TERMS))
            weights[fixed] = fixed_weights
            weights[free] = fitted
            return weights, dropped

        held_out_weights, _ = fit(seen)
        error = self._clip(inputs[PREDICT_HORIZON - 1][held_out] @ held_out_weights) - actual[PREDICT_HORIZON - 1][held_out]
        weights, self.heat_dropped = fit(usable)

        self.weights = weights
        self.validation_mae = float(np.mean(np.abs(error)))
        self.validation_bias = float(np.mean(error))
        self.training_timestamp = datetime.now(timezone.utc)
        self.model_initialized = True
        self.log(
            "LoadReg Predictor: Fitted on {} forecast slots: load per 30 min = {}{}; holdout MAE {:.4f} kWh, bias {:+.4f} kWh".format(
                int(usable.sum()),
                " + ".join("{:.4f} x {}".format(weight, term) if term != "base" else "{:.3f}".format(weight) for term, weight in zip(TERMS, self.weights)),
                " (dropped as negative: {})".format(", ".join(self.heat_dropped)) if self.heat_dropped else "",
                self.validation_mae,
                self.validation_bias,
            )
        )
        return self.validation_mae

    def _clip(self, energy):
        """Keep 30-minute energies between zero and the maximum load."""
        return np.clip(energy, 0, self.max_load_kw * SLOT_MINUTES / 60.0)

    def explain(self, load_minutes, temp_minutes=None, minutes_now=0):
        """
        Break the forecast from now into what each term adds.

        Args:
            load_minutes: Dict of {minute: kwh_per_5min}, history only
            temp_minutes: Dict of {minute: temperature_celsius}, history and forecast (negative keys)
            minutes_now: Local minutes since midnight at minute 0 of the data

        Returns:
            Dict of {minutes_ahead: {term: kwh, ..., "total": kwh}} for each 30-minute slot, the terms being
            weight x input and summing to the total unless it was clipped. Empty when there is no forecast.
        """
        if not self.model_initialized or self.weights is None:
            self.log("Warn: LoadReg Predictor: Model not trained, cannot predict")
            return {}
        if not load_minutes or max(load_minutes) // SLOT_MINUTES + 1 < PROFILE_DAYS * SLOTS_PER_DAY:
            self.log("Warn: LoadReg Predictor: Less than {} days of load history, cannot predict".format(PROFILE_DAYS))
            return {}

        inputs, _ = self._inputs(load_minutes, temp_minutes, np.array([0]), minutes_now)
        terms = inputs[0] * self.weights
        if np.isnan(terms).any():
            self.log("Warn: LoadReg Predictor: Gaps in the load history, cannot predict")
            return {}
        total = self._clip(terms.sum(axis=1))
        breakdown = {}
        for slot in range(PREDICT_HORIZON):
            breakdown[slot * SLOT_MINUTES] = {term: round(float(value), 4) for term, value in zip(TERMS, terms[slot])}
            breakdown[slot * SLOT_MINUTES]["total"] = round(float(total[slot]), 4)
        return breakdown

    def predict(self, load_minutes, temp_minutes=None, minutes_now=0):
        """
        Forecast the next 48 hours.

        Args:
            load_minutes: Dict of {minute: kwh_per_5min}, history only
            temp_minutes: Dict of {minute: temperature_celsius}, history and forecast (negative keys)
            minutes_now: Local minutes since midnight at minute 0 of the data

        Returns:
            Dict of {minute: cumulative_kwh} at 5-minute steps, each 30-minute slot being spread evenly
            over its six steps. Empty dict on failure.
        """
        result = {}
        cumulative = 0
        for minute, terms in self.explain(load_minutes, temp_minutes, minutes_now).items():
            for step in range(SLOT_STEPS):
                cumulative += terms["total"] / SLOT_STEPS
                result[minute + step * STEP_MINUTES] = round(cumulative, 4)
        return result

    def save(self, filepath):
        """
        Save model to file.

        Args:
            filepath: Path to save model
        """
        if not self.model_initialized:
            self.log("Warn: LoadReg Predictor: No model to save")
            return False

        try:
            metadata = {
                "model_version": MODEL_VERSION,
                "weights": dict(zip(TERMS, self.weights.tolist())),
                "training_timestamp": self.training_timestamp.isoformat() if self.training_timestamp else None,
                "validation_mae": self.validation_mae,
                "validation_bias": self.validation_bias,
            }
            np.savez(filepath, metadata_json=json.dumps(metadata))
            self.log("LoadReg Predictor: Model saved to {}".format(filepath))
            return True

        except Exception as e:
            self.log("Error: LoadReg Predictor: Failed to save model: {}".format(e))
            return False

    def load(self, filepath):
        """
        Load model from file.

        Args:
            filepath: Path to model file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                self.log("LoadReg Predictor: No saved model found at {}".format(filepath))
                return False

            with np.load(filepath, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))

            saved_version = metadata.get("model_version", 0)
            if saved_version != MODEL_VERSION:
                self.log("Warn: LoadReg Predictor: Model version mismatch (saved={}, current={}), retraining from scratch".format(saved_version, MODEL_VERSION))
                return False

            self.weights = np.array([metadata["weights"][term] for term in TERMS], dtype=float)
            if metadata.get("training_timestamp"):
                self.training_timestamp = datetime.fromisoformat(metadata["training_timestamp"])
            self.validation_mae = metadata.get("validation_mae")
            self.validation_bias = metadata.get("validation_bias")
            self.model_initialized = True

            self.log(
                "LoadReg Predictor: Model loaded from {} (trained {}, val_mae={:.4f}, val_bias={:+.4f})".format(
                    filepath, self.training_timestamp.strftime("%Y-%m-%d %H:%M") if self.training_timestamp else "unknown", self.validation_mae if self.validation_mae else 0, self.validation_bias if self.validation_bias is not None else 0
                )
            )
            return True

        except Exception as e:
            self.log("Error: LoadReg Predictor: Failed to load model: {}".format(e))
            return False

    def get_model_age_hours(self):
        """Get the age of the model in hours since last training."""
        if self.training_timestamp is None:
            return None

        age = datetime.now(timezone.utc) - self.training_timestamp
        return age.total_seconds() / 3600

    def is_valid(self, validation_threshold=2.0, max_age_hours=48):
        """
        Check if model is valid for predictions.

        Args:
            validation_threshold: Maximum acceptable validation MAE in kWh
            max_age_hours: Maximum model age in hours

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not self.model_initialized:
            return False, "not_initialized"

        if self.weights is None:
            return False, "no_weights"

        if self.validation_mae is not None and self.validation_mae > validation_threshold:
            return False, "validation_threshold"

        age_hours = self.get_model_age_hours()
        if age_hours is not None and age_hours > max_age_hours:
            return False, "stale"

        return True, None
