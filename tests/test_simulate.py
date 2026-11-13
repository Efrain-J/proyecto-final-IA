"""Tests for `energy.rl.simulate`: replaying a policy over a date range.

Offline: prices are passed in via `prices=`, policy files live in `tmp_path`.
Run with `uv run pytest tests/test_simulate.py`.
"""
from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy.rl.battery import get_preset
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, NoBattery
from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem, train_q_learning
from energy.rl.simulate import (
    DEFAULT_POLICIES_DIR,
    POLICY_IDS,
    SimulationResult,
    resolve_policy,
    simulate,
)

ETA = math.sqrt(0.9)
HOURLY_COLUMNS = [
    "timestamp", "price_cop_kwh", "action", "soc_level", "grid_kwh",
    "savings_cop", "cum_savings_cop",
]


class FixedPolicy:
    """Replays a fixed action list (padded with idle) whatever the frame says."""

    name = "fixed"

    def __init__(self, scripted: dict[int, int]):
        self.scripted = scripted

    def actions(self, frame, preset) -> np.ndarray:
        out = np.zeros(len(frame), dtype=np.int8)
        for hour, action in self.scripted.items():
            out[hour] = action
        return out


@pytest.fixture
def day_prices() -> pd.DataFrame:
    """Two days with price 100 + 10 * hour_of_day, so each hour has a distinct price."""
    ts = pd.date_range("2025-03-03", periods=48, freq="h")
    return pd.DataFrame({"timestamp": ts, "price_cop_kwh": 100.0 + 10.0 * ts.hour.to_numpy()})


D1 = date(2025, 3, 3)


def test_constants():
    assert POLICY_IDS == ("no_battery", "night_day", "perfect_foresight", "q_learning")
    assert str(DEFAULT_POLICIES_DIR).replace("\\", "/") == "artifacts/policies"


def test_savings_match_hand_computation(day_prices):
    policy = FixedPolicy({0: 1, 1: 1, 5: 2, 6: 2})
    result = simulate(policy, D1, D1, "2h", prices=day_prices)
    assert isinstance(result, SimulationResult)
    prices = day_prices["price_cop_kwh"].to_numpy()[:24]
    expected_grid = np.zeros(24)
    expected_grid[[0, 1]] = 100 / ETA
    expected_grid[[5, 6]] = -100 * ETA
    hourly = result.hourly
    assert list(hourly.columns) == HOURLY_COLUMNS
    assert len(hourly) == 24
    assert np.allclose(hourly["grid_kwh"], expected_grid)
    assert np.allclose(hourly["savings_cop"], -expected_grid * prices)

    s = result.summary
    assert s.policy == "fixed"
    assert s.preset_id == "2h"
    assert s.hours == 24
    assert s.invalid_actions == 0
    assert s.savings_cop == pytest.approx(float((-expected_grid * prices).sum()))
    assert s.bought_kwh == pytest.approx(2 * 100 / ETA)
    assert s.delivered_kwh == pytest.approx(2 * 100 * ETA)
    assert s.cycles == pytest.approx(1.0)  # delivered / (capacity * eta)


def test_cum_savings_is_running_total(day_prices):
    result = simulate(FixedPolicy({0: 1, 1: 1, 5: 2, 6: 2}), D1, D1, "2h", prices=day_prices)
    hourly = result.hourly
    assert np.allclose(hourly["cum_savings_cop"], hourly["savings_cop"].cumsum())
    assert hourly["cum_savings_cop"].iloc[-1] == pytest.approx(result.summary.savings_cop)


def test_soc_level_stays_in_bounds(day_prices):
    # Over-charge and over-discharge to push against both limits.
    scripted = {h: 1 for h in range(0, 6)} | {h: 2 for h in range(6, 12)}
    result = simulate(FixedPolicy(scripted), D1, D1, "2h", prices=day_prices)
    soc = result.hourly["soc_level"]
    assert soc.min() >= 0 and soc.max() <= get_preset("2h").duration_h


def test_invalid_actions_become_idle_and_are_counted(day_prices):
    # h0 discharge on empty (invalid), h1-h2 charge (ok), h3 charge when full (invalid).
    result = simulate(FixedPolicy({0: 2, 1: 1, 2: 1, 3: 1}), D1, D1, "2h", prices=day_prices)
    hourly = result.hourly
    assert result.summary.invalid_actions == 2
    assert hourly["action"].iloc[0] == 0 and hourly["grid_kwh"].iloc[0] == 0.0
    assert hourly["action"].iloc[3] == 0 and hourly["grid_kwh"].iloc[3] == 0.0
    assert hourly["grid_kwh"].iloc[1] == pytest.approx(100 / ETA)


def test_slices_requested_dates_only(day_prices):
    result = simulate(NoBattery(), date(2025, 3, 4), date(2025, 3, 4), "2h", prices=day_prices)
    assert result.summary.hours == 24
    assert result.hourly["timestamp"].iloc[0] == pd.Timestamp("2025-03-04 00:00")
    assert result.hourly["timestamp"].iloc[-1] == pd.Timestamp("2025-03-04 23:00")
    assert result.summary.start == date(2025, 3, 4)
    assert result.summary.end == date(2025, 3, 4)


def test_accepts_preset_object_and_adds_rel_price(day_prices):
    result = simulate(NoBattery(), D1, D1, get_preset("3h"), prices=day_prices)
    assert result.summary.preset_id == "3h"
    assert result.summary.savings_cop == pytest.approx(0.0)


def test_start_after_end_raises(day_prices):
    with pytest.raises(ValueError):
        simulate(NoBattery(), date(2025, 3, 4), D1, "2h", prices=day_prices)


@pytest.mark.parametrize(
    "start,end",
    [(date(2025, 2, 1), date(2025, 2, 2)), (date(2025, 3, 4), date(2025, 3, 5)),
     (date(2026, 1, 1), date(2026, 1, 2))],
    ids=["before-data", "partly-after-data", "after-data"],
)
def test_out_of_range_dates_raise(day_prices, start, end):
    with pytest.raises(ValueError):
        simulate(NoBattery(), start, end, "2h", prices=day_prices)


@pytest.mark.parametrize("policy_id", ["no_battery", "night_day", "perfect_foresight"])
def test_string_policy_ids_resolve(synthetic_prices, tmp_path, policy_id):
    prices = add_relative_price(synthetic_prices(1))
    result = simulate(
        policy_id, date(2025, 1, 6), date(2025, 1, 7), "2h", prices=prices, policies_dir=tmp_path
    )
    assert result.summary.policy == policy_id
    assert result.summary.hours == 48


def test_unknown_policy_id_raises(day_prices, tmp_path):
    with pytest.raises(ValueError):
        simulate("magic", D1, D1, "2h", prices=day_prices, policies_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_policy("magic", "2h", 2025, tmp_path)


def test_resolve_policy_falls_back_to_night_day_with_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        policy = resolve_policy("q_learning", "2h", 2025, tmp_path)
    assert isinstance(policy, NightDay) and policy.name == "night_day"
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_simulate_q_learning_without_file_falls_back(synthetic_prices, tmp_path, caplog):
    prices = add_relative_price(synthetic_prices(1))
    with caplog.at_level(logging.WARNING):
        result = simulate(
            "q_learning", date(2025, 1, 6), date(2025, 1, 7), "2h",
            prices=prices, policies_dir=tmp_path,
        )
    assert result.summary.policy == "night_day"
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_resolve_policy_loads_saved_q_table(synthetic_prices, tmp_path):
    train = add_relative_price(synthetic_prices(4, noise=0.05, seed=2))
    res = train_q_learning(train, get_preset("2h"), QConfig(max_epochs=3, patience=2))
    TabularPolicy(res.q, res.discretizer, "2h").save(policy_stem(tmp_path, "2h", 2025))
    policy = resolve_policy("q_learning", "2h", 2025, tmp_path)
    assert policy.name == "q_learning"
