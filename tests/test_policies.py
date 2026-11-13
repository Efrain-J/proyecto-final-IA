"""Tests for `energy.rl.policies`: NoBattery, NightDay and PerfectForesight baselines.

Offline. Savings are computed through `simulate` with the policy object. Run with
`uv run pytest tests/test_policies.py`.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy.rl.battery import Action, get_preset, step, valid_mask
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, NoBattery, PerfectForesight
from energy.rl.simulate import simulate

ETA = math.sqrt(0.9)


def _frame(prices, start: str = "2025-01-06") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(prices), freq="h")
    return add_relative_price(pd.DataFrame({"timestamp": ts, "price_cop_kwh": np.asarray(prices, float)}))


def _savings(policy, frame: pd.DataFrame, preset_id: str = "2h") -> float:
    start = frame["timestamp"].iloc[0].date()
    end = frame["timestamp"].iloc[-1].date()
    return simulate(policy, start, end, preset_id, prices=frame).summary.savings_cop


def test_policy_names():
    assert NoBattery().name == "no_battery"
    assert NightDay().name == "night_day"
    assert PerfectForesight().name == "perfect_foresight"


def test_no_battery_always_idle(synthetic_prices):
    frame = add_relative_price(synthetic_prices(1))
    acts = NoBattery().actions(frame, get_preset("2h"))
    assert len(acts) == len(frame)
    assert acts.dtype == np.int8
    assert (acts == Action.IDLE).all()


@pytest.mark.parametrize(
    "preset_id,charge_hours,discharge_hours",
    [("2h", [2, 3], [18, 19]), ("3h", [2, 3, 4], [18, 19, 20])],
)
def test_night_day_hours(synthetic_prices, preset_id, charge_hours, discharge_hours):
    frame = add_relative_price(synthetic_prices(1)).iloc[:24].reset_index(drop=True)
    acts = NightDay().actions(frame, get_preset(preset_id))
    assert acts.dtype == np.int8
    expected = np.zeros(24, dtype=np.int8)
    expected[charge_hours] = Action.CHARGE
    expected[discharge_hours] = Action.DISCHARGE
    assert np.array_equal(acts, expected)


def test_night_day_custom_hours(synthetic_prices):
    frame = add_relative_price(synthetic_prices(1)).iloc[:24].reset_index(drop=True)
    acts = NightDay(charge_start=0, discharge_start=12).actions(frame, get_preset("2h"))
    assert acts[[0, 1]].tolist() == [Action.CHARGE] * 2
    assert acts[[12, 13]].tolist() == [Action.DISCHARGE] * 2
    assert (acts == Action.IDLE).sum() == 20


def test_perfect_foresight_hand_checked_optimum():
    # 2h battery, cheap-cheap-dear-dear: charge twice, discharge twice (spread is linear,
    # so using the full capacity is optimal).
    frame = _frame([100.0, 100.0, 300.0, 300.0])
    preset = get_preset("2h")
    acts = PerfectForesight().actions(frame, preset)
    assert acts.tolist() == [Action.CHARGE, Action.CHARGE, Action.DISCHARGE, Action.DISCHARGE]

    level, total = 0, 0.0
    for action, price in zip(acts, frame["price_cop_kwh"]):
        level, grid = step(level, Action(int(action)), preset)
        total += -grid * price
    expected = 2 * 100 * ETA * 300 - 2 * (100 / ETA) * 100  # ~ 35,839.2 COP
    assert expected == pytest.approx(35839.2, abs=0.5)
    assert total == pytest.approx(expected)


def test_perfect_foresight_hand_checked_optimum_through_simulate():
    # Same market, padded to one full day with a flat tail so simulate can slice by date.
    prices = [100.0, 100.0, 300.0, 300.0] + [200.0] * 20
    frame = _frame(prices)
    expected = 2 * 100 * ETA * 300 - 2 * (100 / ETA) * 100
    assert _savings(PerfectForesight(), frame) == pytest.approx(expected)


def test_perfect_foresight_only_valid_actions(synthetic_prices):
    frame = add_relative_price(synthetic_prices(2, noise=0.1, seed=3))
    preset = get_preset("3h")
    acts = PerfectForesight().actions(frame, preset)
    assert acts.dtype == np.int8 and len(acts) == len(frame)
    mask, level = valid_mask(preset), 0
    for action in acts:
        assert mask[level, action]
        level, _ = step(level, Action(int(action)), preset)


@pytest.mark.parametrize("preset_id", ["2h", "3h", "4h"])
def test_savings_ordering_on_synthetic_prices(synthetic_prices, preset_id):
    frame = add_relative_price(synthetic_prices(2))
    pf = _savings(PerfectForesight(), frame, preset_id)
    nd = _savings(NightDay(), frame, preset_id)
    nb = _savings(NoBattery(), frame, preset_id)
    assert nb == pytest.approx(0.0)
    assert nd > nb
    assert pf >= nd - 1e-6


def test_perfect_foresight_earns_nothing_on_flat_prices():
    frame = _frame([200.0] * 48)
    assert _savings(PerfectForesight(), frame) == pytest.approx(0.0, abs=1e-6)
