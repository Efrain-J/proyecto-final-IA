"""Tests for `energy.rl.policies`: baselines, shift-aware rules and perfect foresight (D14, D15).

Offline. Savings are computed through `simulate` with the policy object. Frames without
`shift_type`/`factory_on` mean "the factory is always on" for `.actions`, but `simulate`
adds random shifts to such frames, so savings tests pass explicit columns.
Run with `uv run pytest tests/test_policies.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from energy.rl.battery import Action, BatteryPreset, get_preset, step, valid_mask
from energy.rl.factory import (
    SHIFT_AFTERNOON,
    SHIFT_FULL,
    SHIFT_HOURS,
    SHIFT_MORNING,
    SHIFT_OFF,
    add_shifts,
)
from energy.rl.features import add_relative_price
from energy.rl.policies import (
    NightDay,
    NoBattery,
    PerfectForesight,
    PriceThreshold,
    ShiftSchedule,
    factory_columns,
)
from energy.rl.simulate import simulate

ETA = math.sqrt(0.9)
P2 = BatteryPreset("2h", "2 h", 200.0, 100.0)
P3 = BatteryPreset("3h", "3 h", 300.0, 100.0)
P4 = get_preset("4h")
MONDAY = "2025-01-06"


def _frame(prices, start: str = MONDAY, on: bool = True) -> pd.DataFrame:
    """Frame with rel_price and explicit always-on (or always-off) shift columns."""
    ts = pd.date_range(start, periods=len(prices), freq="h")
    frame = add_relative_price(
        pd.DataFrame({"timestamp": ts, "price_cop_kwh": np.asarray(prices, float)})
    )
    frame["shift_type"] = np.full(len(frame), SHIFT_FULL if on else SHIFT_OFF, dtype=np.int8)
    frame["factory_on"] = np.full(len(frame), on, dtype=bool)
    return frame


def _day(shift: int, start: str = MONDAY) -> pd.DataFrame:
    """One 24-hour day with a given shift type and a matching factory_on."""
    ts = pd.date_range(start, periods=24, freq="h")
    on = np.array([h in SHIFT_HOURS[shift] for h in ts.hour])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price_cop_kwh": 200.0,
            "rel_price": 1.0,
            "shift_type": np.full(24, shift, dtype=np.int8),
            "factory_on": on,
        }
    )


def _expected(n: int, charge: list[int], discharge: list[int]) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    out[charge] = Action.CHARGE
    out[discharge] = Action.DISCHARGE
    return out


def _savings(policy, frame: pd.DataFrame, preset=P2) -> float:
    start = frame["timestamp"].iloc[0].date()
    end = frame["timestamp"].iloc[-1].date()
    return simulate(policy, start, end, preset, prices=frame).summary.savings_cop


def _shifted(synthetic_prices, weeks: int, seed: int, noise: float = 0.05, price_seed: int = 3):
    return add_shifts(
        add_relative_price(synthetic_prices(weeks, noise=noise, seed=price_seed)), "default", seed
    )


def _replay_levels(acts, factory_on, preset):
    """Yield (t, level_before, action) following the physics with the factory flag."""
    level = 0
    for t, action in enumerate(acts):
        yield t, level, int(action)
        level, _ = step(level, Action(int(action)), preset, factory_on=bool(factory_on[t]))


def test_policy_names():
    assert NoBattery().name == "no_battery"
    assert NightDay().name == "night_day"
    assert ShiftSchedule().name == "shift_schedule"
    assert PriceThreshold(0.9, 1.1).name == "price_threshold"
    assert PerfectForesight().name == "perfect_foresight"


# --- factory_columns ---


def test_factory_columns_defaults_to_always_on_full_day():
    frame = _frame([200.0] * 5).drop(columns=["shift_type", "factory_on"])
    shift_type, factory_on = factory_columns(frame)
    assert len(shift_type) == len(factory_on) == 5
    assert shift_type.dtype == np.int8 and (shift_type == SHIFT_FULL).all()
    assert factory_on.dtype == np.bool_ and factory_on.all()


def test_factory_columns_reads_existing_columns(synthetic_prices):
    frame = add_shifts(synthetic_prices(1), "default", seed=2)
    shift_type, factory_on = factory_columns(frame)
    assert np.array_equal(shift_type, frame["shift_type"].to_numpy())
    assert np.array_equal(factory_on, frame["factory_on"].to_numpy())


# --- baselines from D14 ---


def test_no_battery_always_idle(synthetic_prices):
    frame = add_relative_price(synthetic_prices(1))
    acts = NoBattery().actions(frame, P4)
    assert len(acts) == len(frame)
    assert acts.dtype == np.int8
    assert (acts == Action.IDLE).all()


@pytest.mark.parametrize(
    "preset,charge_hours,discharge_hours",
    [(P2, [2, 3], [18, 19]), (P3, [2, 3, 4], [18, 19, 20])],
)
def test_night_day_hours(synthetic_prices, preset, charge_hours, discharge_hours):
    frame = add_relative_price(synthetic_prices(1)).iloc[:24].reset_index(drop=True)
    acts = NightDay().actions(frame, preset)
    assert acts.dtype == np.int8
    assert np.array_equal(acts, _expected(24, charge_hours, discharge_hours))


def test_night_day_custom_hours(synthetic_prices):
    frame = add_relative_price(synthetic_prices(1)).iloc[:24].reset_index(drop=True)
    acts = NightDay(charge_start=0, discharge_start=12).actions(frame, P2)
    assert acts[[0, 1]].tolist() == [Action.CHARGE] * 2
    assert acts[[12, 13]].tolist() == [Action.DISCHARGE] * 2
    assert (acts == Action.IDLE).sum() == 20


# --- ShiftSchedule ---


@pytest.mark.parametrize(
    "shift,charge,discharge",
    [
        (SHIFT_FULL, [2, 3, 4, 5], [18, 19, 20, 21]),
        (SHIFT_AFTERNOON, [2, 3, 4, 5], [18, 19, 20, 21]),
        (SHIFT_MORNING, [2, 3, 4, 5], [10, 11, 12, 13]),  # last 4 h of 06-14
        (SHIFT_OFF, [], []),
    ],
    ids=["full", "afternoon", "morning", "off"],
)
def test_shift_schedule_hours_4h(shift, charge, discharge):
    acts = ShiftSchedule().actions(_day(shift), P4)
    assert acts.dtype == np.int8
    assert np.array_equal(acts, _expected(24, charge, discharge))


@pytest.mark.parametrize(
    "shift,charge,discharge",
    [
        (SHIFT_FULL, [2, 3], [18, 19]),
        (SHIFT_MORNING, [2, 3], [12, 13]),
        (SHIFT_OFF, [], []),
    ],
    ids=["full", "morning", "off"],
)
def test_shift_schedule_hours_2h(shift, charge, discharge):
    acts = ShiftSchedule().actions(_day(shift), P2)
    assert np.array_equal(acts, _expected(24, charge, discharge))


def test_shift_schedule_follows_each_day_of_a_multi_day_frame():
    frame = pd.concat(
        [
            _day(SHIFT_FULL, "2025-01-06"),
            _day(SHIFT_OFF, "2025-01-07"),
            _day(SHIFT_MORNING, "2025-01-08"),
        ],
        ignore_index=True,
    )
    acts = ShiftSchedule().actions(frame, P2)
    expected = np.concatenate(
        [_expected(24, [2, 3], [18, 19]), _expected(24, [], []), _expected(24, [2, 3], [12, 13])]
    )
    assert np.array_equal(acts, expected)


def test_shift_schedule_never_discharges_while_factory_off(synthetic_prices):
    frame = add_shifts(synthetic_prices(4), "mostly_single", seed=4)
    acts = ShiftSchedule().actions(frame, P4)
    assert not (acts[~frame["factory_on"].to_numpy()] == Action.DISCHARGE).any()


def test_shift_schedule_without_shift_columns_equals_night_day(synthetic_prices):
    frame = add_relative_price(synthetic_prices(1)).iloc[:24].reset_index(drop=True)
    assert np.array_equal(ShiftSchedule().actions(frame, P4), NightDay().actions(frame, P4))


# --- PriceThreshold ---


def _threshold_frame(rel, on) -> pd.DataFrame:
    n = len(rel)
    ts = pd.date_range(MONDAY, periods=n, freq="h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price_cop_kwh": 200.0 * np.asarray(rel, float),
            "rel_price": np.asarray(rel, float),
            "shift_type": np.where(on, SHIFT_FULL, SHIFT_OFF).astype(np.int8),
            "factory_on": np.asarray(on, dtype=bool),
        }
    )


def test_price_threshold_fields():
    policy = PriceThreshold(cheap=0.9, expensive=1.1)
    assert (policy.cheap, policy.expensive) == (0.9, 1.1)


def test_price_threshold_charges_cheap_discharges_expensive():
    frame = _threshold_frame([0.5, 0.5, 1.0, 1.5, 1.5, 1.5], [True] * 6)
    acts = PriceThreshold(0.9, 1.1).actions(frame, P2)
    # charge twice (full at level 2), idle at the middle price, discharge twice, then
    # idle because the battery is empty.
    assert acts.dtype == np.int8
    assert acts.tolist() == [1, 1, 0, 2, 2, 0]


def test_price_threshold_does_not_charge_when_full():
    frame = _threshold_frame([0.5] * 5, [True] * 5)
    assert PriceThreshold(0.9, 1.1).actions(frame, P2).tolist() == [1, 1, 0, 0, 0]


def test_price_threshold_never_discharges_while_factory_off():
    frame = _threshold_frame([0.5, 0.5, 1.5, 1.5, 1.5, 1.5], [True, True, False, False, True, True])
    acts = PriceThreshold(0.9, 1.1).actions(frame, P2)
    assert acts.tolist() == [1, 1, 0, 0, 2, 2]


def test_price_threshold_cut_offs_are_strict():
    frame = _threshold_frame([0.9, 1.1, 0.9, 1.1], [True] * 4)
    assert PriceThreshold(0.9, 1.1).actions(frame, P2).tolist() == [0, 0, 0, 0]


def test_price_threshold_only_valid_actions(synthetic_prices):
    frame = _shifted(synthetic_prices, 4, seed=8)
    acts = PriceThreshold(0.95, 1.05).actions(frame, P3)
    on = frame["factory_on"].to_numpy()
    for t, level, action in _replay_levels(acts, on, P3):
        assert valid_mask(P3, factory_on=bool(on[t]))[level, action]


def test_price_threshold_tune_returns_grid_quantiles(synthetic_prices):
    frames = [_shifted(synthetic_prices, 3, seed=s) for s in (31, 32)]
    tuned = PriceThreshold.tune(frames, P2)
    assert isinstance(tuned, PriceThreshold)
    rel = np.concatenate([f["rel_price"].to_numpy() for f in frames])
    quantile_grid = np.quantile(rel, np.arange(0.10, 0.91, 0.05))
    assert tuned.cheap < tuned.expensive
    assert np.isclose(quantile_grid, tuned.cheap).any()
    assert np.isclose(quantile_grid, tuned.expensive).any()
    assert np.quantile(rel, 0.10) - 1e-9 <= tuned.cheap <= np.quantile(rel, 0.50) + 1e-9
    assert np.quantile(rel, 0.50) - 1e-9 <= tuned.expensive <= np.quantile(rel, 0.90) + 1e-9


def test_price_threshold_tune_is_deterministic(synthetic_prices):
    frames = [_shifted(synthetic_prices, 3, seed=s) for s in (31, 32)]
    a, b = PriceThreshold.tune(frames, P2), PriceThreshold.tune(frames, P2)
    assert (a.cheap, a.expensive) == (b.cheap, b.expensive)


def test_price_threshold_tune_is_at_least_as_good_as_decile_pairs(synthetic_prices):
    frames = [_shifted(synthetic_prices, 3, seed=s) for s in (31, 32)]
    tuned = PriceThreshold.tune(frames, P2)
    rel = np.concatenate([f["rel_price"].to_numpy() for f in frames])

    def mean_savings(policy):
        return float(np.mean([_savings(policy, f) for f in frames]))

    best = mean_savings(tuned)
    assert best > 0
    for q_cheap in (0.1, 0.2, 0.3, 0.4, 0.5):
        for q_exp in (0.5, 0.6, 0.7, 0.8, 0.9):
            if q_exp <= q_cheap:
                continue
            pair = PriceThreshold(float(np.quantile(rel, q_cheap)), float(np.quantile(rel, q_exp)))
            assert best >= mean_savings(pair) - 1e-6


# --- PerfectForesight ---


def test_perfect_foresight_hand_checked_optimum():
    # 2h battery, cheap-cheap-dear-dear: charge twice, discharge twice (spread is linear,
    # so using the full capacity is optimal).
    frame = _frame([100.0, 100.0, 300.0, 300.0])
    acts = PerfectForesight().actions(frame, P2)
    assert acts.tolist() == [Action.CHARGE, Action.CHARGE, Action.DISCHARGE, Action.DISCHARGE]

    level, total = 0, 0.0
    for action, price in zip(acts, frame["price_cop_kwh"]):
        level, grid = step(level, Action(int(action)), P2)
        total += -grid * price
    expected = 2 * 100 * ETA * 300 - 2 * (100 / ETA) * 100  # ~ 35,839.2 COP
    assert expected == pytest.approx(35839.2, abs=0.5)
    assert total == pytest.approx(expected)


def test_perfect_foresight_hand_checked_optimum_through_simulate():
    # Same market, padded to one full day with a flat tail so simulate can slice by date.
    frame = _frame([100.0, 100.0, 300.0, 300.0] + [200.0] * 20)
    expected = 2 * 100 * ETA * 300 - 2 * (100 / ETA) * 100
    assert _savings(PerfectForesight(), frame) == pytest.approx(expected)


def test_perfect_foresight_cannot_sell_while_factory_off():
    # The dear hours fall while the factory is off, so storing is pointless: stay idle.
    frame = _frame([100.0, 100.0, 300.0, 300.0])
    frame["factory_on"] = [True, True, False, False]
    assert PerfectForesight().actions(frame, P2).tolist() == [0, 0, 0, 0]


def test_perfect_foresight_waits_for_the_factory_to_restart():
    frame = _frame([100.0, 100.0, 300.0, 300.0, 300.0, 300.0])
    frame["factory_on"] = [True, True, False, False, True, True]
    acts = PerfectForesight().actions(frame, P2)
    assert acts.tolist() == [1, 1, 0, 0, 2, 2]


def test_perfect_foresight_only_valid_actions(synthetic_prices):
    frame = _shifted(synthetic_prices, 2, seed=3, noise=0.1)
    acts = PerfectForesight().actions(frame, P3)
    assert acts.dtype == np.int8 and len(acts) == len(frame)
    on = frame["factory_on"].to_numpy()
    for t, level, action in _replay_levels(acts, on, P3):
        assert valid_mask(P3, factory_on=bool(on[t]))[level, action]
    assert not (acts[~on] == Action.DISCHARGE).any()


@pytest.mark.parametrize("preset", [P2, P3, P4], ids=["2h", "3h", "4h"])
def test_savings_ordering_on_always_on_prices(synthetic_prices, always_on, preset):
    frame = always_on(add_relative_price(synthetic_prices(2)))
    pf = _savings(PerfectForesight(), frame, preset)
    nd = _savings(NightDay(), frame, preset)
    ss = _savings(ShiftSchedule(), frame, preset)
    nb = _savings(NoBattery(), frame, preset)
    assert nb == pytest.approx(0.0)
    assert nd > nb
    assert ss == pytest.approx(nd)  # always-on frames: the shift schedule is NightDay
    assert pf >= nd - 1e-6


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_perfect_foresight_beats_every_rule_on_shifted_prices(synthetic_prices, seed):
    frame = _shifted(synthetic_prices, 3, seed=seed)
    pf = _savings(PerfectForesight(), frame, P3)
    rules = {
        "shift_schedule": ShiftSchedule(),
        "night_day": NightDay(),
        "price_threshold": PriceThreshold(0.95, 1.05),
        "no_battery": NoBattery(),
    }
    for name, policy in rules.items():
        assert pf >= _savings(policy, frame, P3) - 1e-6, name


def test_perfect_foresight_earns_nothing_on_flat_prices():
    frame = _frame([200.0] * 48)
    assert _savings(PerfectForesight(), frame) == pytest.approx(0.0, abs=1e-6)
