"""Tests for `energy.rl.factory`: shift types, mixes and the seeded shift generator (D15, A5).

Offline and deterministic. Run with `uv run pytest tests/test_factory.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy.rl.factory import (
    MIXES,
    N_SHIFT_TYPES,
    SHIFT_AFTERNOON,
    SHIFT_FULL,
    SHIFT_HOURS,
    SHIFT_MORNING,
    SHIFT_OFF,
    TEST_SHIFT_SEEDS,
    ShiftMix,
    add_shifts,
    get_mix,
    train_shift_seeds,
)


@pytest.fixture(scope="module")
def two_years(synthetic_prices):
    """Two synthetic years (104 weeks from Monday 2025-01-06), no shift columns."""
    return synthetic_prices(104)


def _per_day(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day with the weekday and the day's (first) shift type."""
    days = pd.DatetimeIndex(frame["timestamp"]).normalize()
    out = pd.DataFrame({"day": days, "shift_type": frame["shift_type"].to_numpy()})
    out = out.groupby("day", as_index=False).first()
    out["weekday"] = out["day"].dt.day_of_week
    return out


def test_shift_constants():
    assert (SHIFT_OFF, SHIFT_MORNING, SHIFT_AFTERNOON, SHIFT_FULL) == (0, 1, 2, 3)
    assert N_SHIFT_TYPES == 4
    assert list(SHIFT_HOURS[SHIFT_OFF]) == []
    assert list(SHIFT_HOURS[SHIFT_MORNING]) == list(range(6, 14))
    assert list(SHIFT_HOURS[SHIFT_AFTERNOON]) == list(range(14, 22))
    assert list(SHIFT_HOURS[SHIFT_FULL]) == list(range(6, 22))


def test_test_shift_seeds():
    assert TEST_SHIFT_SEEDS == (1000, 1001, 1002, 1003, 1004)


def test_train_shift_seeds():
    assert train_shift_seeds(7, 3) == [8, 9, 10]
    assert train_shift_seeds(0, 0) == []


def test_default_mix_probabilities():
    mix = MIXES["default"]
    assert mix.id == "default"
    assert mix.weekday == pytest.approx((0.10, 0.30, 0.15, 0.45))
    assert mix.saturday == pytest.approx((0.60, 0.40, 0.0, 0.0))
    assert mix.sunday == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_named_mixes_exist_and_are_valid():
    assert {"default", "mostly_double", "mostly_single"} <= set(MIXES)
    for mix in MIXES.values():
        for probs in (mix.weekday, mix.saturday, mix.sunday):
            assert len(probs) == N_SHIFT_TYPES
            assert sum(probs) == pytest.approx(1.0)
    assert MIXES["mostly_double"].weekday[SHIFT_FULL] > MIXES["default"].weekday[SHIFT_FULL]
    assert MIXES["mostly_single"].weekday[SHIFT_FULL] < MIXES["default"].weekday[SHIFT_FULL]


@pytest.mark.parametrize("field", ["weekday", "saturday", "sunday"])
def test_shift_mix_rejects_probabilities_not_summing_to_one(field):
    good = {
        "weekday": (0.1, 0.3, 0.15, 0.45),
        "saturday": (0.6, 0.4, 0.0, 0.0),
        "sunday": (1.0, 0.0, 0.0, 0.0),
    }
    good[field] = (0.5, 0.3, 0.1, 0.0)  # sums to 0.9
    with pytest.raises(ValueError):
        ShiftMix(id="bad", **good)


def test_shift_mix_accepts_valid_probabilities():
    mix = ShiftMix(
        id="ok",
        weekday=(0.25, 0.25, 0.25, 0.25),
        saturday=(1.0, 0.0, 0.0, 0.0),
        sunday=(1.0, 0.0, 0.0, 0.0),
    )
    assert mix.id == "ok"


def test_shift_mix_is_frozen():
    with pytest.raises(AttributeError):
        MIXES["default"].id = "other"  # type: ignore[misc]


def test_get_mix_by_id_and_object():
    assert get_mix("default") is MIXES["default"]
    assert get_mix(MIXES["mostly_single"]) is MIXES["mostly_single"]


def test_get_mix_unknown_raises():
    with pytest.raises(ValueError):
        get_mix("nonsense")


def test_add_shifts_returns_typed_copy(two_years):
    out = add_shifts(two_years, "default", seed=0)
    assert "shift_type" not in two_years.columns and "factory_on" not in two_years.columns
    assert out is not two_years
    assert len(out) == len(two_years)
    assert out["shift_type"].dtype == np.int8
    assert out["factory_on"].dtype == np.bool_
    assert set(out["shift_type"].unique()) <= set(range(N_SHIFT_TYPES))
    assert np.array_equal(out["price_cop_kwh"], two_years["price_cop_kwh"])


def test_one_shift_type_per_calendar_day(two_years):
    out = add_shifts(two_years, "default", seed=3)
    days = pd.DatetimeIndex(out["timestamp"]).normalize()
    assert (out.groupby(days)["shift_type"].nunique() == 1).all()


def test_factory_on_matches_shift_hours(two_years):
    out = add_shifts(two_years, "default", seed=4)
    hours = pd.DatetimeIndex(out["timestamp"]).hour.to_numpy()
    expected = np.array(
        [h in SHIFT_HOURS[int(s)] for h, s in zip(hours, out["shift_type"].to_numpy())]
    )
    assert np.array_equal(out["factory_on"].to_numpy(), expected)


def test_sundays_are_always_off(two_years):
    out = add_shifts(two_years, "mostly_double", seed=5)
    sundays = pd.DatetimeIndex(out["timestamp"]).day_of_week.to_numpy() == 6
    assert sundays.any()
    assert (out.loc[sundays, "shift_type"] == SHIFT_OFF).all()
    assert not out.loc[sundays, "factory_on"].any()


def test_saturdays_only_off_or_morning(two_years):
    out = add_shifts(two_years, "default", seed=6)
    saturdays = pd.DatetimeIndex(out["timestamp"]).day_of_week.to_numpy() == 5
    assert set(out.loc[saturdays, "shift_type"].unique()) <= {SHIFT_OFF, SHIFT_MORNING}


def test_same_seed_is_identical(two_years):
    a = add_shifts(two_years, "default", seed=11)
    b = add_shifts(two_years, "default", seed=11)
    assert np.array_equal(a["shift_type"], b["shift_type"])
    assert np.array_equal(a["factory_on"], b["factory_on"])


def test_default_arguments_are_mix_default_seed_zero(two_years):
    a, b = add_shifts(two_years), add_shifts(two_years, "default", seed=0)
    assert np.array_equal(a["shift_type"], b["shift_type"])


def test_different_seeds_differ(two_years):
    a = add_shifts(two_years, "default", seed=1)
    b = add_shifts(two_years, "default", seed=2)
    assert not np.array_equal(a["shift_type"], b["shift_type"])


def test_different_mixes_differ(two_years):
    a = add_shifts(two_years, "mostly_double", seed=1)
    b = add_shifts(two_years, "mostly_single", seed=1)
    assert not np.array_equal(a["shift_type"], b["shift_type"])


def test_accepts_mix_object(two_years):
    a = add_shifts(two_years, MIXES["default"], seed=9)
    b = add_shifts(two_years, "default", seed=9)
    assert np.array_equal(a["shift_type"], b["shift_type"])


def test_unknown_mix_in_add_shifts_raises(two_years):
    with pytest.raises(ValueError):
        add_shifts(two_years, "nonsense", seed=0)


def test_weekday_frequencies_follow_the_mix(two_years):
    days = _per_day(add_shifts(two_years, "default", seed=21))
    weekdays = days[days["weekday"] < 5]
    assert len(weekdays) == 104 * 5
    freq = weekdays["shift_type"].value_counts(normalize=True)
    target = MIXES["default"].weekday
    for shift_type in range(N_SHIFT_TYPES):
        assert freq.get(shift_type, 0.0) == pytest.approx(target[shift_type], abs=0.06)


def test_saturday_frequencies_follow_the_mix(two_years):
    days = _per_day(add_shifts(two_years, "default", seed=22))
    saturdays = days[days["weekday"] == 5]
    share_morning = (saturdays["shift_type"] == SHIFT_MORNING).mean()
    assert share_morning == pytest.approx(0.40, abs=0.12)
