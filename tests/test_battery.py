"""Tests for `energy.rl.battery`: presets, one-step physics and lookup tables (D9, D14, D15).

Offline and deterministic. Run with `uv run pytest tests/test_battery.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from energy.rl.battery import (
    PRESETS,
    Action,
    BatteryPreset,
    get_preset,
    step,
    transition_tables,
    valid_mask,
)

ETA = math.sqrt(0.9)
# Small battery for fast physics checks; D15 presets are 4/8/12 h.
P2 = BatteryPreset("2h", "2 h", 200.0, 100.0)
P3 = BatteryPreset("3h", "3 h", 300.0, 100.0)


def test_action_values():
    assert (Action.IDLE, Action.CHARGE, Action.DISCHARGE) == (0, 1, 2)


@pytest.mark.parametrize(
    "preset_id,capacity,levels", [("4h", 400, 5), ("8h", 800, 9), ("12h", 1200, 13)]
)
def test_presets(preset_id, capacity, levels):
    preset = get_preset(preset_id)
    assert preset == PRESETS[preset_id]
    assert preset.id == preset_id
    assert preset.capacity_kwh == capacity
    assert preset.power_kw == 100
    assert preset.duration_h == capacity // 100
    assert preset.n_levels == levels
    assert preset.round_trip_efficiency == pytest.approx(0.9)
    assert preset.one_way_efficiency == pytest.approx(ETA)


def test_presets_are_exactly_4_8_12_hours():
    assert set(PRESETS) == {"4h", "8h", "12h"}


def test_get_preset_accepts_preset_object():
    assert get_preset(P2) is P2


def test_get_preset_unknown_lists_valid_ids():
    with pytest.raises(ValueError, match="4h"):
        get_preset("9h")


def test_preset_requires_whole_hour_duration():
    with pytest.raises(ValueError):
        BatteryPreset(id="x", label="x", capacity_kwh=250, power_kw=100)


def test_charge_from_empty():
    level, grid = step(0, Action.CHARGE, P2)
    assert level == 1
    assert grid == pytest.approx(100 / ETA)


def test_discharge_from_level_one():
    level, grid = step(1, Action.DISCHARGE, P2)
    assert level == 0
    assert grid == pytest.approx(-100 * ETA)


def test_idle_changes_nothing():
    assert step(1, Action.IDLE, P3) == (1, 0.0)


@pytest.mark.parametrize("preset_id", ["4h", "8h", "12h"])
def test_full_cycle_efficiency(preset_id):
    preset = get_preset(preset_id)
    level, bought, delivered = 0, 0.0, 0.0
    for _ in range(preset.duration_h):
        level, grid = step(level, Action.CHARGE, preset)
        bought += grid
    assert level == preset.duration_h
    for _ in range(preset.duration_h):
        level, grid = step(level, Action.DISCHARGE, preset)
        delivered -= grid
    assert level == 0
    assert delivered / bought == pytest.approx(0.9)


def test_impossible_actions_become_idle():
    preset = P2
    assert step(0, Action.DISCHARGE, preset) == (0, 0.0)
    assert step(preset.duration_h, Action.CHARGE, preset) == (preset.duration_h, 0.0)


@pytest.mark.parametrize("bad_level", [-1, 3])
def test_out_of_range_level(bad_level):
    with pytest.raises(ValueError):
        step(bad_level, Action.IDLE, P2)


@pytest.mark.parametrize("preset_id", ["4h", "8h", "12h"])
def test_transition_tables_match_step(preset_id):
    preset = get_preset(preset_id)
    next_level, grid_kwh = transition_tables(preset)
    assert next_level.shape == grid_kwh.shape == (preset.n_levels, 3)
    assert next_level.dtype == np.int8
    assert grid_kwh.dtype == np.float64
    for level in range(preset.n_levels):
        for action in Action:
            expected_level, expected_grid = step(level, action, preset)
            assert next_level[level, action] == expected_level
            assert grid_kwh[level, action] == pytest.approx(expected_grid)


@pytest.mark.parametrize("preset_id", ["4h", "8h", "12h"])
def test_valid_mask(preset_id):
    preset = get_preset(preset_id)
    mask = valid_mask(preset)
    assert mask.shape == (preset.n_levels, 3)
    assert mask.dtype == np.bool_
    assert mask[:, Action.IDLE].all()
    assert not mask[0, Action.DISCHARGE]
    assert not mask[preset.duration_h, Action.CHARGE]
    assert mask[0, Action.CHARGE]
    assert mask[preset.duration_h, Action.DISCHARGE]
    assert mask[1:, Action.DISCHARGE].all()
    assert mask[:-1, Action.CHARGE].all()


# --- D15: discharge is possible only while the factory is on ---


@pytest.mark.parametrize("level", [0, 1, 2])
def test_discharge_while_factory_off_is_idle(level):
    assert step(level, Action.DISCHARGE, P2, factory_on=False) == (level, 0.0)


def test_discharge_while_factory_on_still_works():
    assert step(1, Action.DISCHARGE, P2, factory_on=True) == step(1, Action.DISCHARGE, P2)
    assert step(1, Action.DISCHARGE, P2, factory_on=True)[0] == 0


def test_charge_and_idle_are_allowed_while_factory_off():
    level, grid = step(0, Action.CHARGE, P2, factory_on=False)
    assert level == 1 and grid == pytest.approx(100 / ETA)
    assert step(1, Action.IDLE, P2, factory_on=False) == (1, 0.0)


def test_valid_mask_factory_off_blocks_discharge_everywhere():
    for preset in (P2, P3, get_preset("4h")):
        mask = valid_mask(preset, factory_on=False)
        assert mask.shape == (preset.n_levels, 3)
        assert not mask[:, Action.DISCHARGE].any()
        assert mask[:, Action.IDLE].all()
        assert mask[0, Action.CHARGE]
        assert not mask[preset.duration_h, Action.CHARGE]
        assert mask[:-1, Action.CHARGE].all()


def test_valid_mask_factory_on_matches_default():
    assert np.array_equal(valid_mask(P3, factory_on=True), valid_mask(P3))
