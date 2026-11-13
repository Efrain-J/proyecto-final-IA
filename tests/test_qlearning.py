"""Tests for `energy.rl.qlearning`: shift-aware tabular Q-learning, TabularPolicy, persistence.

Offline and fast: a small 2-hour battery, 12 synthetic training weeks repeated under 4 shift
patterns, small `max_epochs`, 4 held-out weeks under 3 other shift patterns (D15).
Run with `uv run pytest tests/test_qlearning.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from energy.rl.battery import Action, BatteryPreset, get_preset, step, valid_mask
from energy.rl.factory import (
    N_SHIFT_TYPES,
    SHIFT_FULL,
    TEST_SHIFT_SEEDS,
    add_shifts,
    train_shift_seeds,
)
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, PriceThreshold, ShiftSchedule
from energy.rl.qlearning import (
    QConfig,
    TabularPolicy,
    TrainResult,
    policy_stem,
    train_q_learning,
)
from energy.rl.simulate import simulate

P2 = BatteryPreset("2h", "2 h", 200.0, 100.0)
P3 = BatteryPreset("3h", "3 h", 300.0, 100.0)
TRAIN_WEEKS, TEST_WEEKS, N_DRAWS = 12, 4, 4
FAST = QConfig(max_epochs=150, patience=10, seed=0)


@pytest.fixture(scope="module")
def series(synthetic_prices):
    """(train, test) base frames cut from one series so rel_price has a proper history."""
    full = add_relative_price(synthetic_prices(TRAIN_WEEKS + TEST_WEEKS, noise=0.05, seed=1))
    cut = TRAIN_WEEKS * 168
    return full.iloc[:cut].reset_index(drop=True), full.iloc[cut:].reset_index(drop=True)


@pytest.fixture(scope="module")
def train_frames(series):
    return [add_shifts(series[0], "default", s) for s in train_shift_seeds(7, N_DRAWS)]


@pytest.fixture(scope="module")
def test_frames(series):
    """The held-out weeks under three test shift patterns (never used in training)."""
    return [add_shifts(series[1], "default", s) for s in TEST_SHIFT_SEEDS[:3]]


@pytest.fixture(scope="module")
def trained(train_frames) -> TrainResult:
    return train_q_learning(train_frames, P2, FAST)


def _dates(frame):
    return frame["timestamp"].iloc[0].date(), frame["timestamp"].iloc[-1].date()


def _policy(result: TrainResult, preset_id: str = "2h") -> TabularPolicy:
    return TabularPolicy(result.q, result.discretizer, preset_id)


def _savings(policy, frame, preset=P2) -> float:
    start, end = _dates(frame)
    return simulate(policy, start, end, preset, prices=frame).summary.savings_cop


def test_qconfig_defaults():
    cfg = QConfig()
    assert (cfg.max_epochs, cfg.alpha, cfg.gamma) == (300, 0.1, 0.99)
    assert (cfg.epsilon_start, cfg.epsilon_end) == (1.0, 0.05)
    assert (cfg.patience, cfg.episode_hours, cfg.seed) == (20, 168, 0)


def test_q_shape_and_dtype(trained):
    assert N_SHIFT_TYPES == 4
    assert trained.q.shape == (12, 3, 4, 4, 3)
    assert trained.q.dtype == np.float32
    assert 1 <= trained.epochs_run <= FAST.max_epochs
    assert len(trained.epoch_returns) == trained.epochs_run


@pytest.mark.parametrize(
    "preset,levels", [(P3, 4), (get_preset("4h"), 5), (get_preset("8h"), 9)],
    ids=["3h", "4h", "8h"],
)
def test_q_shape_scales_with_preset(train_frames, preset, levels):
    result = train_q_learning(train_frames[:1], preset, QConfig(max_epochs=3, patience=2))
    assert result.q.shape == (12, levels, 4, N_SHIFT_TYPES, 3)


def test_single_frame_equals_one_element_list(train_frames):
    cfg = QConfig(max_epochs=5, patience=3, seed=2)
    single = train_q_learning(train_frames[0], P2, cfg)
    listed = train_q_learning([train_frames[0]], P2, cfg)
    assert np.array_equal(single.q, listed.q)


def test_same_seed_gives_identical_q(train_frames, trained):
    again = train_q_learning(train_frames, P2, FAST)
    assert np.array_equal(trained.q, again.q)
    assert again.epochs_run == trained.epochs_run


def test_different_seeds_give_different_q(train_frames):
    a = train_q_learning(train_frames, P2, QConfig(max_epochs=5, patience=3, seed=0))
    b = train_q_learning(train_frames, P2, QConfig(max_epochs=5, patience=3, seed=1))
    assert not np.array_equal(a.q, b.q)


def test_every_shift_type_gets_learned_values(trained):
    # Weekdays include every shift type, so each slice of the table must be visited.
    visited = (trained.q != 0).any(axis=(0, 1, 2, 4))
    assert visited.all()


@pytest.mark.parametrize("column", ["rel_price", "shift_type", "factory_on"])
def test_missing_column_raises(train_frames, column):
    with pytest.raises(ValueError):
        train_q_learning(train_frames[0].drop(columns=[column]), P2, FAST)


def test_missing_column_in_any_list_element_raises(train_frames):
    with pytest.raises(ValueError):
        train_q_learning([train_frames[0], train_frames[1].drop(columns=["factory_on"])], P2, FAST)


def test_greedy_actions_are_valid_and_never_discharge_while_off(test_frames, trained):
    for frame in test_frames:
        acts = _policy(trained).actions(frame, P2)
        on = frame["factory_on"].to_numpy()
        assert acts.dtype == np.int8 and len(acts) == len(frame)
        assert not (acts[~on] == Action.DISCHARGE).any()
        level = 0
        for t, action in enumerate(acts):
            assert valid_mask(P2, factory_on=bool(on[t]))[level, action]
            level, _ = step(level, Action(int(action)), P2, factory_on=bool(on[t]))


def test_policy_without_shift_columns_means_always_on(test_frames, trained, always_on):
    bare = test_frames[0].drop(columns=["shift_type", "factory_on"])
    explicit = always_on(bare)
    policy = _policy(trained)
    assert np.array_equal(policy.actions(bare, P2), policy.actions(explicit, P2))
    assert (explicit["shift_type"] == SHIFT_FULL).all()


def test_simulation_reports_no_invalid_actions(test_frames, trained):
    for frame in test_frames:
        start, end = _dates(frame)
        result = simulate(_policy(trained), start, end, P2, prices=frame)
        assert result.summary.invalid_actions == 0


def test_reaches_90_percent_of_best_rule_on_held_out_weeks(train_frames, test_frames, trained):
    # Perfect foresight profits from noise no causal policy can see, so the meaningful bar
    # is the best hand rule (D15). Totals over three held-out shift patterns for stability.
    tuned = PriceThreshold.tune(train_frames, P2)
    policy = _policy(trained)
    q = sum(_savings(policy, f) for f in test_frames)
    rules = {
        "shift_schedule": sum(_savings(ShiftSchedule(), f) for f in test_frames),
        "price_threshold": sum(_savings(tuned, f) for f in test_frames),
        "night_day": sum(_savings(NightDay(), f) for f in test_frames),
    }
    best = max(rules.values())
    assert best > 0
    assert q >= 0.9 * best, (q, rules)


def test_early_stopping_on_easy_problem(train_frames):
    result = train_q_learning(train_frames, P2, QConfig(max_epochs=300, patience=5, seed=0))
    assert result.epochs_run < 300


def test_policy_stem(tmp_path):
    assert policy_stem(tmp_path, "4h", 2025) == tmp_path / "q_4h_2025"


def test_save_load_round_trip(tmp_path, test_frames, trained):
    meta = {"year": 2025, "shift_mix": "default", "shift_draws": N_DRAWS}
    policy = TabularPolicy(trained.q, trained.discretizer, "2h", metadata=meta)
    stem = policy_stem(tmp_path, "2h", 2025)
    policy.save(stem)
    assert stem.with_suffix(".npz").exists()
    assert stem.with_suffix(".json").exists()
    assert stem.with_suffix(".npz").stat().st_size < 20 * 1024

    loaded = TabularPolicy.load(stem)
    assert loaded.preset_id == "2h"
    assert loaded.name == "q_learning"
    assert loaded.q.shape == trained.q.shape
    assert loaded.metadata["shift_mix"] == "default"
    assert np.array_equal(loaded.actions(test_frames[0], P2), policy.actions(test_frames[0], P2))


def test_policy_with_wrong_preset_raises(test_frames, trained):
    with pytest.raises(ValueError):
        _policy(trained, "2h").actions(test_frames[0], P3)
