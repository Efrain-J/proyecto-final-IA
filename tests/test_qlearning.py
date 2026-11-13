"""Tests for `energy.rl.qlearning`: tabular Q-learning, TabularPolicy and persistence.

Offline and fast: ~10 synthetic training weeks, small `max_epochs`, 2 held-out weeks.
Run with `uv run pytest tests/test_qlearning.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from energy.rl.battery import Action, get_preset, step, valid_mask
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, PerfectForesight
from energy.rl.qlearning import (
    QConfig,
    TabularPolicy,
    TrainResult,
    policy_stem,
    train_q_learning,
)
from energy.rl.simulate import simulate

TRAIN_WEEKS, TEST_WEEKS = 10, 2
FAST = QConfig(max_epochs=150, patience=10, seed=0)


@pytest.fixture(scope="module")
def frames(synthetic_prices):
    """(train, test) frames cut from one series so rel_price has a proper history."""
    full = add_relative_price(synthetic_prices(TRAIN_WEEKS + TEST_WEEKS, noise=0.05, seed=1))
    cut = TRAIN_WEEKS * 168
    return full.iloc[:cut].reset_index(drop=True), full.iloc[cut:].reset_index(drop=True)


@pytest.fixture(scope="module")
def trained(frames) -> TrainResult:
    return train_q_learning(frames[0], get_preset("2h"), FAST)


def _dates(frame):
    return frame["timestamp"].iloc[0].date(), frame["timestamp"].iloc[-1].date()


def _policy(result: TrainResult, preset_id: str = "2h") -> TabularPolicy:
    return TabularPolicy(result.q, result.discretizer, preset_id)


def test_qconfig_defaults():
    cfg = QConfig()
    assert (cfg.max_epochs, cfg.alpha, cfg.gamma) == (300, 0.1, 0.99)
    assert (cfg.epsilon_start, cfg.epsilon_end) == (1.0, 0.05)
    assert (cfg.patience, cfg.episode_hours, cfg.seed) == (20, 168, 0)


def test_q_shape_and_dtype(trained):
    assert trained.q.shape == (12, 3, 4, 3)
    assert trained.q.dtype == np.float32
    assert 1 <= trained.epochs_run <= FAST.max_epochs
    assert len(trained.epoch_returns) == trained.epochs_run


@pytest.mark.parametrize("preset_id,levels", [("3h", 4), ("4h", 5)])
def test_q_shape_scales_with_preset(frames, preset_id, levels):
    result = train_q_learning(frames[0], get_preset(preset_id), QConfig(max_epochs=3, patience=2))
    assert result.q.shape == (12, levels, 4, 3)


def test_same_seed_gives_identical_q(frames, trained):
    again = train_q_learning(frames[0], get_preset("2h"), FAST)
    assert np.array_equal(trained.q, again.q)
    assert again.epochs_run == trained.epochs_run


def test_missing_rel_price_raises(frames):
    with pytest.raises(ValueError):
        train_q_learning(frames[0].drop(columns=["rel_price"]), get_preset("2h"), FAST)


def test_greedy_actions_are_always_valid(frames, trained):
    preset = get_preset("2h")
    acts = _policy(trained).actions(frames[1], preset)
    assert acts.dtype == np.int8 and len(acts) == len(frames[1])
    mask, level = valid_mask(preset), 0
    for action in acts:
        assert mask[level, action]
        level, _ = step(level, Action(int(action)), preset)


def test_simulation_reports_no_invalid_actions(frames, trained):
    start, end = _dates(frames[1])
    result = simulate(_policy(trained), start, end, "2h", prices=frames[1])
    assert result.summary.invalid_actions == 0


def test_reaches_90_percent_of_night_day_on_held_out_weeks(frames, trained):
    # Perfect foresight profits from the noise no causal policy can see (even NightDay
    # gets only ~75-84% of it here), so the meaningful bar is the fixed schedule.
    start, end = _dates(frames[1])
    q = simulate(_policy(trained), start, end, "2h", prices=frames[1]).summary.savings_cop
    nd = simulate(NightDay(), start, end, "2h", prices=frames[1]).summary.savings_cop
    pf = simulate(PerfectForesight(), start, end, "2h", prices=frames[1]).summary.savings_cop
    assert 0 < nd <= pf
    assert q >= 0.9 * nd


def test_early_stopping_on_easy_problem(frames):
    result = train_q_learning(frames[0], get_preset("2h"), QConfig(max_epochs=300, patience=5, seed=0))
    assert result.epochs_run < 300


def test_policy_stem(tmp_path):
    assert policy_stem(tmp_path, "2h", 2025) == tmp_path / "q_2h_2025"


def test_save_load_round_trip(tmp_path, frames, trained):
    policy = TabularPolicy(trained.q, trained.discretizer, "2h", metadata={"year": 2025})
    stem = policy_stem(tmp_path, "2h", 2025)
    policy.save(stem)
    assert stem.with_suffix(".npz").exists()
    assert stem.with_suffix(".json").exists()
    assert stem.with_suffix(".npz").stat().st_size < 20 * 1024

    loaded = TabularPolicy.load(stem)
    assert loaded.preset_id == "2h"
    assert loaded.name == "q_learning"
    preset = get_preset("2h")
    assert np.array_equal(loaded.actions(frames[1], preset), policy.actions(frames[1], preset))


def test_policy_with_wrong_preset_raises(frames, trained):
    with pytest.raises(ValueError):
        _policy(trained, "2h").actions(frames[1], get_preset("3h"))
