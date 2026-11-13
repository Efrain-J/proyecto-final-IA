"""Tests for the Q-learning tuning harness (offline, tiny synthetic data)."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from energy.eval.tune_q import (
    VAL_SHIFT_SEEDS,
    RunSpec,
    TuneConfig,
    run_config,
    validation_split,
)
from energy.rl.factory import TEST_SHIFT_SEEDS
from energy.rl.features import add_relative_price
from energy.rl.qlearning import QConfig


@pytest.fixture(scope="module")
def featured(synthetic_prices):
    # 52 weeks from 2025-01-01 cover the whole year up to Dec 30.
    return add_relative_price(synthetic_prices(52, start="2025-01-01", noise=0.05, seed=1))


def test_validation_split_boundaries(featured):
    fit, val, test = validation_split(featured, 2025)
    assert fit["timestamp"].iloc[-1] == pd.Timestamp("2025-06-30 23:00")
    assert val["timestamp"].iloc[0] == pd.Timestamp("2025-07-01 00:00")
    assert val["timestamp"].iloc[-1] == pd.Timestamp("2025-09-30 23:00")
    assert test["timestamp"].iloc[0] == pd.Timestamp("2025-10-01 00:00")


def test_validation_split_parts_do_not_overlap(featured):
    fit, val, test = validation_split(featured, 2025)
    stamps = [set(part["timestamp"]) for part in (fit, val, test)]
    assert not (stamps[0] & stamps[1]) and not (stamps[1] & stamps[2])
    assert not (stamps[0] & stamps[2])


def test_validation_split_unknown_year_raises(featured):
    with pytest.raises(ValueError):
        validation_split(featured, 1999)


def test_tune_config_maps_to_qconfig():
    cfg = TuneConfig(alpha=0.2, gamma=0.95, max_epochs=7)
    q = cfg.to_qconfig(seed=42)
    assert isinstance(q, QConfig)
    assert (q.alpha, q.gamma, q.max_epochs, q.seed) == (0.2, 0.95, 7, 42)
    assert q.alpha_end == QConfig.alpha_end and q.epsilon_end == QConfig.epsilon_end


def test_tune_config_defaults_match_qconfig():
    base = QConfig()
    cfg = TuneConfig()
    assert dataclasses.replace(base, seed=0) == cfg.to_qconfig(seed=0)


def test_val_and_test_shift_seeds_are_disjoint():
    assert not set(VAL_SHIFT_SEEDS) & set(TEST_SHIFT_SEEDS)


def _spec(featured, mode="val", seed_index=0):
    return RunSpec(
        config=TuneConfig(max_epochs=3, shift_draws=2),
        seed_index=seed_index,
        base_seed=0,
        preset_id="8h",
        year=2025,
        mode=mode,
        featured=featured,
    )


def test_run_config_val_returns_expected_keys_and_is_deterministic(featured):
    first = run_config(_spec(featured))
    second = run_config(_spec(featured))
    for key in ("savings_cop", "threshold_savings_cop", "score", "epochs_run", "seconds"):
        assert key in first
    assert first["mode"] == "val"
    assert first["epochs_run"] <= 3
    deterministic = {k: v for k, v in first.items() if k != "seconds"}
    assert deterministic == {k: v for k, v in second.items() if k != "seconds"}


def test_run_config_depends_on_seed_index(featured):
    assert run_config(_spec(featured, seed_index=0))["run_seed"] != (
        run_config(_spec(featured, seed_index=1))["run_seed"]
    )


def test_run_config_test_mode_reports_all_policies(featured):
    out = run_config(_spec(featured, mode="test"))
    for key in (
        "shift_schedule_savings_cop",
        "night_day_savings_cop",
        "perfect_savings_cop",
        "q_vs_best_rule",
        "q_vs_perfect",
    ):
        assert key in out


def test_run_config_rejects_unknown_mode(featured):
    with pytest.raises(ValueError):
        run_config(_spec(featured, mode="bogus"))
