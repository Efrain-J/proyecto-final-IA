"""Tests for `energy.app.model`: the pure logic behind the dashboard (no streamlit).

Offline: short synthetic prices, a tiny Q-table and threshold file in a temp dir, one or
two shift seeds. Run with `uv run pytest tests/test_app_model.py`.
"""
from __future__ import annotations

import inspect
from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy.app.model import (
    APP_POLICIES,
    MIX_LABELS,
    SCENARIOS,
    Kpis,
    PolicyRun,
    compute_kpis,
    n_weeks,
    run_policies,
    size_comparison,
    week_slice,
)
from energy.rl.battery import PRESETS, get_preset
from energy.rl.factory import MIXES, TEST_SHIFT_SEEDS, add_shifts, train_shift_seeds
from energy.rl.features import add_relative_price
from energy.rl.policies import PriceThreshold, threshold_path
from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem, train_q_learning
from energy.rl.simulate import resolve_policy, simulate

PRESET = "4h"
START, END = date(2025, 10, 6), date(2025, 10, 12)
SEEDS = (1000, 1001)


@pytest.fixture(scope="module")
def featured(synthetic_prices):
    # 10 weeks from 2025-09-01 cover the test week and its warm-up history.
    return add_relative_price(synthetic_prices(10, start="2025-09-01", noise=0.05, seed=4))


@pytest.fixture(scope="module")
def policies_dir(featured, tmp_path_factory):
    d = tmp_path_factory.mktemp("policies")
    frames = [add_shifts(featured, "default", s) for s in train_shift_seeds(1, 2)]
    res = train_q_learning(frames, get_preset(PRESET), QConfig(max_epochs=3, patience=2))
    TabularPolicy(res.q, res.discretizer, PRESET).save(policy_stem(d, PRESET, 2025))
    PriceThreshold(0.95, 1.1).save(threshold_path(d, PRESET, 2025))
    return d


@pytest.fixture(scope="module")
def runs(featured, policies_dir):
    return run_policies(featured, PRESET, START, END, seeds=SEEDS, policies_dir=policies_dir)


def make_run(policy_id: str, savings: float, cycles: float = 1.0) -> PolicyRun:
    idx = pd.date_range("2025-10-06", periods=3, freq="h")
    cum = pd.Series([0.0, savings / 2, savings], index=idx)
    return PolicyRun(
        policy_id=policy_id, policy_used=policy_id, savings_cop=savings,
        savings_min_cop=savings, savings_max_cop=savings, cycles=cycles,
        cum_savings=cum, hourly=pd.DataFrame({"timestamp": idx}),
    )


# --- constants ---


def test_app_policies_exact_and_exclude_oracle_and_no_battery():
    assert APP_POLICIES == ("q_learning", "price_threshold", "shift_schedule", "night_day")
    assert "perfect_foresight" not in APP_POLICIES
    assert "no_battery" not in APP_POLICIES


def test_mix_labels():
    assert MIX_LABELS == {
        "default": "Mixto",
        "mostly_double": "Principalmente turnos dobles",
        "mostly_single": "Principalmente un turno",
    }
    assert set(MIX_LABELS) <= set(MIXES)


def test_scenarios():
    assert SCENARIOS == {
        "2025 (normal)": (date(2025, 10, 1), date(2025, 12, 31)),
        "2024 (El Niño, precios altos)": (date(2024, 10, 1), date(2024, 12, 31)),
    }


# --- run_policies ---


def test_run_policies_keys_in_order(runs):
    assert list(runs) == list(APP_POLICIES)
    assert all(isinstance(r, PolicyRun) and r.policy_id == k for k, r in runs.items())


def test_run_policies_respects_policy_ids_order(featured, policies_dir):
    ids = ("night_day", "q_learning")
    out = run_policies(
        featured, PRESET, START, END, policy_ids=ids, seeds=(1000,), policies_dir=policies_dir
    )
    assert list(out) == list(ids)


@pytest.mark.parametrize("pid", APP_POLICIES)
def test_run_policies_matches_direct_simulation(featured, policies_dir, runs, pid):
    policy = resolve_policy(pid, PRESET, START.year, policies_dir)
    direct = [
        simulate(policy, START, END, PRESET, prices=add_shifts(featured, "default", s))
        for s in SEEDS
    ]
    sav = [d.summary.savings_cop for d in direct]
    run = runs[pid]
    assert run.savings_cop == pytest.approx(np.mean(sav))
    assert run.savings_min_cop == pytest.approx(min(sav))
    assert run.savings_max_cop == pytest.approx(max(sav))
    assert run.cycles == pytest.approx(np.mean([d.summary.cycles for d in direct]))
    assert run.policy_used == direct[0].summary.policy
    pd.testing.assert_frame_equal(
        run.hourly.reset_index(drop=True), direct[0].hourly.reset_index(drop=True)
    )


def test_run_policies_cum_savings_is_seed_mean(runs):
    run = runs["q_learning"]
    assert isinstance(run.cum_savings, pd.Series)
    assert run.cum_savings.iloc[-1] == pytest.approx(run.savings_cop)
    assert len(run.cum_savings) == len(run.hourly)
    assert list(pd.DatetimeIndex(run.cum_savings.index)) == list(run.hourly["timestamp"])


def test_run_policies_min_le_mean_le_max(runs):
    for r in runs.values():
        assert r.savings_min_cop <= r.savings_cop + 1e-9
        assert r.savings_cop <= r.savings_max_cop + 1e-9


def test_run_policies_missing_q_table_is_detectable(featured, tmp_path):
    out = run_policies(
        featured, PRESET, START, END, policy_ids=("q_learning",), seeds=(1000,),
        policies_dir=tmp_path,
    )
    assert out["q_learning"].policy_id == "q_learning"
    assert out["q_learning"].policy_used == "shift_schedule"


def test_run_policies_mix_changes_shift_pattern(featured, policies_dir):
    kw = dict(policy_ids=("shift_schedule",), seeds=(1000,), policies_dir=policies_dir)
    a = run_policies(featured, PRESET, START, END, mix="mostly_single", **kw)
    b = run_policies(featured, PRESET, START, END, mix="mostly_double", **kw)
    assert not a["shift_schedule"].hourly["shift_type"].equals(
        b["shift_schedule"].hourly["shift_type"]
    )


# --- size_comparison ---


def test_size_comparison_shape_and_columns(featured, tmp_path):
    presets = ("4h", "8h")
    table = size_comparison(
        featured, START, END, presets=presets, seeds=(1000,), policies_dir=tmp_path
    )
    assert list(table.columns) == ["preset_id", "policy_id", "savings_cop"]
    assert len(table) == len(presets) * len(APP_POLICIES)
    assert not table.duplicated(["preset_id", "policy_id"]).any()
    assert set(table["preset_id"]) == set(presets)
    assert set(table["policy_id"]) == set(APP_POLICIES)


def test_size_comparison_value_is_seed_mean(featured, tmp_path):
    table = size_comparison(
        featured, START, END, presets=("4h",), policy_ids=("night_day",), seeds=SEEDS,
        policies_dir=tmp_path,
    )
    expected = np.mean([
        simulate(
            "night_day", START, END, "4h", prices=add_shifts(featured, "default", s)
        ).summary.savings_cop
        for s in SEEDS
    ])
    assert table["savings_cop"].iloc[0] == pytest.approx(expected)


def test_size_comparison_defaults_cover_all_presets_and_test_seeds():
    params = inspect.signature(size_comparison).parameters
    assert params["presets"].default == tuple(PRESETS)
    assert params["seeds"].default == TEST_SHIFT_SEEDS


# --- compute_kpis ---


def test_kpis_best_rule_and_percentage():
    runs = {
        "q_learning": make_run("q_learning", 120.0, cycles=2.5),
        "price_threshold": make_run("price_threshold", 100.0),
        "shift_schedule": make_run("shift_schedule", 80.0),
        "night_day": make_run("night_day", 50.0),
    }
    k = compute_kpis(runs)
    assert isinstance(k, Kpis)
    assert k.q_savings_cop == 120.0
    assert k.best_rule_id == "price_threshold"
    assert k.best_rule_savings_cop == 100.0
    assert k.q_vs_best_rule_pct == pytest.approx(20.0)
    assert k.q_cycles == 2.5


def test_kpis_q_below_best_rule_is_negative():
    k = compute_kpis(
        {"q_learning": make_run("q_learning", 50.0), "night_day": make_run("night_day", 100.0)}
    )
    assert k.best_rule_id == "night_day"
    assert k.q_vs_best_rule_pct == pytest.approx(-50.0)


def test_kpis_tie_goes_to_first_in_app_order():
    runs = {
        "q_learning": make_run("q_learning", 10.0),
        "night_day": make_run("night_day", 70.0),
        "shift_schedule": make_run("shift_schedule", 70.0),
        "price_threshold": make_run("price_threshold", 70.0),
    }
    assert compute_kpis(runs).best_rule_id == "price_threshold"


@pytest.mark.parametrize("best", [0.0, -5.0])
def test_kpis_pct_none_when_best_not_positive(best):
    k = compute_kpis(
        {"q_learning": make_run("q_learning", 10.0), "night_day": make_run("night_day", best)}
    )
    assert k.best_rule_savings_cop == best
    assert k.q_vs_best_rule_pct is None


def test_kpis_missing_q_raises():
    with pytest.raises((KeyError, ValueError)):
        compute_kpis({"night_day": make_run("night_day", 10.0)})


def test_kpis_ignore_q_when_picking_best_rule():
    k = compute_kpis(
        {"q_learning": make_run("q_learning", 999.0), "night_day": make_run("night_day", 1.0)}
    )
    assert k.best_rule_id == "night_day"


# --- n_weeks / week_slice ---


def hourly(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": pd.date_range("2025-10-06", periods=n, freq="h"), "v": range(n)}
    )


@pytest.mark.parametrize("rows,expected", [(336, 2), (170, 2), (100, 1), (168, 1), (169, 2)])
def test_n_weeks(rows, expected):
    assert n_weeks(hourly(rows)) == expected


def test_week_slice_full_weeks():
    h = hourly(336)
    assert list(week_slice(h, 0)["v"]) == list(range(168))
    assert list(week_slice(h, 1)["v"]) == list(range(168, 336))


def test_week_slice_partial_last_week():
    assert list(week_slice(hourly(170), 1)["v"]) == [168, 169]


@pytest.mark.parametrize("rows,week", [(336, -1), (336, 2), (100, 1), (170, 2)])
def test_week_slice_out_of_range_raises(rows, week):
    with pytest.raises(ValueError):
        week_slice(hourly(rows), week)
