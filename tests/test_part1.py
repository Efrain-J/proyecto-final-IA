"""Tests for `energy.eval.part1`: the Part 1 results tables (D14, D15).

Offline: two years of synthetic prices, a tiny Q-table and threshold files in
`tmp_path`, one test shift pattern. Run with `uv run pytest tests/test_part1.py`.
"""
from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd
import pytest

from energy.eval.part1 import (
    RULE_IDS,
    evaluate,
    load_contenders,
    main,
    summarize,
)
from energy.rl.factory import add_shifts, train_shift_seeds
from energy.rl.features import add_relative_price
from energy.rl.policies import PriceThreshold, threshold_path
from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem, train_q_learning
from energy.rl.simulate import simulate

PRESET = "4h"
SEED = 1000


@pytest.fixture(scope="module")
def raw_prices(synthetic_prices):
    # 105 weeks from 2024-01-01 cover October-December of both 2024 and 2025.
    return synthetic_prices(105, start="2024-01-01", noise=0.05, seed=3)


@pytest.fixture(scope="module")
def featured(raw_prices):
    return add_relative_price(raw_prices)


@pytest.fixture(scope="module")
def policies_dir(tmp_path_factory, synthetic_prices):
    """Q-tables and thresholds for 4h, trained on 2024 and 2025 (tiny, not converged)."""
    out = tmp_path_factory.mktemp("policies")
    base = add_relative_price(synthetic_prices(4, noise=0.05, seed=2))
    frames = [add_shifts(base, "default", s) for s in train_shift_seeds(1, 2)]
    for year in (2024, 2025):
        res = train_q_learning(frames, PRESET, QConfig(max_epochs=3, patience=2, seed=year))
        TabularPolicy(res.q, res.discretizer, PRESET).save(policy_stem(out, PRESET, year))
    PriceThreshold(0.95, 1.10).save(threshold_path(out, PRESET, 2024))
    PriceThreshold(0.90, 1.20).save(threshold_path(out, PRESET, 2025))
    return out


# --- load_contenders ---


def test_load_contenders_lists_rules_optimum_and_trained_policies(policies_dir):
    found = {(c.policy_id, c.trained_on) for c in load_contenders(PRESET, [2024, 2025], policies_dir)}
    assert found == {
        ("shift_schedule", None),
        ("night_day", None),
        ("perfect_foresight", None),
        ("q_learning", 2024),
        ("q_learning", 2025),
        ("price_threshold", 2024),
        ("price_threshold", 2025),
    }


def test_load_contenders_reads_the_saved_cutoffs(policies_dir):
    by_key = {(c.policy_id, c.trained_on): c for c in load_contenders(PRESET, [2024, 2025], policies_dir)}
    assert by_key[("price_threshold", 2025)].policy.cheap == pytest.approx(0.90)
    assert by_key[("price_threshold", 2024)].policy.expensive == pytest.approx(1.10)


def test_load_contenders_skips_missing_files_with_warning(tmp_path, caplog):
    # A missing file must not fall back to another rule under the wrong label.
    with caplog.at_level(logging.WARNING):
        found = load_contenders(PRESET, [2025], tmp_path)
    assert {c.policy_id for c in found} == {"shift_schedule", "night_day", "perfect_foresight"}
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_contender_label_includes_training_year(policies_dir):
    labels = {c.label for c in load_contenders(PRESET, [2025], policies_dir)}
    assert "q_learning (2025)" in labels
    assert "shift_schedule" in labels


# --- evaluate ---


@pytest.fixture(scope="module")
def runs(featured, policies_dir):
    contenders = load_contenders(PRESET, [2024, 2025], policies_dir)
    return evaluate(featured, PRESET, 2025, contenders, seeds=(SEED, SEED + 1))


def test_evaluate_one_row_per_contender_and_seed(runs):
    assert len(runs) == 7 * 2
    assert set(runs["seed"]) == {SEED, SEED + 1}
    assert set(runs["test_year"]) == {2025}
    assert set(runs["preset_id"]) == {PRESET}
    for col in ("policy_id", "trained_on", "label", "savings_cop", "cycles", "invalid_actions"):
        assert col in runs.columns


def test_evaluate_perfect_foresight_is_an_upper_bound(runs):
    for _, group in runs.groupby("seed"):
        perfect = group.loc[group["policy_id"] == "perfect_foresight", "savings_cop"].item()
        assert (group["savings_cop"] <= perfect + 1e-6).all()


def test_evaluate_matches_simulate(featured, policies_dir, runs):
    # Same numbers as the app and the agent tool, which call simulate() directly.
    q = TabularPolicy.load(policy_stem(policies_dir, PRESET, 2025))
    shifted = add_shifts(featured, "default", SEED)
    expected = simulate(q, date(2025, 10, 1), date(2025, 12, 31), PRESET, prices=shifted)
    row = runs[(runs["label"] == "q_learning (2025)") & (runs["seed"] == SEED)]
    assert row["savings_cop"].item() == pytest.approx(expected.summary.savings_cop)


def test_evaluate_covers_october_to_december(featured, policies_dir):
    contenders = [c for c in load_contenders(PRESET, [], policies_dir) if c.policy_id == "night_day"]
    out = evaluate(featured, PRESET, 2024, contenders, seeds=(SEED,))
    assert out["start"].item() == date(2024, 10, 1)
    assert out["end"].item() == date(2024, 12, 31)


def test_evaluate_unknown_year_raises(featured, policies_dir):
    with pytest.raises(ValueError):
        evaluate(featured, PRESET, 1999, load_contenders(PRESET, [], policies_dir), seeds=(SEED,))


# --- summarize ---


def _row(policy_id, savings, trained_on=None, seed=SEED, preset="8h", test_year=2025):
    label = policy_id if trained_on is None else f"{policy_id} ({trained_on})"
    return {
        "preset_id": preset, "test_year": test_year, "policy_id": policy_id,
        "trained_on": trained_on, "label": label, "seed": seed, "savings_cop": savings,
    }


def test_summarize_compares_q_with_strongest_rule_over_both_cutoff_years():
    runs = pd.DataFrame(
        [
            _row("q_learning", 110.0, 2025),
            _row("q_learning", 90.0, 2024),
            _row("shift_schedule", 80.0),
            _row("night_day", 60.0),
            _row("price_threshold", 95.0, 2024),
            _row("price_threshold", 100.0, 2025),
            _row("perfect_foresight", 200.0),
        ]
    )
    out = summarize(runs).set_index("trained_on")
    assert len(out) == 2
    assert (out["best_rule"] == "price_threshold (2025)").all()
    assert out.loc[2025, "q_vs_best_rule"] == pytest.approx(1.10)
    assert out.loc[2024, "q_vs_best_rule"] == pytest.approx(0.90)
    assert out.loc[2025, "q_vs_perfect"] == pytest.approx(0.55)
    assert out.loc[2025, "perfect_savings_cop"] == pytest.approx(200.0)


def test_summarize_averages_over_seeds_and_reports_the_range():
    runs = pd.DataFrame(
        [
            _row("q_learning", 100.0, 2025, seed=1),
            _row("q_learning", 140.0, 2025, seed=2),
            _row("shift_schedule", 100.0, seed=1),
            _row("shift_schedule", 100.0, seed=2),
            _row("perfect_foresight", 200.0, seed=1),
            _row("perfect_foresight", 200.0, seed=2),
        ]
    )
    row = summarize(runs).iloc[0]
    assert row["q_savings_cop"] == pytest.approx(120.0)
    assert row["q_min_cop"] == pytest.approx(100.0)
    assert row["q_max_cop"] == pytest.approx(140.0)
    assert row["q_vs_best_rule"] == pytest.approx(1.20)


def test_summarize_tie_goes_to_first_rule_in_order():
    runs = pd.DataFrame(
        [_row("q_learning", 50.0, 2025)] + [_row(r, 40.0, 2025 if r == "price_threshold" else None) for r in RULE_IDS]
        + [_row("perfect_foresight", 100.0)]
    )
    assert summarize(runs).iloc[0]["best_rule"] == RULE_IDS[0]


def test_summarize_one_row_per_preset_test_year_and_training_year():
    rows = []
    for preset in ("4h", "8h"):
        for test_year in (2024, 2025):
            rows += [
                _row("q_learning", 10.0, 2024, preset=preset, test_year=test_year),
                _row("q_learning", 12.0, 2025, preset=preset, test_year=test_year),
                _row("shift_schedule", 10.0, preset=preset, test_year=test_year),
                _row("perfect_foresight", 20.0, preset=preset, test_year=test_year),
            ]
    out = summarize(pd.DataFrame(rows))
    assert len(out) == 8
    assert not out.duplicated(["preset_id", "test_year", "trained_on"]).any()


def test_summarize_non_positive_best_rule_gives_nan():
    runs = pd.DataFrame(
        [_row("q_learning", 10.0, 2025), _row("shift_schedule", 0.0), _row("perfect_foresight", 20.0)]
    )
    assert math.isnan(summarize(runs).iloc[0]["q_vs_best_rule"])


# --- main ---


def test_main_writes_results(raw_prices, policies_dir, tmp_path, capsys):
    prices_csv = tmp_path / "prices.csv"
    raw_prices.to_csv(prices_csv, index=False)
    out = tmp_path / "results"
    code = main(
        [
            "--presets", PRESET, "--years", "2024", "2025", "--seeds", "1",
            "--prices", str(prices_csv), "--policies", str(policies_dir), "--out", str(out),
        ]
    )
    assert code == 0
    runs = pd.read_csv(out / "runs.csv")
    summary = pd.read_csv(out / "summary.csv")
    assert len(runs) == 7 * 2  # 7 contenders x 2 test years x 1 seed
    assert len(summary) == 4  # 2 training years x 2 test years
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "q_vs_best_rule" in text or "Q ÷" in text
    assert "4h" in capsys.readouterr().out


def test_main_rejects_too_many_seeds(tmp_path):
    with pytest.raises(SystemExit):
        main(["--seeds", "99", "--out", str(tmp_path)])


# --- shift mixes ---


def test_evaluate_records_the_mix(featured, policies_dir):
    contenders = [c for c in load_contenders(PRESET, [], policies_dir) if c.policy_id == "night_day"]
    out = evaluate(featured, PRESET, 2025, contenders, seeds=(SEED,), mix="mostly_single")
    assert set(out["mix"]) == {"mostly_single"}


def test_summarize_separates_rows_per_mix():
    rows = []
    for mix, q in (("default", 12.0), ("mostly_single", 15.0)):
        for r in [_row("q_learning", q, 2025), _row("shift_schedule", 10.0), _row("perfect_foresight", 20.0)]:
            rows.append({**r, "mix": mix})
    out = summarize(pd.DataFrame(rows)).set_index("mix")
    assert len(out) == 2
    assert out.loc["default", "q_vs_best_rule"] == pytest.approx(1.2)
    assert out.loc["mostly_single", "q_vs_best_rule"] == pytest.approx(1.5)


def test_summarize_orders_batteries_by_size():
    rows = []
    for preset in ("12h", "4h", "8h"):
        rows += [
            _row("q_learning", 10.0, 2025, preset=preset),
            _row("shift_schedule", 10.0, preset=preset),
            _row("perfect_foresight", 20.0, preset=preset),
        ]
    assert list(summarize(pd.DataFrame(rows))["preset_id"]) == ["4h", "8h", "12h"]
