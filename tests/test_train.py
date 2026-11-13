"""Tests for `energy.rl.train`: job seeds, parallel/sequential equivalence and the CLI (D15).

Offline. Synthetic data covers all of 2025 so `scenario_split` works; QConfig uses
`max_epochs=3` and `shift_draws=2` to stay fast. Everything the workers need is built from
plain arguments (Windows spawns fresh processes, so no fixtures inside workers).
`run_jobs(..., shift_draws=, mix=)` take the D15 options as keyword arguments.
Run with `uv run pytest tests/test_train.py`.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem
from energy.rl.train import JobReport, TrainJob, job_seed, main, run_jobs

TINY = QConfig(max_epochs=3, patience=2, seed=0)
DRAWS = 2
REPORT_FIELDS = {
    "preset_id", "year", "epochs_run", "seconds", "q_savings_cop",
    "shift_schedule_savings_cop", "threshold_savings_cop", "night_day_savings_cop",
    "perfect_savings_cop", "best_rule", "q_vs_best_rule", "thresholds", "stem",
}
RULE_COLUMN = {
    "shift_schedule": "shift_schedule_savings_cop",
    "price_threshold": "threshold_savings_cop",
    "night_day": "night_day_savings_cop",
}


@pytest.fixture(scope="module")
def year_prices(synthetic_prices):
    """Hourly synthetic prices covering all of 2025 (371 days from Jan 1)."""
    return synthetic_prices(53, start="2025-01-01", noise=0.05, seed=5)


@pytest.fixture(scope="module")
def one_report(year_prices, tmp_path_factory):
    out = tmp_path_factory.mktemp("one_job")
    (report,) = run_jobs(
        [TrainJob("4h", 2025)], year_prices, TINY, out, workers=1, shift_draws=DRAWS
    )
    return report, out


def _comparable(report: JobReport) -> dict:
    data = dataclasses.asdict(report)
    data.pop("seconds")
    data["stem"] = report.stem.name
    return data


def _threshold_file(directory, preset_id: str, year: int = 2025):
    return directory / f"threshold_{preset_id}_{year}.json"


def test_job_seed_is_deterministic():
    assert job_seed(0, "4h", 2025) == job_seed(0, "4h", 2025)
    assert isinstance(job_seed(0, "4h", 2025), int)


def test_job_seed_differs_per_job_and_base_seed():
    seeds = {job_seed(0, p, y) for p in ("4h", "8h", "12h") for y in (2024, 2025)}
    assert len(seeds) == 6
    assert job_seed(0, "4h", 2025) != job_seed(1, "4h", 2025)


def test_job_report_fields():
    assert {f.name for f in dataclasses.fields(JobReport)} == REPORT_FIELDS


def test_run_jobs_workers_one_vs_two_identical(year_prices, tmp_path):
    jobs = [TrainJob("4h", 2025), TrainJob("8h", 2025)]
    dir1, dir2 = tmp_path / "w1", tmp_path / "w2"
    seq = run_jobs(jobs, year_prices, TINY, dir1, workers=1, shift_draws=DRAWS)
    par = run_jobs(jobs, year_prices, TINY, dir2, workers=2, shift_draws=DRAWS)

    assert [(r.preset_id, r.year) for r in seq] == [("4h", 2025), ("8h", 2025)]
    assert [(r.preset_id, r.year) for r in par] == [("4h", 2025), ("8h", 2025)]
    assert [_comparable(r) for r in seq] == [_comparable(r) for r in par]
    for preset_id in ("4h", "8h"):
        q1 = TabularPolicy.load(policy_stem(dir1, preset_id, 2025)).q
        q2 = TabularPolicy.load(policy_stem(dir2, preset_id, 2025)).q
        assert q1.dtype == np.float32
        assert q1.shape[3] == 4  # shift-type dimension
        assert np.array_equal(q1, q2)
        assert (
            _threshold_file(dir1, preset_id).read_bytes()
            == _threshold_file(dir2, preset_id).read_bytes()
        )


def test_run_jobs_is_repeatable(year_prices, tmp_path):
    jobs = [TrainJob("4h", 2025)]
    a = run_jobs(jobs, year_prices, TINY, tmp_path / "a", workers=1, shift_draws=DRAWS)
    b = run_jobs(jobs, year_prices, TINY, tmp_path / "b", workers=1, shift_draws=DRAWS)
    assert _comparable(a[0]) == _comparable(b[0])


def test_run_jobs_report_contents(one_report):
    report, out = one_report
    assert 1 <= report.epochs_run <= TINY.max_epochs
    assert report.seconds >= 0
    assert report.stem == policy_stem(out, "4h", 2025)
    assert report.stem.with_suffix(".npz").exists()
    assert report.stem.with_suffix(".json").exists()

    rules = {name: getattr(report, col) for name, col in RULE_COLUMN.items()}
    assert report.perfect_savings_cop >= max(rules.values()) - 1e-6
    assert report.best_rule in RULE_COLUMN
    assert rules[report.best_rule] == pytest.approx(max(rules.values()))
    assert report.q_vs_best_rule == pytest.approx(report.q_savings_cop / max(rules.values()))


def test_run_jobs_writes_threshold_file(one_report):
    report, out = one_report
    path = _threshold_file(out, "4h")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cheap"] < data["expensive"]
    # JobReport.thresholds is the tuned pair (cheap, expensive).
    assert tuple(report.thresholds) == pytest.approx((data["cheap"], data["expensive"]))


def test_q_table_metadata_records_shift_setup(one_report):
    report, _ = one_report
    meta = TabularPolicy.load(report.stem).metadata
    assert meta["shift_mix"] == "default"
    assert meta["shift_draws"] == DRAWS


def test_main_writes_policy_and_threshold_files(year_prices, tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    year_prices.to_csv(csv, index=False)
    out = tmp_path / "policies"
    code = main([
        "--prices", str(csv), "--out", str(out), "--max-epochs", "3",
        "--presets", "4h", "--scenarios", "2025", "--workers", "1",
        "--shift-draws", str(DRAWS), "--mix", "default",
    ])
    assert code == 0
    assert (out / "q_4h_2025.npz").exists()
    assert (out / "q_4h_2025.json").exists()
    assert (out / "threshold_4h_2025.json").exists()
    assert "4h" in capsys.readouterr().out
