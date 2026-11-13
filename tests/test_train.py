"""Tests for `energy.rl.train`: job seeds, parallel/sequential equivalence and the CLI.

Offline. Synthetic data covers all of 2025 so `scenario_split` works; QConfig uses
`max_epochs=3` to stay fast. Everything the workers need is built from plain arguments
(Windows spawns fresh processes, so no fixtures inside workers).
Run with `uv run pytest tests/test_train.py`.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem
from energy.rl.train import JobReport, TrainJob, job_seed, main, run_jobs

TINY = QConfig(max_epochs=3, patience=2, seed=0)


@pytest.fixture(scope="module")
def year_prices(synthetic_prices):
    """Hourly synthetic prices covering all of 2025 (371 days from Jan 1)."""
    return synthetic_prices(53, start="2025-01-01", noise=0.05, seed=5)


def _comparable(report: JobReport) -> dict:
    data = dataclasses.asdict(report)
    data.pop("seconds")
    data["stem"] = report.stem.name
    return data


def test_job_seed_is_deterministic():
    assert job_seed(0, "2h", 2025) == job_seed(0, "2h", 2025)
    assert isinstance(job_seed(0, "2h", 2025), int)


def test_job_seed_differs_per_job_and_base_seed():
    seeds = {job_seed(0, p, y) for p in ("2h", "3h", "4h") for y in (2024, 2025)}
    assert len(seeds) == 6
    assert job_seed(0, "2h", 2025) != job_seed(1, "2h", 2025)


def test_run_jobs_workers_one_vs_two_identical(year_prices, tmp_path):
    jobs = [TrainJob("2h", 2025), TrainJob("3h", 2025)]
    dir1, dir2 = tmp_path / "w1", tmp_path / "w2"
    seq = run_jobs(jobs, year_prices, TINY, dir1, workers=1)
    par = run_jobs(jobs, year_prices, TINY, dir2, workers=2)

    assert [(r.preset_id, r.year) for r in seq] == [("2h", 2025), ("3h", 2025)]
    assert [(r.preset_id, r.year) for r in par] == [("2h", 2025), ("3h", 2025)]
    assert [_comparable(r) for r in seq] == [_comparable(r) for r in par]
    for preset_id in ("2h", "3h"):
        q1 = TabularPolicy.load(policy_stem(dir1, preset_id, 2025)).q
        q2 = TabularPolicy.load(policy_stem(dir2, preset_id, 2025)).q
        assert q1.dtype == np.float32
        assert np.array_equal(q1, q2)


def test_run_jobs_report_contents(year_prices, tmp_path):
    (report,) = run_jobs([TrainJob("2h", 2025)], year_prices, TINY, tmp_path, workers=1)
    assert 1 <= report.epochs_run <= TINY.max_epochs
    assert report.seconds >= 0
    assert report.perfect_savings_cop >= report.night_day_savings_cop
    assert report.stem == policy_stem(tmp_path, "2h", 2025)
    assert report.stem.with_suffix(".npz").exists()


def test_main_writes_policy_files(year_prices, tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    year_prices.to_csv(csv, index=False)
    out = tmp_path / "policies"
    code = main([
        "--prices", str(csv), "--out", str(out), "--max-epochs", "3",
        "--presets", "2h", "--scenarios", "2025", "--workers", "1",
    ])
    assert code == 0
    assert (out / "q_2h_2025.npz").exists()
    assert (out / "q_2h_2025.json").exists()
    assert "2h" in capsys.readouterr().out
